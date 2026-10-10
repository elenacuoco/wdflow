"""The NumPy local cosine (p4TSA's LocalCos<M> layout) and the dechirped local
cosine: geometry, inverse, noise statistics, concentration, and the limits of
the dechirp (not orthonormal, analytic signal not circular)."""
import numpy as np
import pytest
from scipy.signal import hilbert

from wdf.analysis.dechirp import (alpha_grid, analytic_signal, dechirp, dechirped_geometry,
                                  dechirped_local_cosine, inverse_dechirped_local_cosine,
                                  rechirp)
from wdf.analysis.local_cosine import inverse_local_cosine, local_cosine, local_cosine_geometry

FS, N, M = 2048.0, 1024, 128


def test_local_cosine_is_orthonormal_and_inverted():
    x = np.random.default_rng(0).standard_normal((5, N))
    c = local_cosine(x, M)
    assert np.sum(c ** 2) == pytest.approx(np.sum(x ** 2), rel=1e-12)
    np.testing.assert_allclose(inverse_local_cosine(c, M), x, atol=1e-12)
    np.testing.assert_allclose(local_cosine(x[0], M), c[0])


def test_local_cosine_equals_the_core():
    pytest.importorskip("py4tsa")
    from py4tsa.tsa import WaveletTransform
    from wdf.structures.array2SeqView import array2SeqView

    x = np.random.default_rng(1).standard_normal(N)
    view = array2SeqView(0.0, FS, N)
    view.Fill(0.0, x.copy())
    WaveletTransform(N, WaveletTransform.LocalCos, 7).Forward(view.SV)
    core = np.array([view.SV.GetY(0, i) for i in range(N)])
    np.testing.assert_allclose(local_cosine(x, M), core, atol=1e-12)


def test_local_cosine_geometry_is_the_packet_layout():
    from wdf.analysis.wavelets import coeff_freq_bands, coeff_time_bounds

    g = local_cosine_geometry(N, M, FS)
    np.testing.assert_allclose(g["f_lo"], coeff_freq_bands(N, FS, "LocalCos128")[0])
    np.testing.assert_allclose(g["t_hi"], coeff_time_bounds(N, FS, "LocalCos128")[1])


def test_a_tone_lands_in_its_bin():
    t = np.arange(N) / FS
    c = local_cosine(np.cos(2 * np.pi * 204.0 * t), M)
    g = local_cosine_geometry(N, M, FS)
    top = np.argsort(c ** 2)[::-1][:8]
    width = FS / (2 * M)
    assert np.all((g["f_lo"][top] <= 204.0 + width) & (g["f_hi"][top] >= 204.0 - width))


def test_alpha_zero_is_the_local_cosine():
    x = np.random.default_rng(2).standard_normal(N)
    np.testing.assert_array_equal(dechirped_local_cosine(x, FS, 0.0, M), local_cosine(x, M))


def test_white_noise_stays_unit_variance():
    x = np.random.default_rng(3).standard_normal((200, N))
    c = dechirped_local_cosine(x, FS, 40.0, M)
    assert np.var(c) == pytest.approx(1.0, abs=0.02)


def test_the_matched_sweep_concentrates_a_chirp():
    """A chirp through 62.5 Hz sweeping 40 Hz/s over 2 s, in 0.5 s x 1 Hz
    tiles: with the matched alpha nearly all its energy falls into 8 tiles (one
    row, four segments), against a fifth without, or with the opposite sweep."""
    n, m = 4096, 1024
    t = np.arange(n) / FS - (n - 1) / (2 * FS)
    s = np.cos(2 * np.pi * (62.5 * t + 0.5 * 40.0 * t ** 2))

    def top8(alpha):
        c = np.sort(dechirped_local_cosine(s, FS, alpha, m) ** 2)[::-1]
        return c[:8].sum() / c.sum()

    assert top8(40.0) > 0.95
    assert top8(0.0) < 0.3 and top8(-40.0) < 0.3


def test_the_inverse_rechirps():
    """Away from 0 Hz and the Nyquist frequency the re-chirp inverts the map."""
    t = np.arange(N) / FS
    x = np.sin(2 * np.pi * (150.0 * t + 0.5 * 120.0 * t ** 2)) * np.hanning(N)
    c = dechirped_local_cosine(x, FS, 120.0, M)
    back = inverse_dechirped_local_cosine(c, FS, 120.0, M)
    assert np.linalg.norm(back - x) / np.linalg.norm(x) < 1e-2
    np.testing.assert_allclose(rechirp(dechirp(x, FS, 120.0), FS, 120.0), x, atol=1e-2)


def test_it_is_not_orthonormal_only_where_the_sweep_folds():
    """The singular values of the map leave 1 only for the bands the sweep
    moves across 0 Hz or the Nyquist frequency: about |alpha| T / 2 on each
    side, T the window; the rest are 1."""
    alpha = 20.0
    Y = dechirped_local_cosine(np.eye(N), FS, alpha, M)
    sv = np.linalg.svd(Y, compute_uv=False)
    off = np.abs(sv - 1.0) > 0.05
    folded_bins = 2 * 2 * int(np.ceil(abs(alpha) * (N / FS) / 2 / (FS / N)))
    assert 0 < off.sum() <= 4 * folded_bins
    assert np.mean(~off) > 0.95


def test_the_analytic_signal_does_not_wrap():
    """An impulse at the start of the window: the circular Hilbert transform
    puts part of its response at the far end, the padded one does not."""
    x = np.zeros(N)
    x[3] = 1.0
    far = slice(N - 200, N)
    assert np.sum(np.imag(hilbert(x))[far] ** 2) > 1e-3
    assert np.sum(np.imag(analytic_signal(x))[far] ** 2) < 1e-3 * np.sum(np.imag(hilbert(x))[far] ** 2) + 1e-6
    np.testing.assert_allclose(np.real(analytic_signal(x)), x, atol=1e-12)


def test_the_tiles_are_parallelograms():
    alpha = 100.0
    g = dechirped_geometry(N, M, FS, alpha)
    t_mid = (N - 1) / (2 * FS)
    i = 3 * (N // M) + 1
    assert g["corners"][i, 0, 1] == pytest.approx(g["f_lo"][i] + alpha * (g["t_lo"][i] - t_mid))
    assert g["corners"][i, 2, 1] == pytest.approx(g["f_hi"][i] + alpha * (g["t_hi"][i] - t_mid))
    assert g["bf_lo"][i] <= g["corners"][i, :, 1].min() + 1e-9
    assert g["bf_hi"][i] >= g["corners"][i, :, 1].max() - 1e-9
    assert np.all(g["bf_lo"] >= 0.0) and np.all(g["bf_hi"] <= FS / 2)


def test_the_alpha_grid_is_set_by_the_tile():
    grid = alpha_grid(M / FS, 550.0)
    assert grid[1] - grid[0] == pytest.approx(1.0 / (2 * (M / FS) ** 2))
    assert grid[0] == -grid[-1] and 0.0 in grid and grid[-1] <= 550.0
    assert list(alpha_grid(0.5, 14.0, step=2.0)) == list(np.arange(-14.0, 15.0, 2.0))
    with pytest.raises(ValueError):
        alpha_grid(0.0, 10.0)
