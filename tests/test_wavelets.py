import numpy as np
import pytest

from wdf.analysis.wavelets import (
    coeff_freq_bands,
    coeff_levels,
    coeff_time_bounds,
    donoho_johnstone_threshold,
    plot_wavelet_tiles,
    wavelet_coeff_tiles,
    wavelet_energy_snr,
)

NCOEFF = 512
FS = 2048.0


def test_coeff_levels_partition_is_exhaustive():
    level, position = coeff_levels(NCOEFF)
    assert level[0] == -1
    J = int(np.log2(NCOEFF))
    for j in range(J):
        idx = np.where(level == j)[0]
        assert len(idx) == 2 ** j
        assert idx.min() == 2 ** j and idx.max() == 2 ** (j + 1) - 1
        assert (position[idx] == np.arange(2 ** j)).all()


def test_freq_bands_span_zero_to_nyquist_contiguously():
    f_lo, f_hi = coeff_freq_bands(NCOEFF, FS)
    level, _ = coeff_levels(NCOEFF)
    J = int(np.log2(NCOEFF))
    assert f_lo[0] == 0.0
    assert f_hi[level == J - 1][0] == FS / 2
    # band edges double each level and are contiguous level-to-level
    for j in range(J - 1):
        this_hi = f_hi[level == j][0]
        next_lo = f_lo[level == j + 1][0]
        assert this_hi == pytest.approx(next_lo)
    assert f_hi[0] == pytest.approx(f_lo[level == 0][0])


def test_time_bounds_tile_window_without_gaps_or_overlap():
    t_lo, t_hi = coeff_time_bounds(NCOEFF, FS)
    level, position = coeff_levels(NCOEFF)
    window_s = NCOEFF / FS
    assert t_lo[0] == 0.0 and t_hi[0] == pytest.approx(window_s)
    for j in range(int(np.log2(NCOEFF))):
        idx = np.where(level == j)[0]
        order = idx[np.argsort(position[idx])]
        assert t_lo[order[0]] == pytest.approx(0.0)
        assert t_hi[order[-1]] == pytest.approx(window_s)
        for a, b in zip(order[:-1], order[1:]):
            assert t_hi[a] == pytest.approx(t_lo[b])


def test_wavelet_coeff_tiles_shape_and_ordering():
    wt = np.random.default_rng(0).normal(size=NCOEFF)
    tiles = wavelet_coeff_tiles(wt, FS)
    assert len(tiles) == NCOEFF
    for (t_lo, t_hi, f_lo, f_hi, mag) in tiles:
        assert t_hi >= t_lo and f_hi >= f_lo and mag >= 0


def test_donoho_johnstone_threshold_matches_closed_form():
    sigma, n = 2.5, 512
    assert donoho_johnstone_threshold(sigma, n) == pytest.approx(sigma * np.sqrt(2 * np.log(n)))
    # scales linearly with sigma, monotonically (slowly) with n
    assert donoho_johnstone_threshold(2 * sigma, n) == pytest.approx(2 * donoho_johnstone_threshold(sigma, n))
    assert donoho_johnstone_threshold(sigma, 4 * n) > donoho_johnstone_threshold(sigma, n)


def test_wavelet_energy_snr_counts_every_surviving_coefficient():
    """`wt` reaches Python already thresholded by p4TSA's WaveletThreshold
    (sub-threshold coefficients are exactly 0), so the energy sum must take
    every non-zero coefficient -- including small ones, which in the default
    soft mode are legitimate survivors shrunk by the threshold.
    """
    sigma = 1.0
    thresh = donoho_johnstone_threshold(sigma, NCOEFF)
    wt = np.zeros(NCOEFF)          # C++ zeroed everything sub-threshold
    wt[10] = thresh * 3            # a loud survivor
    wt[20] = thresh * 1e-3         # a soft-shrunk survivor: tiny but real

    result = wavelet_energy_snr(wt, sigma)
    assert result["n_nonzero"] == 2
    expected = (thresh * 3) ** 2 + (thresh * 1e-3) ** 2
    assert result["energy"] == pytest.approx(expected, rel=1e-6)
    assert result["snr"] == pytest.approx(np.sqrt(result["energy"]) / sigma, rel=1e-6)


def test_wavelet_energy_snr_does_not_reapply_the_threshold():
    """Regression guard: re-applying Donoho-Johnstone here discarded 69-90% of
    each real trigger's energy (measured on GW250114) and left most triggers at
    snr == 0, because soft thresholding upstream shrinks survivors below it.
    """
    sigma = 1.0
    thresh = donoho_johnstone_threshold(sigma, NCOEFF)
    wt = np.zeros(NCOEFF)
    wt[[5, 6, 7]] = thresh * 0.1   # all survivors, all below a second DJ cut

    result = wavelet_energy_snr(wt, sigma)
    assert result["n_nonzero"] == 3
    assert result["energy"] > 0
    assert result["snr"] > 0


def test_wavelet_energy_snr_energy_is_additive_across_coefficients():
    sigma = 1.0
    thresh = donoho_johnstone_threshold(sigma, NCOEFF)
    wt = np.zeros(NCOEFF)
    wt[[10, 20, 30]] = thresh * np.array([2.0, 3.0, 4.0])

    result = wavelet_energy_snr(wt, sigma)
    assert result["n_nonzero"] == 3
    expected_energy = sum((thresh * m) ** 2 for m in (2.0, 3.0, 4.0))
    assert result["energy"] == pytest.approx(expected_energy, rel=1e-6)


def test_plot_wavelet_tiles_draws_top_fraction():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wt = np.random.default_rng(3).normal(size=NCOEFF)
    fig, ax = plt.subplots()
    n_drawn = plot_wavelet_tiles(ax, wt, FS, t0=1000.0, top_frac=0.1)
    assert n_drawn == pytest.approx(NCOEFF * 0.1, abs=NCOEFF * 0.02)
    assert len(ax.patches) == n_drawn
    plt.close(fig)
