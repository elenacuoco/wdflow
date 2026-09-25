import numpy as np
import pytest
from scipy.signal import lfilter, welch

from py4tsa.tsa import LatticeFilter
from wdf.processes.zero_phase_whitening import (
    ZeroPhaseWhitening,
    levinson,
    sqrt_ar_polynomial,
    sqrt_polynomial_from_spectrum,
    sqrt_lattice_view,
)
from wdf.structures.array2SeqView import array2SeqView

FS = 2048.0
ORDER = 128


def coloured_ar_model(order=40, seed=0):
    """An AR model with the coefficient layout ArBurgEstimator uses."""
    rng = np.random.default_rng(seed)
    poles = 0.9 * rng.uniform(-1.0, 1.0, order)
    polynomial = np.poly(poles)
    polynomial = polynomial / polynomial[0]
    return np.concatenate([[1.0], -polynomial[1:]])


def lattice_filter_output(view, x):
    data = array2SeqView(0.0, FS, len(x))
    data = data.Fill(0.0, np.asarray(x, dtype=float).copy())
    out = array2SeqView(0.0, FS, len(x))
    out = out.Fill(0.0, np.zeros(len(x)))
    lf = LatticeFilter(view)
    lf.init(view)
    lf(data, out)
    return np.array([out.GetY(0, i) for i in range(len(x))])


def forward_backward(a, x):
    return lfilter(a, [1.0], lfilter(a, [1.0], x)[::-1])[::-1]


def test_levinson_recovers_a_known_ar_model():
    a = np.array([1.0, -0.6, 0.2])
    impulse = lfilter([1.0], a, np.eye(1, 4096, 0).ravel())
    autocorrelation = np.correlate(impulse, impulse, mode="full")[len(impulse) - 1:]
    fitted, _, _ = levinson(autocorrelation, 2)
    assert fitted == pytest.approx(a, abs=1e-6)


def test_squared_magnitude_of_the_square_root_matches_the_model():
    ar = coloured_ar_model()
    a_half, _, _ = sqrt_ar_polynomial(ar, order=ORDER)

    grid = 4096
    model = np.abs(np.fft.rfft(np.concatenate([[1.0], -ar[1:]]), grid))
    half = np.abs(np.fft.rfft(a_half, grid))

    ratio = half * half / model
    assert ratio.std() / ratio.mean() < 0.02


def test_lattice_view_reproduces_the_polynomial():
    """The reflection coefficients must drive p4TSA's own filter, not just numpy."""
    ar = coloured_ar_model()
    a_half, _, _ = sqrt_ar_polynomial(ar, order=ORDER)

    x = np.random.default_rng(1).standard_normal(8000)
    lattice = lattice_filter_output(sqrt_lattice_view(ar, order=ORDER), x)
    direct = lfilter(a_half, [1.0], x)

    settled = slice(2 * ORDER, None)
    difference = np.linalg.norm(lattice[settled] - direct[settled])
    assert difference / np.linalg.norm(direct[settled]) < 1e-6


def test_forward_backward_whitens_coloured_noise():
    ar = coloured_ar_model()
    a_half, error, _ = sqrt_ar_polynomial(ar, order=ORDER)

    rng = np.random.default_rng(2)
    coloured = lfilter([1.0], np.concatenate([[1.0], -ar[1:]]), rng.standard_normal(200000))

    whitened = forward_backward(a_half, coloured)[4000:-4000]

    f, p = welch(whitened, fs=FS, nperseg=2048)
    band = (f > 20.0) & (f < 900.0)
    flatness = np.exp(np.mean(np.log(p[band]))) / np.mean(p[band])
    assert flatness > 0.95

    # The driving noise has unit variance here, so the predicted output scale
    # is the final prediction error alone.
    assert np.std(whitened) == pytest.approx(error, rel=0.05)


def test_latency_is_bounded_by_the_filter_order():
    """The backward pass is an FIR sum over at most `order` future samples.

    `order` is the bound the construction guarantees; how much of it is really
    needed depends on where the coefficients of the fitted model die out, so the
    test asserts the bound is exact and that a short lookahead is not enough.
    """
    ar = coloured_ar_model()
    a_half, _, _ = sqrt_ar_polynomial(ar, order=ORDER)

    x = np.random.default_rng(3).standard_normal(20000)
    forward = lfilter(a_half, [1.0], x)
    reference = lfilter(a_half, [1.0], forward[::-1])[::-1]
    keep = slice(2000, len(x) - 2000)

    def truncated(lookahead):
        taps = a_half[:lookahead + 1]
        out = np.convolve(forward, taps[::-1])[lookahead:lookahead + len(forward)]
        return np.linalg.norm(out[keep] - reference[keep]) / np.linalg.norm(reference[keep])

    assert truncated(ORDER) < 1e-12
    assert truncated(8) > 1e-3


def test_whitening_object_exposes_its_scale_and_latency():
    ar = coloured_ar_model()
    whitening = ZeroPhaseWhitening(ar, output_size=2048, order=ORDER)

    assert whitening.latency == ORDER
    assert whitening.sigma == pytest.approx(whitening.error * ar[0])
    assert len(whitening.polynomial) == ORDER + 1


def test_forward_backward_does_not_shift_the_signal():
    ar = coloured_ar_model()
    a_half, _, _ = sqrt_ar_polynomial(ar, order=ORDER)

    t = (np.arange(8192) - 4096) / FS
    h = np.exp(-((t / 0.01) ** 2) / 2.0) * np.cos(2 * np.pi * 150.0 * t)

    def centroid(x):
        e = np.asarray(x) ** 2
        return float((np.arange(len(e)) * e).sum() / e.sum())

    causal = lfilter(np.concatenate([[1.0], -ar[1:]]), [1.0], h)
    zero_phase = forward_backward(a_half, h)

    assert abs(centroid(zero_phase) - centroid(h)) / FS < 1e-4
    assert abs(centroid(causal) - centroid(h)) / FS > 1e-3


def test_the_spectrum_is_held_flat_outside_the_band():
    """What is outside the band is not fitted, it is held at the edge."""
    from wdf.processes.zero_phase_whitening import held_outside

    freq = np.linspace(0.0, 1024.0, 513)
    psd = 1.0 + freq                      # rising, so the edges are distinct
    held = held_outside(freq, psd, (100.0, 800.0))

    inside = (freq >= 100.0) & (freq < 800.0)
    assert np.allclose(held[inside], psd[inside])
    assert np.all(held[freq < 100.0] == held[inside][0])
    assert len(np.unique(held[freq >= 800.0])) == 1


def test_the_spectral_fit_whitens_what_it_was_measured_on():
    """The filter fitted to a measured spectrum flattens that spectrum.

    Coloured noise, its own spectrum measured, the filter fitted to it and run
    both ways: the result sits at the white level of a stream of standard
    deviation `sigma`, which for a one-sided density is sigma sqrt(2/fs).
    """
    from scipy.signal import lfilter, welch
    from wdf.processes.zero_phase_whitening import _both_ways

    rng = np.random.default_rng(11)
    coloured = lfilter([1.0], [1.0, -0.9, 0.2], rng.standard_normal(200000))
    whitening = ZeroPhaseWhitening.from_spectrum(
        coloured, FS, 2048, 0, order=256, grid=1 << 14, band=(8.0, FS / 2))

    whitened = _both_ways(whitening.polynomial, coloured)[1000:-1000]
    freq, power = welch(whitened, fs=FS, nperseg=8192)
    white = np.sqrt(power) / (whitening.sigma * np.sqrt(2.0 / FS))
    band = (freq >= 16.0) & (freq <= 0.45 * FS)

    assert 0.9 < np.median(white[band]) < 1.1
    assert np.std(np.log10(white[band])) < 0.05


def test_the_two_fits_agree_where_the_model_is_a_good_one():
    """Burg and the spectrum are two ways to the same filter.

    On noise an autoregressive model describes well, the filter fitted to the
    model and the filter fitted to the measured spectrum have the same
    response to within the scatter of the spectral estimate. Where the model is
    a poor one they part company, which is the reason the second path exists.
    """
    from scipy.signal import lfilter, welch

    rng = np.random.default_rng(3)
    coloured = lfilter([1.0], [1.0, -0.7], rng.standard_normal(200000))

    order, grid, bins = 64, 1 << 14, 4096
    ar = np.concatenate([[1.0], [0.7], np.zeros(order - 1)])
    from_model, _, _ = sqrt_ar_polynomial(ar, order=order, grid=grid)

    freq, psd = welch(coloured, fs=FS, nperseg=8192, average="median")
    from_spectrum, _, _ = sqrt_polynomial_from_spectrum(
        freq, psd, order, grid=grid, band=(8.0, FS / 2))

    axis = np.fft.rfftfreq(bins, 1.0 / FS)
    inband = (axis >= 32.0) & (axis <= 0.45 * FS)
    ratio = (np.abs(np.fft.rfft(np.asarray(from_spectrum), bins))
             / np.abs(np.fft.rfft(np.asarray(from_model), bins)))
    ratio = ratio[inband] / np.median(ratio[inband])

    assert np.std(np.log10(ratio)) < 0.05

