import numpy as np
import pytest
from scipy.signal import lfilter, welch

from pytsa.tsa import LatticeFilter
from wdf.processes.zero_phase_whitening import (
    ZeroPhaseWhitening,
    levinson,
    sqrt_ar_polynomial,
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
