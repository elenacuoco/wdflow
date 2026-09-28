"""The whitening whitens the band-pass's passband and leaves its stop band alone.

The noise model is fitted on band-passed data, so its ``|A|`` also inverts the
band-pass's stop band. Held, the zero-phase whitenings apply ``|A|`` inside the
passband the band-pass's design defines and its edge values outside it.
"""
import numpy as np
import pytest
from scipy.signal import fftconvolve, lfilter, sosfreqz, welch

from wdf.filtering import sosfiltfilt
from wdf.processes.BandPassDownSampling import BandPassDownSampling
from wdf.processes.Whitening import Whitening
from wdf.processes.zero_phase_whitening import (
    DEFAULT_GRID,
    MagnitudeWhitening,
    ZeroPhaseWhitening,
    _both_ways,
    held_modulus,
    held_response,
    magnitude_taps,
)
from wdf.structures.array2SeqView import array2SeqView

SAMPLING, FACTOR = 4096, 2
FS = SAMPLING / FACTOR


class _Parameters:
    sampling, ResamplingFactor, resampling = SAMPLING, FACTOR, SAMPLING // FACTOR
    LowFrequencyCut, FilterOrder = 12.0, 10


@pytest.fixture(scope="module")
def front():
    """The worker's band-pass: Chebyshev II, order 10, 60 dB, 12 Hz to 0.9 Nyquist/R."""
    return BandPassDownSampling(_Parameters())


@pytest.fixture(scope="module")
def conditioned(front):
    """A red process with a resonance, through the band-pass, and its Burg model."""
    rng = np.random.default_rng(0)
    r = 0.99
    colour = np.convolve([1.0, -2 * r * np.cos(2 * np.pi * 300.0 / SAMPLING), r * r],
                         [1.0, -0.95])
    x = lfilter([1.0], colour, rng.standard_normal(200 * SAMPLING))
    y = sosfiltfilt(front.sos, x)[::FACTOR]
    model = Whitening(1000)
    model.ParametersEstimate(array2SeqView(0.0, FS, y.size).Fill(0.0, y))
    ar = np.array([model.ADE.GetAR(j) for j in range(1001)])
    return y, ar


def white_asd(samples, scale):
    """ASD of `samples / scale` in units of unit-variance white noise."""
    f, p = welch(samples / scale, fs=FS, nperseg=8192)
    return f, np.sqrt(p * FS / 2.0)


def band_std(samples, scale, band):
    """Standard deviation of `samples / scale` over `band`, as if it were all of it."""
    f, p = welch(samples / scale, fs=FS, nperseg=8192)
    return float(np.sqrt(np.mean(p[(f >= band[0]) & (f <= band[1])]) * FS / 2.0))


def test_the_passband_is_read_off_the_design(front):
    """The edges are where the forward-backward response |H|^2 is -3 dB."""
    low, high = front.passband()
    assert front.low_freq_hp < low < high < front.cutoff_frequency
    _, h = sosfreqz(front.sos, worN=[low, high], fs=front.sampling)
    assert 20 * np.log10(np.abs(h) ** 2) == pytest.approx([-3.0, -3.0], abs=0.02)
    inner = front.passband(-0.1)
    assert low < inner[0] < inner[1] < high


def test_the_held_modulus_is_the_modulus_in_band_and_flat_outside():
    freq = np.linspace(0.0, 1024.0, 1 << 16)
    modulus = 1.0 + 1e4 * (freq < 16.0) + 1e6 * (freq > 744.0) + np.sin(freq / 7.0) ** 2
    held = held_modulus(freq, modulus, (20.0, 700.0), blend=1.0)
    inside = (freq >= 21.0) & (freq <= 699.0)
    assert np.array_equal(held[inside], modulus[inside])
    assert np.ptp(held[freq < 20.0]) == 0.0 and np.ptp(held[freq > 700.0]) == 0.0
    # No jump at the edges: neighbouring bins, 1/64 Hz apart, differ by little.
    assert np.abs(np.diff(held)).max() < 0.01


def test_without_a_band_nothing_changes(conditioned):
    """No band is the full-band |A| and the previous scales, exactly."""
    _, ar = conditioned
    magnitude = MagnitudeWhitening(ar, 1, 0)
    assert np.array_equal(magnitude.taps, magnitude_taps(ar))
    assert magnitude.sigma == ar[0]
    root = ZeroPhaseWhitening(ar, 1, 0, order=1000)
    assert root.sigma == ar[0] * root.error


def test_the_magnitude_filter_whitens_the_band_at_unit_variance(front, conditioned):
    y, ar = conditioned
    band = front.passband()
    whitening = MagnitudeWhitening(ar, 1, 0, band=band, sampling=FS)
    whitened = fftconvolve(y, whitening.taps, mode="valid")
    f, asd = white_asd(whitened, ar[0])
    inside = (f >= band[0] + 1.0) & (f <= band[1] - 1.0)
    assert np.median(asd[inside]) == pytest.approx(1.0, abs=0.02)
    assert np.percentile(asd[inside], 95) < 1.08 and np.percentile(asd[inside], 5) > 0.92
    # The scale is the level in band, ar[0], not the variance over the circle,
    # which is the band's share of it.
    assert whitening.sigma == pytest.approx(ar[0], rel=1e-3)
    assert band_std(whitened, whitening.sigma, band) == pytest.approx(1.0, rel=0.01)
    assert np.std(whitened) < 0.9 * whitening.sigma


def test_the_root_follows_the_held_target_and_whitens_at_unit_variance(front, conditioned):
    y, ar = conditioned
    band = front.passband()
    whitening = ZeroPhaseWhitening(ar, 1, 0, order=3000, band=band, sampling=FS)
    _, target = held_response(ar, DEFAULT_GRID, band, FS)
    applied = np.abs(np.fft.rfft(whitening.polynomial, DEFAULT_GRID)) ** 2
    freq = np.fft.rfftfreq(DEFAULT_GRID, 1.0 / FS)
    inside = (freq >= band[0]) & (freq <= band[1])
    ratio = applied[inside] / (whitening.error * target[inside])
    assert np.abs(ratio - 1.0).max() < 0.02

    whitened = _both_ways(whitening.polynomial, y)[4 * 3000:-4 * 3000]
    f, asd = white_asd(whitened, ar[0] * whitening.error)
    inside = (f >= band[0] + 1.0) & (f <= band[1] - 1.0)
    assert np.median(asd[inside]) == pytest.approx(1.0, abs=0.02)
    assert whitening.sigma == pytest.approx(ar[0] * whitening.error, rel=0.01)
    assert band_std(whitened, whitening.sigma, band) == pytest.approx(1.0, rel=0.01)


@pytest.mark.parametrize("name", ["magnitude", "root"])
def test_the_stop_band_is_left_to_the_band_pass(front, conditioned, name):
    """Full band, the stop band comes out whitened like the band; held, it stays down."""
    y, ar = conditioned
    band = front.passband()

    def whiten(**held):
        if name == "magnitude":
            return fftconvolve(y, MagnitudeWhitening(ar, 1, 0, **held).taps, mode="valid")
        root = ZeroPhaseWhitening(ar, 1, 0, order=1000, **held)
        return _both_ways(root.polynomial, y)[8000:-8000] / root.error

    stop = None
    for held, expected in (({}, "white"), (dict(band=band, sampling=FS), "down")):
        f, asd = white_asd(whiten(**held), ar[0])
        stop = (f > band[1] + 50.0) & (f < 0.5 * FS - 10.0)
        level = np.median(asd[stop])
        if expected == "white":
            assert level > 0.3
        else:
            assert level < 1e-3


def test_the_held_stream_is_the_stream_whitened_at_once(front, conditioned):
    """Blocks joined are one convolution with the whole stream, joins included."""
    from test_magnitude_whitening import stream

    y, ar = conditioned
    whitening = MagnitudeWhitening(ar, 2000, 0, band=front.passband(), sampling=FS)
    whitening.SetOutputSize(2000, whitening.latency)
    x = y[:int(60 * FS)]
    streamed, _ = stream(whitening, x, [777, 5000, 5001, 40000, 90001])
    reference = fftconvolve(x, whitening.taps, mode="full")[whitening.latency:][:streamed.size]
    settled = slice(whitening.latency, None)
    assert np.abs(streamed[settled] - reference[settled]).max() < 1e-12 * np.abs(reference).max()


@pytest.mark.parametrize("hold", [True, False])
def test_the_worker_records_whether_it_held(tmp_outdir, hold):
    import glob
    import json

    from conftest import run_segment_process

    triggers = run_segment_process(tmp_outdir, HoldOutsideBand=hold)
    assert len(triggers) > 0
    with open(glob.glob(f"{tmp_outdir}**/parametersUsed-Win*.json", recursive=True)[0]) as fh:
        used = json.load(fh)
    assert used["HoldOutsideBand"] is hold
    if hold:
        low, high = used["Passband"]
        assert used["LowFrequencyCut"] < low < high < 0.5 * used["resampling"]
        assert used["BandEdgeBlend"] == 1.0 and used["PassbandLevel"] == -3.0
        assert used["sigma"] == used["sigmaWhitened"]
    else:
        assert "Passband" not in used
