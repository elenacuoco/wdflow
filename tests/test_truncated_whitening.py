"""The truncated zero-phase whitening, and the level every zero-phase path gives.

`MagnitudeWhitening` with a duration is gwpy's inverse-spectrum truncation with
the Burg model's spectrum: the model at the resolution of the filter, inverted,
held outside the passband and Hann-truncated to the duration. Every zero-phase
path divides its output by its level in band, so that white-in-band noise comes
out at unit density there and unit standard deviation over the band.
"""
import glob
import json

import numpy as np
import pytest
from scipy.signal import fftconvolve, lfilter

from test_hold_outside_band import FACTOR, FS, SAMPLING, _Parameters, band_std, white_asd
from wdf.filtering import sosfiltfilt
from wdf.processes.BandPassDownSampling import BandPassDownSampling
from wdf.processes.Whitening import Whitening
from wdf.processes.zero_phase_whitening import (
    DEFAULT_TRUNCATION_S,
    MagnitudeWhitening,
    ZeroPhaseWhitening,
    _both_ways,
    truncated_taps,
)
from wdf.structures.array2SeqView import array2SeqView


@pytest.fixture(scope="module")
def front():
    return BandPassDownSampling(_Parameters())


def _conditioned(front, colour, seed, order=300, seconds=600):
    rng = np.random.default_rng(seed)
    x = lfilter([1.0], colour, rng.standard_normal(seconds * SAMPLING))
    y = sosfiltfilt(front.sos, x)[::FACTOR]
    model = Whitening(order)
    fit = y[:int(100 * FS)]
    model.ParametersEstimate(array2SeqView(0.0, FS, fit.size).Fill(0.0, fit))
    ar = np.array([model.ADE.GetAR(j) for j in range(order + 1)])
    return y, ar


@pytest.fixture(scope="module")
def white(front):
    """White noise through the band-pass: white in band."""
    return _conditioned(front, [1.0], 1)


@pytest.fixture(scope="module")
def coloured(front):
    """A red process with a resonance at 300 Hz, through the band-pass."""
    r = 0.99
    colour = np.convolve([1.0, -2 * r * np.cos(2 * np.pi * 300.0 / SAMPLING), r * r],
                         [1.0, -0.95])
    return _conditioned(front, colour, 2)


def _level_and_std(whitened, sigma, band):
    f, asd = white_asd(whitened, sigma)
    inside = (f >= band[0] + 2.0) & (f <= band[1] - 2.0)
    return float(np.median(asd[inside])), band_std(whitened, sigma, band)


def _whiten(name, y, ar, band):
    """The stream through one zero-phase path, over its own sigma."""
    edge = int(20 * FS)
    if name == "root":
        w = ZeroPhaseWhitening(ar, 1, 0, order=1000, band=band, sampling=FS)
        return _both_ways(w.polynomial, y)[edge:-edge] / w.sigma
    if name == "magnitude":
        w = MagnitudeWhitening(ar, 1, 0, band=band, sampling=FS)
    elif name == "truncated":
        w = MagnitudeWhitening(ar, 1, 0, band=band, sampling=FS, duration=4.0)
    elif name == "spectrum":
        # 300 s: a filter divides by its own estimate of the spectrum, and on
        # data it was not measured on that leaves E[S / S_hat] > 1 -- 1.2 %
        # in level from 100 s, 0.1 % from 300 s.
        w = MagnitudeWhitening.from_spectrum(y[:int(300 * FS)], FS, 1, 0, band=band)
    else:  # spectrum, truncated: gwpy's own construction
        w = MagnitudeWhitening.from_spectrum(y[:int(300 * FS)], FS, 1, 0, band=band,
                                             taper=1.0, nperseg=int(4 * FS))
    return fftconvolve(y, w.taps, mode="valid")[edge:-edge] / w.sigma


@pytest.mark.parametrize("data", ["white", "coloured"])
@pytest.mark.parametrize("name", ["magnitude", "root", "truncated", "spectrum",
                                  "spectrum-truncated"])
def test_every_zero_phase_path_is_at_unit_level_in_band(front, request, data, name):
    y, ar = request.getfixturevalue(data)
    band = front.passband()
    level, std = _level_and_std(_whiten(name, y, ar, band), 1.0, band)
    assert level == pytest.approx(1.0, rel=0.01)
    assert std == pytest.approx(1.0, rel=0.01)


def test_the_truncated_filter_is_the_duration_long_and_symmetric(white):
    _, ar = white
    taps = truncated_taps(ar, DEFAULT_TRUNCATION_S, FS)
    assert taps.size == int(DEFAULT_TRUNCATION_S * FS) + 1
    assert np.array_equal(taps, taps[::-1])
    assert taps[0] == 0.0 and taps[-1] == 0.0          # Hann to the last tap
    w = MagnitudeWhitening(ar, 1, 0, sampling=FS, duration=2.0)
    assert w.latency == int(1.0 * FS) and w.duration == 2.0


def test_a_duration_needs_the_sampling(white):
    with pytest.raises(ValueError):
        MagnitudeWhitening(white[1], 1, 0, duration=4.0)


def test_the_truncated_stream_is_the_stream_whitened_at_once(front, coloured):
    """Blocks joined are one convolution with the whole stream, joins included."""
    from test_magnitude_whitening import stream

    y, ar = coloured
    whitening = MagnitudeWhitening(ar, 2000, 0, band=front.passband(), sampling=FS,
                                   duration=4.0)
    whitening.SetOutputSize(2000, whitening.latency)
    x = y[:int(60 * FS)]
    streamed, _ = stream(whitening, x, [777, 5000, 5001, 40000, 90001])
    reference = fftconvolve(x, whitening.taps, mode="full")[whitening.latency:][:streamed.size]
    settled = slice(whitening.latency, None)
    assert np.abs(streamed[settled] - reference[settled]).max() < 1e-12 * np.abs(reference).max()


@pytest.fixture(scope="module")
def reference_triggers(tmp_path_factory):
    from conftest import run_segment_process
    return run_segment_process(str(tmp_path_factory.mktemp("magnitude")) + "/")


@pytest.mark.parametrize("model", ["burg", "spectrum"])
def test_the_worker_searches_the_truncated_stream_like_the_full_band_one(
        tmp_outdir, reference_triggers, model):
    """On the pure-noise fixture, the count and the energy of the reference, to 10%."""
    from conftest import run_segment_process

    triggers = run_segment_process(tmp_outdir, ZeroPhaseFilter="truncated",
                                   WhiteningModel=model)
    with open(glob.glob(f"{tmp_outdir}**/parametersUsed-Win*.json", recursive=True)[0]) as fh:
        used = json.load(fh)
    assert used["ZeroPhaseFilter"] == "truncated"
    assert used["ZeroPhaseDuration"] == DEFAULT_TRUNCATION_S
    assert used["ZeroPhaseLatency"] == int(DEFAULT_TRUNCATION_S * used["resampling"] / 2) - (
        1 if model == "spectrum" else 0)
    assert used["sigma"] == used["sigmaWhitened"]
    assert len(triggers) == pytest.approx(len(reference_triggers), rel=0.10)
    assert np.median(triggers.EnWDF) == pytest.approx(np.median(reference_triggers.EnWDF),
                                                      rel=0.10)
