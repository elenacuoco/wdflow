"""Parameter estimation against waveforms whose parameters are known analytically.

`tests/test_parameter_estimation_observer.py` covers the degenerate inputs; this
file checks that the numbers mean what they claim on well-posed ones.
"""
import numpy as np
import pytest

from wdf.mock import waveforms as w
from wdf.observers.ParameterEstimationObserver import (
    energy_interval,
    estimate_duration,
    extract_meta_features,
)

FS = 2048.0
N = 512
SIGMA = 1.0
TARGET_SNR = 10.0


def centred(samples, snr=TARGET_SNR):
    """One waveform, scaled to a known l2 norm, centred in an analysis window."""
    y = np.asarray(samples, dtype=float)
    x = np.zeros(N)
    n = min(len(y), N)
    offset = (N - n) // 2
    x[offset:offset + n] = y[:n] / np.linalg.norm(y[:n]) * snr
    return x


def features(x, sigma=SIGMA):
    """Meta features on the convention the observer uses: EnWDF = ||x|| / sigma."""
    return extract_meta_features(x, fs=FS, sigma=sigma,
                                 EnWDF=np.linalg.norm(x) / sigma, f_low=0.0)


@pytest.mark.parametrize("f0", [150.0, 250.0, 400.0])
def test_frequency_of_a_narrowband_burst_is_recovered(f0):
    """A sine-Gaussian carries its frequency in both freqMean and freqPeak."""
    _, _, freq_min, freq_mean, freq_max, freq_peak, _, _ = features(
        centred(w.sine_gaussian(f0, 12.0, int(FS))))

    assert freq_mean == pytest.approx(f0, rel=0.02)
    assert freq_peak == pytest.approx(f0, rel=0.02)
    assert freq_min < f0 < freq_max


def test_bandwidth_scales_with_the_quality_factor():
    """Constant Q means constant fractional bandwidth, not constant bandwidth."""
    widths = {}
    for f0 in (150.0, 400.0):
        _, _, freq_min, _, freq_max, _, _, _ = features(
            centred(w.sine_gaussian(f0, 12.0, int(FS))))
        widths[f0] = (freq_max - freq_min) / f0

    assert widths[150.0] == pytest.approx(widths[400.0], rel=0.15)


@pytest.mark.parametrize("f0,quality", [(150.0, 12.0), (400.0, 12.0), (300.0, 8.0)])
def test_duration_matches_the_envelope_it_was_generated_with(f0, quality):
    """The 90% central energy interval of a Gaussian envelope is analytic.

    `sine_gaussian` has envelope exp(-t^2 / 2 tau^2) with tau = Q / (2 pi f0), so
    its energy density is Gaussian with standard deviation tau / sqrt(2) and the
    interval holding the central 90% spans 2 * 1.645 of those.
    """
    tau = quality / (2.0 * np.pi * f0)
    expected = 2.0 * 1.6449 * tau / np.sqrt(2.0)

    duration = estimate_duration(centred(w.sine_gaussian(f0, quality, int(FS))), FS)
    assert duration == pytest.approx(expected, rel=0.12)


def test_peak_time_finds_the_centre_of_the_window():
    """tPeak is measured from the start of the analysis window."""
    t_peak = features(centred(w.sine_gaussian(200.0, 12.0, int(FS))))[0]
    assert t_peak == pytest.approx((N // 2) / FS, abs=3e-3)


def test_snr_peak_and_mean_are_amplitudes_on_the_noise_scale():
    """snrPeak is the peak sample over sigma; snrMean the rms over the support."""
    x = centred(w.sine_gaussian(200.0, 12.0, int(FS)))
    _, _, _, _, _, _, snr_mean, snr_peak = features(x)

    start, end = energy_interval(x, 0.90)
    assert snr_peak == pytest.approx(np.abs(x).max() / SIGMA, rel=1e-6)
    assert snr_mean == pytest.approx(
        np.sqrt(np.mean(x[start:end + 1] ** 2)) / SIGMA, rel=1e-6)


def test_ordering_holds_for_every_shape():
    """snrMean <= snrPeak <= EnWDF, which is what the observer documents."""
    shapes = (w.sine_gaussian(150.0, 12.0, int(FS)),
              w.gaussian(0.005, int(FS)),
              w.blip(250.0, 3.0, sample_rate=int(FS)),
              w.chirplike(40.0, 400.0, 0.25, int(FS)))

    for samples in shapes:
        x = centred(samples)
        *_, snr_mean, snr_peak = features(x)
        assert snr_mean <= snr_peak + 1e-9
        assert snr_peak <= np.linalg.norm(x) / SIGMA + 1e-9


def test_snr_mean_is_not_diluted_by_the_window_length():
    """A short transient's mean amplitude is set by its support, not the window.

    Averaging over the whole window instead would scale the answer by the square
    root of the ratio of the two lengths.
    """
    x = centred(w.sine_gaussian(400.0, 12.0, int(FS)))
    start, end = energy_interval(x, 0.90)
    *_, snr_mean, _ = features(x)

    over_window = np.sqrt(np.mean(x ** 2)) / SIGMA
    assert snr_mean > 3.0 * over_window
    assert (end - start + 1) < N // 4
