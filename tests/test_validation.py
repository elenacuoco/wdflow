"""The check before the search: each criterion fails on a stream built to
violate it, and Gaussian white noise passes them all."""
import numpy as np
import pytest

from wdf.processes.gating import band_stream, octave_bands, transients
from wdf.processes.validation import (bns_range, ConditioningRejected,
                                      validate)

RATE = 2048.0
SECONDS = 480
BANDS = octave_bands(RATE, 16.0)
STOP_BAND = (921.6, 1024.0)


def check(samples, gates=np.zeros((0, 2))):
    found = transients(samples, RATE, BANDS)
    return validate("X1", samples, RATE, 1000.0, BANDS, STOP_BAND, found, gates,
                    taper_s=0.25, window_pad_s=0.25)


def noise(seed=0, seconds=SECONDS):
    return np.random.default_rng(seed).standard_normal(int(seconds * RATE))


def test_gaussian_white_noise_passes():
    report = check(noise())
    assert report.passed, report.message()
    assert report.power == pytest.approx(1.0, abs=0.05)
    assert np.all(np.abs(report.kurtosis - report.gaussian) < 0.05)
    assert report.stop == pytest.approx(1000.0 + SECONDS)


def test_an_octave_that_is_not_white_fails_and_is_named():
    x = noise()
    x += 0.5 * band_stream(x, RATE, (128.0, 256.0))
    report = check(x)
    assert not report.passed
    assert any(f.startswith("128-256 Hz: power") for f in report.failures)
    assert "X1 fails" in report.message() and "128-256 Hz" in report.message()


def test_an_octave_that_is_not_gaussian_fails():
    """Intermittent noise in one octave, louder a tenth of the time: whatever
    its power reads, its distribution has heavy tails."""
    rng = np.random.default_rng(1)
    x = noise()
    octave = band_stream(x, RATE, (32.0, 64.0))
    blocks = rng.random(x.size // int(RATE / 2) + 1) < 0.1
    envelope = np.repeat(np.where(blocks, 3.0, 1.0), int(RATE / 2))[:x.size] / np.sqrt(1.8)
    report = check(x - octave + octave * envelope)
    assert any(f.startswith("32-64 Hz: window kurtosis") for f in report.failures)


def test_the_stop_band_is_read_too():
    """The band the conditioning empties and the whitening lifts back up is
    where a rounding floor shows, as spikes."""
    rng = np.random.default_rng(2)
    x = noise()
    spikes = np.zeros(x.size)
    spikes[rng.integers(0, x.size, 400)] = rng.normal(0.0, 40.0, 400)
    report = check(x + band_stream(spikes, RATE, STOP_BAND))
    assert any(f.startswith("921.6-1024 Hz: window kurtosis") for f in report.failures)


def test_transients_covering_more_than_their_share_fail():
    x = noise()
    t = np.arange(x.size) / RATE
    for centre in np.arange(20.0, SECONDS - 20.0, 20.0):
        x += 30.0 * np.exp(-((t - centre) / 0.02) ** 2) * np.sin(2 * np.pi * 100.0 * t)
    report = check(x)
    assert report.transients >= 20
    assert any(f.startswith("transients and gates cover") for f in report.failures)


def test_a_stretch_whose_noise_changes_fails():
    x = noise()
    x[2 * x.size // 3:] *= 1.3
    report = check(x)
    assert any("power in third" in f for f in report.failures)


def test_gated_windows_are_not_read_for_the_kurtosis_but_count_as_lost():
    """A window a gate zeroed in part is not searched there, so its kurtosis
    says nothing about the noise the search reads; its time is lost, and is
    counted as such."""
    x = noise()
    gate = np.array([[1000.0 + 200.0, 1000.0 + 201.5]])
    t = 1000.0 + np.arange(x.size) / RATE
    from wdf.processes.gating import gate_weights
    report = check(x * gate_weights(t, gate, 0.25), gates=gate)
    assert report.passed, report.message()
    assert report.transient_fraction == pytest.approx(2.0 / SECONDS, rel=0.05)
    assert len(report.gates) == 1


def test_the_report_can_be_raised_and_recorded():
    report = check(noise() * 1.5)
    error = ConditioningRejected(report)
    assert error.report is report and "power" in str(error)
    record = report.to_dict()
    assert record["passed"] is False and record["detector"] == "X1"
    assert len(record["kurtosis"]) == len(BANDS) + 1


def test_the_range_is_the_inspiral_integral():
    """On a flat spectrum the integral has a closed form, which fixes the
    constants, the chirp mass and the upper limit at the innermost stable
    orbit of the binary."""
    frequency = np.linspace(0.0, 4096.0, 2 ** 20 + 1)
    psd = np.full_like(frequency, 1e-46)
    c, g, sun, mpc = 299792458.0, 6.6743e-11, 1.988409870698051e30, 3.085677581491367e22
    m = 1.4 * sun
    chirp = (m * m) ** 0.6 / (2 * m) ** 0.2
    isco = c ** 3 / (g * 6 ** 1.5 * np.pi * 2 * m)
    prefactor = 1.77 ** 2 * 5 * c ** (1 / 3) * (chirp * g / c ** 2) ** (5 / 3) / (96 * np.pi ** (4 / 3) * 64)
    expected = np.sqrt(prefactor / 1e-46 * 0.75 * (10.0 ** (-4 / 3) - isco ** (-4 / 3))) / mpc
    assert bns_range(frequency, psd) == pytest.approx(expected, rel=1e-4)
    assert bns_range(frequency, 4 * psd) == pytest.approx(expected / 2, rel=1e-4)
