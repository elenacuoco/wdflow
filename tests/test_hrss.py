"""The hrss estimate: the whitening undone on a known signal, the noise
floor of pure noise, the band and model choices, and the network columns."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

from wdf.analysis.hrss import (HRSS_COLUMNS, NETWORK_HRSS_COLUMN, HrssSettings, ModelHistory,
                               event_hrss, network_hrss, strain_from_whitened)
from wdf.config.Parameters import Parameters

FS = 2048.0


def band_hrss(x, fs, band):
    spec = np.fft.rfft(x)
    freq = np.fft.rfftfreq(x.size, 1.0 / fs)
    spec[(freq < band[0]) | (freq > band[1])] = 0.0
    return np.sqrt(np.sum(np.fft.irfft(spec, x.size) ** 2) / fs)


def ar2_model(scale=1e-22):
    # A(z) = 1 - 1.2 z^-1 + 0.5 z^-2: a coloured spectrum, peaked at low frequency.
    return np.array([scale, 1.2, -0.5])


def sine_gaussian(f0=150.0, q=9.0, amplitude=1e-21, n=8192):
    t = (np.arange(n) - n / 2) / FS
    tau = q / (np.sqrt(2.0) * np.pi * f0)
    return amplitude * np.exp(-(t / tau) ** 2) * np.sin(2 * np.pi * f0 * t)


def test_the_whitening_is_undone_on_a_known_signal():
    from scipy.signal import fftconvolve
    from wdf.processes.zero_phase_whitening import truncated_taps

    taps = truncated_taps(ar2_model(), 4.0, FS)
    x = sine_gaussian()
    white = fftconvolve(np.pad(x, taps.size), taps, mode="same")
    band = (32.0, 900.0)
    strain, _ = strain_from_whitened(white, FS, taps, band)
    assert np.sqrt(np.sum(strain ** 2) / FS) == pytest.approx(band_hrss(x, FS, band), rel=1e-3)


def test_the_noise_floor_is_the_hrss_of_pure_noise():
    rng = np.random.default_rng(3)
    sigma, n = 2e-22, 20000
    white = sigma * rng.standard_normal(n)
    got = event_hrss(white, FS, np.array([1.0]), (0.0, FS / 2), sigma, n)
    assert got["hrss_noise"] == pytest.approx(got["hrss"], rel=0.02)
    assert got["hrss_err"] > 0


def test_a_loud_signal_has_a_small_relative_error():
    rng = np.random.default_rng(4)
    sigma = 1e-23
    x = sine_gaussian(amplitude=1e-21)
    got = event_hrss(x + sigma * rng.standard_normal(x.size), FS, np.array([1.0]),
                     (0.0, FS / 2), sigma, 200)
    assert got["hrss"] == pytest.approx(band_hrss(x, FS, (0.0, FS / 2)), rel=0.01)
    assert got["hrss_err"] / got["hrss"] < 0.01
    assert got["hrss_noise"] < got["hrss"]


def test_the_band_defaults_to_min_frequency_and_the_low_pass_corner():
    par = Parameters(MinFrequency=32.0, LowFrequencyCut=8.0, ResamplingFactor=2,
                     BandPassFilter="butterworth", LowPassCornerFraction=0.87890625)
    settings = HrssSettings.from_parameters(par)
    assert not settings.active
    assert settings.band(FS) == pytest.approx((32.0, 900.0))
    par = Parameters(MinFrequency=0.0, LowFrequencyCut=8.0, ResamplingFactor=2,
                     BandPassFilter="butterworth", HrssEstimate=True, HrssHighFrequency=500.0)
    settings = HrssSettings.from_parameters(par)
    assert settings.active and settings.band(FS) == pytest.approx((8.0, 500.0))
    with pytest.raises(ValueError, match="HrssLowFrequency"):
        HrssSettings.from_parameters(Parameters(HrssLowFrequency="nyquist")).band(FS)


def test_the_model_in_force_and_merged_files(tmp_path):
    from wdf.processes.wdfUnitOnline import save_ar_models

    attrs = dict(ARorder=2, WhiteningMode="zerophase", resampling=FS, ZeroPhaseDuration=4.0, itf="H1")
    model = lambda t: dict(switch_gps=t, fit_start=t - 300, fit_end=t, sigma=1e-22, ar=ar2_model())
    save_ar_models(str(tmp_path / "a.h5"), [model(100.0), model(200.0)], attrs)
    save_ar_models(str(tmp_path / "b.h5"), [model(200.0), model(300.0)], attrs)
    history = ModelHistory.from_files([str(tmp_path / "a.h5"), str(tmp_path / "b.h5")])
    assert history.switch.tolist() == [100.0, 200.0, 300.0]
    assert [history.index_at(t) for t in (50.0, 100.0, 250.0, 400.0)] == [0, 0, 1, 2]
    with pytest.raises(ValueError, match="zero-phase"):
        ModelHistory([model(1.0)], dict(attrs, WhiteningMode="causal"))


def test_the_network_value_is_the_root_sum_square():
    table = pd.DataFrame({"hrss_H1": [3.0, np.nan, np.nan], "hrss_L1": [4.0, 2.0, np.nan]})
    out = network_hrss(table, ["H1", "L1"])
    assert out[NETWORK_HRSS_COLUMN].iloc[:2].tolist() == [5.0, 2.0]
    assert np.isnan(out[NETWORK_HRSS_COLUMN].iloc[2])
    assert NETWORK_HRSS_COLUMN not in network_hrss(pd.DataFrame({"x": [1.0]}), ["H1", "L1"])


# ---------------------------------------------------------------- run_network
pytest.importorskip("py4tsa")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
sys.path.insert(0, FIXTURES)
import make_paper_path_snapshot as snap  # noqa: E402


#: A sine-Gaussian burst (150 Hz, Q 9, amplitude 10 on unit white noise:
#: matched-filter SNR about 60), L1 5 ms after H1.
BURST_AT, BURST_F0, BURST_Q, BURST_AMPLITUDE = 80.0, 150.0, 9.0, 10.0


def burst(ifo, fs=snap.FS):
    t = np.arange(int(snap.SECONDS * fs)) / fs - BURST_AT - snap.DELAY[ifo]
    tau = BURST_Q / (np.sqrt(2.0) * np.pi * BURST_F0)
    return BURST_AMPLITUDE * np.exp(-(t / tau) ** 2) * np.sin(2 * np.pi * BURST_F0 * t)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    from wdf.analysis.io import triggers_from_files
    from wdf.processes.wdfUnitOnline import ArraySource, wdfUnitOnline

    out = str(tmp_path_factory.mktemp("hrss"))
    got = {}
    for ifo in snap.DELAY:
        rng = np.random.default_rng({"H1": 11, "L1": 12}[ifo])
        x = rng.standard_normal(int(snap.SECONDS * snap.FS)) + burst(ifo)
        worker = wdfUnitOnline(snap.parameters(ifo, out, **snap.FROZEN))
        worker.segmentProcess(snap.SEGMENT, ArraySource(x, snap.T0, snap.FS))
        files = sorted(__import__("glob").glob(worker.dir + "WDFTriggers-*.parquet"))
        got[ifo] = (triggers_from_files(files, ifo), worker)
    return got


def test_run_network_adds_no_column_by_default_and_asks_for_the_models(runs):
    from wdf.analysis.network_driver import run_network

    triggers = {ifo: r[0] for ifo, r in runs.items()}
    res = run_network(Parameters(), triggers=triggers)
    assert not any(c.startswith("hrss") for c in res.candidates.columns)
    assert not any(c in ev for ev in res.events.values() for c in HRSS_COLUMNS)
    with pytest.raises(ValueError, match="noise models"):
        run_network(Parameters(HrssEstimate=True), triggers=triggers)


def test_run_network_recovers_the_injected_burst_hrss(runs):
    """A compact burst well above threshold: the loudest event's hrss is the
    burst's in-band hrss within its error and the threshold's loss, and the
    candidates carry the per-detector and network columns."""
    import glob
    from wdf.analysis.network_driver import run_network

    triggers = {ifo: r[0] for ifo, r in runs.items()}
    models = {ifo: glob.glob(r[1].dir + "ARModels-*.h5") for ifo, r in runs.items()}
    par = snap.parameters("H1", "/unused", HrssEstimate=True)
    res = run_network(par, triggers=triggers, ar_models=models)
    band = HrssSettings.from_parameters(par).band(FS)
    for ifo in snap.DELAY:
        ev = res.events[ifo]
        loud = ev.loc[ev["EnWDF"].idxmax()]
        truth = band_hrss(burst(ifo), snap.FS, band)
        assert 0.85 * truth < loud["hrss"] < 1.1 * truth
        assert 0.0 < loud["hrss_noise"] < 0.5 * loud["hrss"]
        assert 0.0 < loud["hrss_err"] < 0.2 * loud["hrss"]
    c = res.candidates
    assert len(c) and {"hrss_H1", "hrss_L1", "hrss_err_H1", "hrss_noise_L1", NETWORK_HRSS_COLUMN} <= set(c)
    row = c.iloc[0]
    assert row[NETWORK_HRSS_COLUMN] == pytest.approx(np.hypot(row["hrss_H1"], row["hrss_L1"]))
