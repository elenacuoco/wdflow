"""The white three-detector benchmark set and the pieces it is built from."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from wdf.mock import waveforms as w
from wdf.mock.dataset import (_draw_snr, chirp_mass_of, component_masses,
                              draw_injections, generate_dataset, optimal_snr,
                              truth_columns, GROUND_TRUTH_COLUMNS)
from wdf.mock.noise import white_noise, white_psd

FS = 2048
NETWORK = ("H1", "L1", "V1")


def small_white_set(outdir, **overrides):
    settings = dict(duration=1200.0, start_gps=1400000000.0, sample_rate=FS,
                    n_cbc=5, n_glitch=0, seed=11, detectors=NETWORK,
                    edge_pad=100.0, noise="white",
                    relative_sensitivity={"V1": 0.32}, cbc_population={},
                    snr_range=(6.0, 20.0), snr_core=(7.0, 12.0, 0.6),
                    minimum_injection_gap=20.0)
    settings.update(overrides)
    return generate_dataset(str(outdir), **settings)


def test_white_noise_is_unit_variance_and_reproducible():
    a = np.asarray(white_noise(0.0, 64.0, seed=3, sample_rate=FS))
    b = np.asarray(white_noise(0.0, 64.0, seed=3, sample_rate=FS))
    assert a.size == 64 * FS
    assert np.array_equal(a, b)
    assert a.var() == pytest.approx(1.0, abs=0.02)


def test_the_flat_spectrum_makes_the_snr_the_norm_of_the_samples():
    """In unit-variance white noise the optimal SNR is the plain sample norm."""
    psd = white_psd(int(0.5 * FS * 16) + 1, 1.0 / 16.0, FS)
    y = w.sine_gaussian(150.0, 9.0, FS)
    assert optimal_snr(y, FS, 0.0, psd=psd) == pytest.approx(
        np.sqrt(y @ y), rel=1e-4)


def test_a_third_detector_gets_its_own_columns():
    assert truth_columns(("H1", "L1")) == GROUND_TRUTH_COLUMNS
    columns = truth_columns(NETWORK)
    for prefix in ("gps_", "gps_start_", "gps_end_", "snr_"):
        assert f"{prefix}V1" in columns
    assert columns.index("snr_V1") < columns.index("network_snr")


def test_component_masses_give_back_the_chirp_mass():
    for chirp_mass, q in ((5.0, 1.0), (12.0, 2.5), (30.0, 4.0)):
        m1, m2 = component_masses(chirp_mass, q)
        assert m1 >= m2
        assert m1 / m2 == pytest.approx(q)
        assert chirp_mass_of(m1, m2) == pytest.approx(chirp_mass)


def test_the_snr_core_holds_the_share_it_is_given():
    rng = np.random.default_rng(0)
    values = np.array([_draw_snr(rng, (6.0, 20.0), (7.0, 12.0, 0.6))
                       for _ in range(20000)])
    assert values.min() >= 6.0 and values.max() <= 20.0
    # 0.6 inside by construction, plus the uniform part's 5/14 of the rest.
    assert np.mean((values >= 7.0) & (values <= 12.0)) == pytest.approx(
        0.6 + 0.4 * 5.0 / 14.0, abs=0.015)


def test_an_snr_core_outside_the_range_is_refused():
    with pytest.raises(ValueError):
        _draw_snr(np.random.default_rng(0), (6.0, 20.0), (5.0, 12.0, 0.5))
    with pytest.raises(ValueError):
        draw_injections(n_cbc=1, n_glitch=0, snr_core=(7.0, 12.0, 0.5))


def test_the_chirp_mass_population_stays_in_its_range():
    specs = draw_injections(n_cbc=40, n_glitch=0, duration=40000.0,
                            edge_pad=100.0, seed=2, detectors=NETWORK,
                            cbc_population={"chirp_mass": (5.0, 30.0)},
                            snr_range=(6.0, 20.0))
    chirp = np.array([chirp_mass_of(s["mass1"], s["mass2"]) for s in specs])
    assert chirp.min() >= 5.0 - 1e-9 and chirp.max() <= 30.0 + 1e-9
    assert all(s["approximant"] == "IMRPhenomD" for s in specs)


def test_the_track_rises_through_the_band_and_ends_at_the_merger():
    hp, hc = w.cbc_polarisations(12.0, 8.0, inclination=0.0, f_lower=20.0)
    time, frequency = w.cbc_track(hp, hc, FS, float(hp.start_time), 32)
    assert np.all(np.diff(frequency) > 0) and np.all(np.diff(time) >= 0)
    assert 10.0 < frequency[0] < 25.0
    assert frequency[-1] > 200.0
    assert time[0] < -1.0 and abs(time[-1]) < 0.05


def test_virgo_receives_its_sensitivity_and_nothing_else(tmp_path):
    """Sensitivity scales V1's amplitude relative to the LIGOs and moves
    nothing: not the draw, not the arrival times, not the H1-L1 ratio."""
    kw = dict(write_background=False, track_points=0)
    full = small_white_set(tmp_path / "a", relative_sensitivity={"V1": 1.0}, **kw)
    scaled = small_white_set(tmp_path / "b", **kw)
    assert np.allclose(full["gps_V1"], scaled["gps_V1"])
    ratio_full = full["snr_V1"] / full["snr_H1"]
    ratio_scaled = scaled["snr_V1"] / scaled["snr_H1"]
    assert np.allclose(ratio_scaled / ratio_full, 0.32, rtol=1e-6)
    assert np.allclose(full["snr_L1"] / full["snr_H1"],
                       scaled["snr_L1"] / scaled["snr_H1"], rtol=1e-6)
    quad = np.sqrt(sum(scaled[f"snr_{ifo}"] ** 2 for ifo in NETWORK))
    assert np.allclose(quad, scaled["network_snr"], rtol=1e-6)
    assert np.allclose(scaled["network_snr"], scaled["target_snr"], rtol=1e-3)


def test_an_unknown_detector_sensitivity_is_refused(tmp_path):
    with pytest.raises(ValueError):
        small_white_set(tmp_path, relative_sensitivity={"K1": 0.1})


def test_the_benchmark_validates_on_the_frames_it_wrote(tmp_path):
    from wdf.mock.benchmark import BENCHMARK_CONFIG, write_benchmark

    config = dict(BENCHMARK_CONFIG, duration=1200.0, n_cbc=5, edge_pad=100.0,
                  frame_length=512.0)
    truth = write_benchmark(tmp_path, config=config)
    for name in ("benchmark_config.json", "provenance.json", "README.md",
                 "validation.json", "injections.parquet", "tracks.parquet"):
        assert os.path.getsize(tmp_path / name) > 0
    for ifo in NETWORK:
        for kind in ("FOREGROUND", "BACKGROUND"):
            assert os.path.getsize(tmp_path / f"{ifo}-MOCK-{kind}.ffl") > 0

    summary = json.loads((tmp_path / "validation.json").read_text())
    assert summary["worker_read_max_difference"] == 0.0
    for ifo in NETWORK:
        mf = summary["matched_filter"][ifo]
        assert mf["n"] == len(truth)
        assert mf["injected_over_recorded_max_deviation"] < 1e-3
        assert mf["injected_lag_samples_max"] == 0
        for band in summary["spectrum"][ifo]["BACKGROUND"]:
            assert abs(band["mean_ratio"] - 1.0) < 5.0 * band["mean_sigma"]
    for dt, bound in summary["light_travel"].values():
        assert dt <= bound + 1e-6

    tracks = pd.read_parquet(tmp_path / "tracks.parquet")
    assert set(tracks["injection_id"]) == set(truth["injection_id"])


def test_the_background_holds_no_injection(tmp_path):
    from wdf.mock.validation import read_ffl

    truth = small_white_set(tmp_path, track_points=0)
    fg, _, t0 = read_ffl(str(tmp_path / "V1-MOCK-FOREGROUND.ffl"), "V1:MOCK-STRAIN")
    bg, _, _ = read_ffl(str(tmp_path / "V1-MOCK-BACKGROUND.ffl"), "V1:MOCK-STRAIN")
    inside = np.zeros(fg.size, dtype=bool)
    for _, row in truth.iterrows():
        i0 = int(round((row["gps_start_V1"] - t0) * FS))
        i1 = int(round((row["gps_end_V1"] - t0) * FS))
        inside[i0:i1] = True
    assert np.all(fg[~inside] == bg[~inside])
    assert np.any(fg[inside] != bg[inside])
