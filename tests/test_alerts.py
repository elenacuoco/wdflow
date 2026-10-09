"""The alert table: the selection on the rate, the fields, what is left empty,
and the CSV `run_network` writes when asked."""
import os

import numpy as np
import pandas as pd
import pytest

from wdf.analysis.alerts import (
    ALERT_COLUMNS, PROVENANCE_COLUMNS, AlertSettings, alert_path, alert_table, far_hz_of,
    per_ifo_columns, per_pair_columns, provenance, write_alerts,
)
from wdf.config.Parameters import Parameters

DAY = 86400.0


def _candidates(n=6):
    """Six H1-L1 pairs: rates 0.1, 0.5, 1, 2, 10, 100 per day."""
    t0 = 1400000000.0 + np.arange(n) * 50.0
    per_day = np.array([0.1, 0.5, 1.0, 2.0, 10.0, 100.0])[:n]
    return pd.DataFrame(dict(
        candidate_id=np.arange(n), ifos_involved="H1,L1", n_ifos=2,
        ifo_i="H1", ifo_j="L1", cluster_i=np.arange(n), cluster_j=10 + np.arange(n),
        gps_candidate=t0 + 0.5, event_instant="gpsEnvelope",
        instant_H1=t0 + 0.5, instant_L1=t0 + 0.5 - 0.004,
        gpsPeak_H1=t0 + 0.5, gpsPeak_L1=t0 + 0.496,
        gpsStart_H1=t0 + 0.4, gpsStart_L1=t0 + 0.42, duration_H1=0.3, duration_L1=0.25,
        network_enwdf=np.linspace(30.0, 10.0, n), network_min_enwdf=np.linspace(20.0, 5.0, n),
        network_morphology=np.linspace(15.0, 3.0, n), rank=np.arange(1, n + 1),
        network_lag_s=0.004, network_lag_width_s=0.0005, network_lag_window_s=0.02,
        hrss_H1=1e-22, hrss_L1=2e-22, hrss_err_H1=1e-23, hrss_err_L1=2e-23,
        hrss_noise_H1=5e-24, hrss_noise_L1=5e-24, hrss_network_det=np.sqrt(5.0) * 1e-22,
        glitch_single_ifo=[False] * (n - 1) + [True], glitch_i=False, glitch_j=False,
        far_per_day=per_day, n_background_ge=(per_day * 1000.0 - 1).astype(int),
    ))


def _events(n=6):
    rows = []
    for k in range(n):
        for ifo, cluster, f in (("H1", k, 100.0), ("L1", 10 + k, 120.0)):
            rows.append(dict(ifo=ifo, cluster_id=cluster, EnWDF=20.0 if ifo == "H1" else 10.0,
                             snrPeak=6.0, freqMin=f - 40, freqMean=f, freqMax=f + 60,
                             freqQ05=f - 30, freqQ95=f + 50, duration90=0.2 + 0.01 * k,
                             fs=2048.0))
    return pd.DataFrame(rows)


def test_the_defaults_are_one_per_day_and_one_per_year():
    s = AlertSettings.from_parameters(Parameters())
    assert s.far_threshold_hz == pytest.approx(1.0 / DAY)
    assert s.trials_factor == 1.0 and s.csv is None and s.sky
    assert s.significant_far_hz == pytest.approx(1.0 / (365.25 * DAY))
    assert (s.pipeline, s.group, s.search) == ("WDF", "Burst", "AllSky")
    assert s.sigma_floor_s(2048.0) == pytest.approx(1.0 / 2048.0)
    with pytest.raises(ValueError, match="fs"):
        s.sigma_floor_s(None)


def test_bad_settings_are_refused():
    with pytest.raises(ValueError, match="AlertFARThreshold"):
        AlertSettings.from_parameters(Parameters(AlertFARThreshold=0.0))
    with pytest.raises(ValueError, match="AlertTrialsFactor"):
        AlertSettings.from_parameters(Parameters(AlertTrialsFactor=0.5))
    with pytest.raises(ValueError, match="AlertSkyTiming"):
        AlertSettings.from_parameters(Parameters(AlertSkyTiming="peak"))
    with pytest.raises(ValueError, match="AlertSkySigmaFloor"):
        AlertSettings.from_parameters(Parameters(AlertSkySigmaFloor="tick"))


def test_the_rate_is_read_from_whatever_the_table_states():
    c = _candidates()
    assert far_hz_of(c) == pytest.approx(c.far_per_day / DAY)
    counts = c.drop(columns=["far_per_day"])
    assert far_hz_of(counts, livetime_s=1000.0 * DAY) == pytest.approx(c.far_per_day / DAY)
    with pytest.raises(ValueError, match="livetime"):
        far_hz_of(counts)
    with pytest.raises(ValueError, match="no false-alarm rate"):
        far_hz_of(c.drop(columns=["far_per_day", "n_background_ge"]))


def test_the_selection_is_at_or_below_the_threshold_and_the_trials_count():
    c = _candidates()
    a = alert_table(c, par=Parameters(), fs=2048.0)
    assert list(a.candidate_id) == [0, 1, 2]        # 0.1, 0.5 and 1 per day; 2 per day is out
    assert list(a.far_per_day) == pytest.approx([0.1, 0.5, 1.0])
    assert not a.significant.any()
    # One per ten years in the table: significant unless the trials factor takes it over 1/yr.
    rare = c.copy()
    rare.loc[0, "far_per_day"] = 1.0 / (10 * 365.25)
    one = alert_table(rare, par=Parameters(), fs=2048.0)
    assert bool(one.significant.iloc[0]) and one.far_trials_hz.iloc[0] == one.far_hz.iloc[0]
    many = alert_table(rare, par=Parameters(AlertTrialsFactor=20), fs=2048.0)
    assert not many.significant.iloc[0]
    assert many.far_trials_hz.iloc[0] == pytest.approx(20 * many.far_hz.iloc[0])
    wide = alert_table(c, par=Parameters(AlertFARThreshold=200.0 / DAY), fs=2048.0)
    assert len(wide) == 6 and list(wide.far_hz) == sorted(wide.far_hz)


def test_the_fields_and_their_order():
    c = _candidates()
    a = alert_table(c, events=_events(), par=Parameters(AlertRunName="unit"), fs=2048.0)
    assert list(a.columns) == list(ALERT_COLUMNS) + per_ifo_columns(["H1", "L1"]) \
        + per_pair_columns(["H1", "L1"]) + list(PROVENANCE_COLUMNS)
    r = a.iloc[0]
    assert (r.pipeline, r.group, r.search, r.run, r.ifos, r.n_ifos) == \
        ("WDF", "Burst", "AllSky", "unit", "H1,L1", 2)
    assert r.gpstime == c.gps_candidate[0] and r.gps_H1 == c.instant_H1[0]
    assert r.gps_L1 == c.instant_L1[0] and r.peak_time_L1 == c.gpsPeak_L1[0]
    assert r.detection_statistic_name == "network_morphology"
    assert r.detection_statistic == c.network_morphology[0]
    # Morphology from the events: per detector and combined.
    assert r.central_freq_H1 == 100.0 and r.bandwidth_L1 == 80.0 and r.freq_low_H1 == 60.0
    assert r.EnWDF_H1 == 20.0 and r.EnWDF_L1 == 10.0 and r.snrPeak_L1 == 6.0
    assert r.central_freq == pytest.approx((20 * 100 + 10 * 120) / 30)   # EnWDF-weighted
    assert r.bandwidth == pytest.approx(170.0 - 70.0)
    assert r.duration == pytest.approx(0.2) and r.duration90_H1 == pytest.approx(0.2)
    assert r.start_time == c.gpsStart_H1[0] and r.end_time == pytest.approx(c.gpsStart_L1[0] + 0.25)
    # hrss, with the rss error propagated.
    assert r.hrss_network_det == pytest.approx(np.sqrt(5.0) * 1e-22)
    assert r.hrss_err_network_det == pytest.approx(
        np.sqrt((1e-22 * 1e-23) ** 2 + (2e-22 * 2e-23) ** 2) / (np.sqrt(5.0) * 1e-22))
    # Timing: the instants' difference, the light travel time and the xcorr lag.
    assert r.dt_envelope_s_H1L1 == pytest.approx(0.004, abs=1e-6) and r.dt_physical_H1L1
    assert r.light_travel_s_H1L1 == pytest.approx(0.010, abs=2e-4)
    assert r.dt_xcorr_s == 0.004 and r.sky_timing == "xcorr" and r.sky_dt_s == 0.004
    assert r.sky_sigma_s == pytest.approx(0.0005)       # the width, above one sample
    assert bool(r.glitch) is False and bool(r.glitch_H1) is False
    assert bool(a.glitch.iloc[-1]) is False       # the glitch pair is at 100 per day: not kept
    # The sky: a ring, far from the whole sky and far from a point.
    assert 100.0 < r.sky_area90_deg2 < 20000.0 and r.sky_area50_deg2 < r.sky_area90_deg2
    assert 0.0 <= r.sky_ra_max_deg < 360.0 and -90.0 <= r.sky_dec_max_deg <= 90.0
    assert r.sky_map_path is None
    assert r.wdflow_version is None or isinstance(r.wdflow_version, str)


def test_what_is_not_given_stays_empty():
    c = _candidates().drop(columns=[col for col in _candidates() if col.startswith("hrss")
                                    or col.startswith("network_lag") or col.startswith("glitch")])
    a = alert_table(c, par=Parameters(), fs=2048.0)
    r = a.iloc[0]
    assert np.isnan(r.central_freq) and np.isnan(r.EnWDF_H1) and np.isnan(r.hrss_H1)
    assert np.isnan(r.hrss_network_det) and np.isnan(r.dt_xcorr_s) and r.glitch is None
    assert r.sky_timing == "envelope" and r.sky_dt_s == pytest.approx(0.004, abs=1e-6)
    assert r.sky_sigma_s == pytest.approx(1.0 / 2048.0)
    assert np.isfinite(r.sky_area90_deg2)
    off = alert_table(c, par=Parameters(AlertSkyMap=False), fs=None)
    assert off.sky_timing.isna().all() and off.sky_area90_deg2.isna().all()
    empty = alert_table(c.iloc[:0], par=Parameters())
    assert len(empty) == 0 and list(empty.columns)[:3] == list(ALERT_COLUMNS[:3])


def test_the_sky_map_and_the_csv_are_written(tmp_path):
    c = _candidates()
    par = Parameters(AlertSkyMapDir=str(tmp_path / "sky"), AlertRunName="r")
    a = alert_table(c, par=par, fs=2048.0)
    m = np.load(a.sky_map_path.iloc[0])
    assert m["weight"].sum() == pytest.approx(1.0) and m["ra"].shape == (180, 360)
    assert list(m["ifos"]) == ["H1", "L1"]
    path = write_alerts(a, tmp_path / "out" / "alerts.csv")
    back = pd.read_csv(path)
    assert len(back) == 3 and back.gpstime.iloc[0] == c.gps_candidate[0]    # GPS kept exactly
    assert back.hrss_H1.iloc[0] == pytest.approx(1e-22)
    assert alert_path("a/{run}-{start}-{end}.csv", 1400000000.2, 1400000100.9, "r") == \
        "a/r-1400000000-1400000101.csv"


def test_the_provenance_names_the_files_and_the_commit(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text("x = 1\n")
    p = provenance(Parameters(ConfigFile=str(cfg), GNNModel=str(tmp_path / "none.pt")),
                   data_start=1.0, data_end=2.0)
    assert p["config_path"] == str(cfg) and len(p["config_sha256"]) == 64
    assert p["model_sha256"] is None and p["data_start"] == 1.0
    assert p["wdflow_commit"] is None or len(p["wdflow_commit"]) >= 40
    assert p["created_utc"].endswith("Z")


@pytest.mark.skipif(not pytest.importorskip("py4tsa", reason="no py4tsa"), reason="no py4tsa")
def test_run_network_writes_the_csv_only_when_asked(tmp_path):
    import sys

    fixtures = os.path.join(os.path.dirname(__file__), "fixtures")
    sys.path.insert(0, fixtures)
    import make_paper_path_snapshot as snap  # noqa: E402

    from wdf.analysis.network_driver import run_network

    triggers = {ifo: pd.read_parquet(os.path.join(fixtures, f"paper_path_triggers_{ifo}.parquet"))
                for ifo in snap.DELAY}
    plain = run_network(Parameters(NetworkSlides=5, NetworkSlideStep=4.0), triggers=triggers)
    assert plain.alerts_path is None
    with pytest.raises(ValueError, match="NetworkSlides"):
        run_network(Parameters(AlertCSV=str(tmp_path / "x.csv")), triggers=triggers)
    pattern = str(tmp_path / "alerts-{start}-{end}.csv")
    res = run_network(Parameters(NetworkSlides=5, NetworkSlideStep=4.0, AlertCSV=pattern,
                                 AlertFARThreshold=1.0, AlertRunName="snap"), triggers=triggers)
    assert res.alerts_path and os.path.isfile(res.alerts_path)
    assert "{start}" not in res.alerts_path
    pd.testing.assert_frame_equal(res.candidates, plain.candidates)
    table = pd.read_csv(res.alerts_path)
    assert len(table) == int((plain.candidates.far_hz <= 1.0).sum()) > 0
    assert table.run.iloc[0] == "snap" and np.isfinite(table.central_freq).all()
    assert np.isfinite(table.EnWDF_H1).all() and table.data_start.notna().all()


def test_the_statistic_is_the_penalised_column_the_rate_was_measured_on():
    c = _candidates().assign(gnn_logit=np.linspace(2.0, -3.0, 6))
    assert alert_table(c, par=Parameters(), fs=2048.0).detection_statistic_name.iloc[0] == "gnn_logit"
    penalised = c.assign(network_rank_statistic=c.gnn_logit + np.log(0.5))
    a = alert_table(penalised, par=Parameters(), fs=2048.0)
    assert a.detection_statistic_name.iloc[0] == "network_rank_statistic"
    assert a.detection_statistic.iloc[0] == pytest.approx(2.0 + np.log(0.5))
