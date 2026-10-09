import numpy as np
import pandas as pd
import pytest

from wdf.analysis.combine import (CombineSettings, combine_candidates, combine_paths, combined_far,
                                  match_candidates)
from wdf.analysis.robust_events import CoincidenceConfig


def _table(times, far, stat, name, dt_h=0.0, dt_l=0.0, duration=0.5):
    t = np.asarray(times, float)
    return pd.DataFrame({
        "candidate_id": np.arange(len(t)), "gps_candidate": t, "far_per_day": np.asarray(far, float),
        stat: np.arange(len(t), 0, -1, dtype=float),
        "instant_H1": t + dt_h, "instant_L1": t + dt_l - 0.004,
        "gpsStart_H1": t - 0.3, "duration_H1": np.full(len(t), duration),
        "gpsStart_L1": t - 0.3, "duration_L1": np.full(len(t), duration),
        "n_background_ge": np.arange(len(t)), "rank": np.arange(len(t)), "stretch": name,
        "event_instant": name, "ifos_involved": "H1,L1", "n_ifos": 2,
    })


def test_default_window_is_the_pair_coincidence_window():
    s = CombineSettings()
    assert s.window() == pytest.approx(CoincidenceConfig().maximum_tolerance(("H1", "L1")))
    assert 0.02 < s.window() < 0.03
    assert CombineSettings(window_s=0.1).window() == 0.1
    assert s.trials(2) == 2.0
    assert CombineSettings(trials_factor=3).trials(2) == 3.0


def test_settings_reject_bad_values():
    with pytest.raises(ValueError):
        CombineSettings(match="nearest")
    with pytest.raises(ValueError):
        CombineSettings(window_s=0.0)
    with pytest.raises(ValueError):
        CombineSettings(trials_factor=0.5)


def test_settings_from_parameters_reads_the_keys():
    class Par:
        CombineWindow = 0.05
        CombineMatch = "extent"
        NetworkTrialsFactor = 1
        CombineTimePreference = "t, w"
        Detectors = ["H1", "L1"]

    s = CombineSettings.from_parameters(Par())
    assert s.window() == 0.05 and s.match == "extent" and s.trials(2) == 1.0
    assert s.time_preference == ("t", "w") and s.ifos == ("H1", "L1")
    d = CombineSettings.from_parameters(object())
    assert d.match == "instant" and d.trials_factor is None and d.time_preference == ("w", "t")


def test_combined_far_takes_the_best_rate_times_the_trials():
    r = np.array([[1.0, 0.1], [np.nan, 2.0], [0.5, np.nan], [np.nan, np.nan]])
    out = combined_far(r, 2.0)
    assert out[0] == pytest.approx(0.2) and out[1] == pytest.approx(4.0) and out[2] == pytest.approx(1.0)
    assert np.isnan(out[3])


def test_match_is_one_to_one_and_closest_first():
    a = _table([100.0, 100.010, 200.0], [1, 1, 1], "sw", "w")
    b = _table([100.004, 200.3], [1, 1], "st", "t")
    ia, ib, d = match_candidates(a, b, CombineSettings(window_s=0.02))
    assert list(ia) == [0] and list(ib) == [0]
    assert d[0] == pytest.approx(0.004)
    # with a wider window the second track row pairs with the second wavegram row, the closest one
    ia, ib, d = match_candidates(a, b, CombineSettings(window_s=0.5))
    assert sorted(zip(ia, ib)) == [(0, 0), (2, 1)]


def test_instant_rule_reads_every_detector():
    a = _table([100.0], [1], "sw", "w")
    b = _table([100.0], [1], "st", "t", dt_l=0.03)     # L1 instant 30 ms off, H1 aligned
    ia, _, _ = match_candidates(a, b, CombineSettings(window_s=0.02))
    assert len(ia) == 0
    ia, _, _ = match_candidates(a, b, CombineSettings(window_s=0.02, match="candidate"))
    assert len(ia) == 1


def test_extent_rule_overlaps_the_padded_extents():
    a = _table([100.0], [1], "sw", "w", duration=0.5)
    b = _table([100.4], [1], "st", "t", duration=0.5)   # extents [99.7, 100.2] and [100.1, 100.6] overlap
    assert len(match_candidates(a, b, CombineSettings(window_s=0.01))[0]) == 0
    assert len(match_candidates(a, b, CombineSettings(window_s=0.01, match="extent"))[0]) == 1
    c = _table([101.0], [1], "st", "t", duration=0.5)   # [100.7, 101.2]: 0.5 s away
    assert len(match_candidates(a, c, CombineSettings(window_s=0.01, match="extent"))[0]) == 0
    assert len(match_candidates(a, c, CombineSettings(window_s=0.6, match="extent"))[0]) == 1


def test_combine_paths_merges_and_keeps_both_paths():
    w = _table([100.0, 300.0], [0.01, 5.0], "gnn_logit", "w")
    t = _table([100.005, 500.0], [0.1, 0.02], "track_statistic", "t")
    out = combine_paths(w, t, CombineSettings(window_s=0.02), rank_w="gnn_logit", rank_t="track_statistic")
    assert len(out) == 3
    assert list(out.found_by) == ["w+t", "w", "t"]
    assert list(out.n_paths) == [2, 1, 1]
    merged = out.iloc[0]
    assert merged.time_path == "w" and merged.gps_candidate == 100.0 and merged.event_instant == "w"
    assert merged.far_per_day == pytest.approx(0.02) and merged.far_path == "w"
    assert merged.far_per_day_w == pytest.approx(0.01) and merged.far_per_day_t == pytest.approx(0.1)
    assert merged.statistic_w == 2.0 and merged.statistic_t == 2.0
    assert merged.combine_dt_s == pytest.approx(0.005)
    assert merged.instant_H1 == 100.0 and merged.instant_H1_t == pytest.approx(100.005)
    only_w = out.iloc[1]
    assert only_w.far_per_day == pytest.approx(10.0) and np.isnan(only_w.far_per_day_t)
    assert np.isnan(only_w.statistic_t) and np.isnan(only_w.combine_dt_s)
    only_t = out.iloc[2]
    assert only_t.time_path == "t" and only_t.gps_candidate == 500.0 and only_t.far_per_day == pytest.approx(0.04)
    assert np.isnan(only_t.candidate_id_w) and only_t.candidate_id_t == 1
    assert list(out.trials_factor) == [2.0, 2.0, 2.0]
    assert out.combined_statistic.iloc[0] == pytest.approx(-np.log10(0.02))
    assert out.attrs["paths"] == ["w", "t"] and out.attrs["trials_factor"] == 2.0


def test_time_preference_and_trials_are_parameters():
    w = _table([100.0], [0.01], "gnn_logit", "w")
    t = _table([100.005], [0.1], "track_statistic", "t")
    s = CombineSettings(window_s=0.02, trials_factor=1.0, time_preference=("t", "w"))
    out = combine_paths(w, t, s)
    assert out.time_path.iloc[0] == "t" and out.gps_candidate.iloc[0] == pytest.approx(100.005)
    assert out.far_per_day.iloc[0] == pytest.approx(0.01)
    assert out.event_instant.iloc[0] == "t"


def test_empty_and_single_path():
    w = _table([100.0], [0.01], "gnn_logit", "w")
    out = combine_paths(w, w.iloc[:0], CombineSettings(window_s=0.02))
    assert len(out) == 1 and out.found_by.iloc[0] == "w" and out.far_per_day.iloc[0] == pytest.approx(0.02)
    out = combine_paths(w.iloc[:0], w.iloc[:0], CombineSettings(window_s=0.02))
    assert len(out) == 0
    out = combine_candidates({"w": w})
    assert out.far_per_day.iloc[0] == pytest.approx(0.01) and out.trials_factor.iloc[0] == 1.0


def test_missing_rate_is_an_error():
    w = _table([100.0], [0.01], "gnn_logit", "w").drop(columns=["far_per_day"])
    with pytest.raises(ValueError):
        combine_paths(w, w)
