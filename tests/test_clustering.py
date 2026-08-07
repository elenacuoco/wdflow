import numpy as np

from wdf.analysis.clustering import TriggerClusterer
from wdf.analysis.io import clean_triggers

from _synth import synth_raw_triggers


def test_burst_forms_one_dominant_cluster():
    df = synth_raw_triggers("H1", n_background=40, gps0=1000.0, span_s=100.0,
                             seed=1, burst_gps=1050.0, burst_n=8, burst_snr=15.0)
    tc = TriggerClusterer(time_eps_s=0.5, freq_eps_hz=50.0, min_samples=2)
    labeled = tc.fit_predict(df)
    events = tc.clustered_events(labeled)

    near_burst = events[(events["gpsPeak"] - 1050.0).abs() < 1.0]
    assert len(near_burst) >= 1
    top = near_burst.sort_values("n_triggers", ascending=False).iloc[0]
    assert top["n_triggers"] >= 5
    assert top["EnWDF"] > 10.0


def test_noise_points_kept_as_singletons():
    df = synth_raw_triggers("H1", n_background=20, gps0=1000.0, span_s=200.0, seed=2)
    tc = TriggerClusterer(time_eps_s=0.5, freq_eps_hz=50.0, min_samples=2)
    labeled = tc.fit_predict(df)
    # sparse background alone (no burst) -> mostly/all noise
    assert (labeled["cluster_id"] == -1).sum() > 0
    events = tc.clustered_events(labeled)
    assert len(events) == len(df[labeled["cluster_id"] == -1]) + labeled[labeled["cluster_id"] != -1]["cluster_id"].nunique()


def test_rejects_multi_ifo_input():
    df = synth_raw_triggers("H1", n_background=5, gps0=1000.0, span_s=10.0, seed=3)
    df2 = synth_raw_triggers("L1", n_background=5, gps0=1000.0, span_s=10.0, seed=4)
    import pandas as pd
    mixed = pd.concat([df, df2], ignore_index=True)
    tc = TriggerClusterer()
    try:
        tc.fit_predict(mixed)
        assert False, "expected ValueError for multi-IFO input"
    except ValueError:
        pass


def test_greedy_method_runs():
    df = synth_raw_triggers("H1", n_background=20, gps0=1000.0, span_s=100.0,
                             seed=5, burst_gps=1050.0, burst_n=6)
    tc = TriggerClusterer(method="greedy", time_eps_s=0.5, freq_eps_hz=50.0, min_samples=2)
    labeled = tc.fit_predict(df)
    events = tc.clustered_events(labeled)
    assert len(events) > 0


def test_clean_triggers_drops_artifacts():
    df = synth_raw_triggers("H1", n_background=10, gps0=1000.0, span_s=50.0, seed=6)
    df.loc[0, "snrPeak"] = 1e20  # inject a WDF numerical artifact
    cleaned = clean_triggers(df, snr_ceiling=500.0, edge_guard_s=0.0)
    assert cleaned["snrPeak"].max() < 500.0
    assert len(cleaned) == len(df) - 1


def test_clustered_events_cluster_id_joins_back_to_fit_predict_output():
    """Regression test for a real bug: clustered_events() used to emit
    cluster_id as a string ("0", "n956") while fit_predict()'s own raw
    output keeps it int64 (0, -1) -- any join between the two silently
    matched zero rows, always, for every cluster including real
    multi-trigger ones.
    """
    df = synth_raw_triggers("H1", n_background=30, gps0=1000.0, span_s=100.0,
                             seed=9, burst_gps=1050.0, burst_n=8, burst_snr=15.0)
    tc = TriggerClusterer(time_eps_s=0.5, freq_eps_hz=50.0, min_samples=2)
    labeled = tc.fit_predict(df)
    events = tc.clustered_events(labeled)

    real_clusters = events[~events["is_noise"]]
    assert len(real_clusters) > 0, "fixture should produce at least one real (non-noise) cluster"
    for _, row in real_clusters.iterrows():
        members = labeled[labeled["cluster_id"] == row["cluster_id"]]
        assert len(members) == row["n_triggers"]

    noise_rows = events[events["is_noise"]]
    assert len(noise_rows) > 0, "fixture should produce at least one noise singleton"
    for _, row in noise_rows.iterrows():
        assert row["n_triggers"] == 1
        source = labeled.iloc[row["trigger_index"]]
        assert source["cluster_id"] == -1


def test_clustered_events_when_every_trigger_is_noise():
    # regression test: pandas' default "str" dtype breaks boolean-mask assignment
    # (group_key[noise_mask] = [...]) specifically when the mask selects every row
    df = synth_raw_triggers("H1", n_background=10, gps0=1000.0, span_s=50.0, seed=8)
    tc = TriggerClusterer(time_eps_s=0.5, freq_eps_hz=50.0, min_samples=2)
    labeled = tc.fit_predict(df)
    assert (labeled["cluster_id"] == -1).all()
    events = tc.clustered_events(labeled)
    assert len(events) == len(df)
