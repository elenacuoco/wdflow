from wdf.analysis.clustering import TriggerClusterer
from wdf.analysis.coincidence import CoincidenceFinder
from wdf.analysis.significance import BackgroundEstimator, pool_backgrounds

from _synth import synth_raw_triggers


def _clustered_and_bounds(ifo, burst_gps, seed):
    df = synth_raw_triggers(ifo, n_background=40, gps0=1000.0, span_s=500.0,
                             seed=seed, burst_gps=burst_gps, burst_n=6, burst_snr=15.0)
    tc = TriggerClusterer(time_eps_s=0.5, freq_eps_hz=50.0, min_samples=2)
    events = tc.clustered_events(tc.fit_predict(df))
    return events, (df["gpsPeak"].min(), df["gpsPeak"].max())


def test_background_distribution_and_fap():
    h1_events, h1_bounds = _clustered_and_bounds("H1", burst_gps=1250.0, seed=1)
    l1_events, l1_bounds = _clustered_and_bounds("L1", burst_gps=1250.002, seed=2)
    clustered = {"H1": h1_events, "L1": l1_events}
    bounds = {"H1": h1_bounds, "L1": l1_bounds}

    # burst_gps offsets in _synth are drawn from +/-0.2s independently per IFO,
    # so the two clusters' peak times can differ by up to ~0.4s -- widen the
    # jitter budget enough to comfortably cover that for this synthetic check.
    cf = CoincidenceFinder(timing_jitter_s=0.3)
    real_candidates = cf.find(clustered)
    assert len(real_candidates) > 0

    be = BackgroundEstimator(cf, n_slides=50, min_shift_s=2.0, seed=0)
    bg = be.background_distribution(clustered, bounds)
    # background rows (if any) must carry a slide_index and stay within [0, n_slides)
    if not bg.empty:
        assert bg["slide_index"].between(0, be.n_slides - 1).all()

    best = real_candidates.loc[real_candidates["network_snr"].idxmax()]
    result = be.false_alarm_probability(best, bg, segment_duration_s=h1_bounds[1] - h1_bounds[0])
    assert 0.0 <= result["fap"] <= 1.0
    assert result["n_slides"] == 50
    assert "far_per_day" in result


def test_background_distribution_with_find_network_for_three_ifos():
    h1_events, h1_bounds = _clustered_and_bounds("H1", burst_gps=1250.0, seed=1)
    l1_events, l1_bounds = _clustered_and_bounds("L1", burst_gps=1250.002, seed=2)
    v1_events, v1_bounds = _clustered_and_bounds("V1", burst_gps=1250.010, seed=7)
    clustered = {"H1": h1_events, "L1": l1_events, "V1": v1_events}
    bounds = {"H1": h1_bounds, "L1": l1_bounds, "V1": v1_bounds}

    cf = CoincidenceFinder(timing_jitter_s=0.3)
    real_candidates = cf.find_network(clustered, min_ifos=3)

    be = BackgroundEstimator(cf, n_slides=50, min_shift_s=2.0, seed=0)
    bg = be.background_distribution(clustered, bounds, finder_method="find_network", min_ifos=3)
    if not bg.empty:
        assert (bg["n_ifos"] == 3).all()

    if len(real_candidates) > 0:
        best = real_candidates.loc[real_candidates["network_snr"].idxmax()]
        result = be.false_alarm_probability(best, bg, segment_duration_s=h1_bounds[1] - h1_bounds[0])
        assert 0.0 <= result["fap"] <= 1.0
        assert "far_per_day" in result


def test_pool_backgrounds_tags_segment_id():
    import pandas as pd
    a = pd.DataFrame({"network_snr": [1.0, 2.0]})
    b = pd.DataFrame({"network_snr": [3.0]})
    pooled = pool_backgrounds({"seg_a": a, "seg_b": b})
    assert set(pooled["segment_id"]) == {"seg_a", "seg_b"}
    assert len(pooled) == 3
