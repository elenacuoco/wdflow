from wdf.analysis.clustering import TriggerClusterer
from wdf.analysis.coincidence import CoincidenceFinder

from _synth import synth_raw_triggers


def _clustered(ifo, burst_gps, seed):
    df = synth_raw_triggers(ifo, n_background=30, gps0=1000.0, span_s=200.0,
                             seed=seed, burst_gps=burst_gps, burst_n=6, burst_snr=15.0)
    tc = TriggerClusterer(time_eps_s=0.5, freq_eps_hz=50.0, min_samples=2)
    return tc.clustered_events(tc.fit_predict(df))


def test_finds_true_coincidence_within_window():
    clustered = {
        "H1": _clustered("H1", burst_gps=1100.000, seed=1),
        "L1": _clustered("L1", burst_gps=1100.003, seed=2),  # well within H1-L1 window
    }
    cf = CoincidenceFinder(timing_jitter_s=0.05)
    candidates = cf.find(clustered)
    assert len(candidates) > 0
    near = candidates[(candidates["gps_candidate"] - 1100.0).abs() < 1.0]
    assert len(near) >= 1
    best = near.sort_values("network_snr", ascending=False).iloc[0]
    assert best["network_snr"] > 10.0
    assert abs(best["dt_s"]) < cf.coincidence_window("H1", "L1")


def test_no_coincidence_when_far_apart():
    clustered = {
        "H1": _clustered("H1", burst_gps=1050.0, seed=3),
        "L1": _clustered("L1", burst_gps=1150.0, seed=4),  # 100s apart, no real coincidence
    }
    cf = CoincidenceFinder(timing_jitter_s=0.05)
    candidates = cf.find(clustered)
    near_h1_burst = candidates[(candidates["gps_candidate"] - 1050.0).abs() < 0.5]
    assert len(near_h1_burst) == 0


def test_unknown_ifo_pair_raises():
    cf = CoincidenceFinder()
    try:
        cf.coincidence_window("H1", "V1")
        assert False, "expected KeyError for unknown baseline"
    except KeyError:
        pass
