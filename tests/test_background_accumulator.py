"""The accumulated background gives the thresholds the held one gives."""
import numpy as np
import pandas as pd

from wdf.analysis.background import BackgroundAccumulator
from wdf.analysis.evaluation import threshold_at_far
from wdf.analysis.robust_events import (CoincidenceConfig, FARConfig,
                                        IndexedCoincidenceFinder, TimeSlideFAR)

SPAN = 4000.0
COUNT = 900


def _events(seed):
    rng = np.random.default_rng(seed)
    gps = np.sort(rng.uniform(0.0, SPAN, COUNT))
    duration = rng.uniform(0.05, 0.4, COUNT)
    return pd.DataFrame(dict(gpsStart=gps, gpsEnd=gps + duration,
                             gpsPeak=gps + duration / 2,
                             gpsCentroid=gps + duration / 2,
                             duration=duration,
                             freqMin=np.full(COUNT, 40.0),
                             freqMax=np.full(COUNT, 300.0),
                             EnWDF=rng.lognormal(1.6, 0.5, COUNT)))


def _slides():
    events = {"H1": _events(1), "L1": _events(2)}
    bounds = {ifo: (0.0, SPAN) for ifo in events}
    finder = IndexedCoincidenceFinder(
        CoincidenceConfig(light_travel_time_s=0.01))
    return events, bounds, finder, FARConfig(n_slides=40, min_shift_s=5.0,
                                             seed=3)


def test_reduce_sees_every_slide_and_holds_none():
    events, bounds, finder, config = _slides()
    held = TimeSlideFAR(finder, config).background_distribution(events, bounds)
    accumulator = BackgroundAccumulator(["network_enwdf"], keep=200,
                                        extras=["slide_index"])
    empty = TimeSlideFAR(finder, config).background_distribution(
        events, bounds, reduce=lambda table, live: accumulator.add(table, live))

    assert len(empty) == 0
    assert accumulator.total == len(held)
    assert accumulator.n_slides == held.attrs["n_slides"]
    assert np.isclose(accumulator.livetime_s,
                      held.attrs["total_livetime_s"])
    assert list(accumulator.tail("network_enwdf").columns) == [
        "network_enwdf", "slide_index"]


def test_a_threshold_inside_the_kept_tail_is_the_held_one():
    events, bounds, finder, config = _slides()
    held = TimeSlideFAR(finder, config).background_distribution(events, bounds)
    accumulator = BackgroundAccumulator(["network_enwdf"], keep=200)
    TimeSlideFAR(finder, config).background_distribution(
        events, bounds, reduce=lambda table, live: accumulator.add(table, live))

    days = held.attrs["total_livetime_s"] / 86400.0
    values = held["network_enwdf"].to_numpy(dtype=float)
    for rate in (1.0, 10.0, 100.0):
        wanted = threshold_at_far(values, rate, days)
        got, exact = accumulator.threshold("network_enwdf", rate, days)
        if not exact:
            continue
        assert np.isclose(got, wanted), f"at {rate} per day"


def test_a_rate_the_livetime_cannot_resolve_has_no_threshold():
    accumulator = BackgroundAccumulator(["network_enwdf"], keep=10)
    accumulator.add(pd.DataFrame({"network_enwdf": [1.0, 2.0]}), 3600.0)
    value, exact = accumulator.threshold("network_enwdf", 1e-9)
    assert np.isnan(value) and not exact


def test_a_ranking_the_background_never_carried_has_no_threshold():
    accumulator = BackgroundAccumulator(["gnn_logit"], keep=10)
    accumulator.add(pd.DataFrame({"network_enwdf": [1.0]}), 3600.0)
    value, exact = accumulator.threshold("gnn_logit", 1.0, 1.0)
    assert np.isnan(value) and not exact
