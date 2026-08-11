import numpy as np
import pandas as pd
import pytest

from wdf.analysis.injections import efficiency, false_alarms, match_injections


def _candidates(times, snrs):
    return pd.DataFrame({"gpsPeak": times, "EnWDF": snrs})


def _injections(times, snrs=None):
    n = len(times)
    return pd.DataFrame({"gps": times, "network_snr": snrs if snrs is not None else np.full(n, 10.0)})


def test_injection_inside_the_window_is_found():
    matched = match_injections(_candidates([100.2], [7.0]), _injections([100.0]), window_s=0.5)
    assert bool(matched.loc[0, "found"])
    assert matched.loc[0, "recovered_snr"] == pytest.approx(7.0)
    assert matched.loc[0, "dt_s"] == pytest.approx(0.2)


def test_injection_outside_the_window_is_missed():
    matched = match_injections(_candidates([101.0], [7.0]), _injections([100.0]), window_s=0.5)
    assert not bool(matched.loc[0, "found"])
    assert matched.loc[0, "candidate_index"] == -1
    assert np.isnan(matched.loc[0, "recovered_snr"])


def test_loudest_candidate_in_the_window_is_the_one_recorded():
    cands = _candidates([99.9, 100.1, 100.3], [3.0, 12.0, 5.0])
    matched = match_injections(cands, _injections([100.0]), window_s=0.5)
    assert matched.loc[0, "recovered_snr"] == pytest.approx(12.0)
    assert matched.loc[0, "candidate_index"] == 1


def test_no_candidates_gives_all_missed():
    matched = match_injections(_candidates([], []), _injections([100.0, 200.0]))
    assert not matched["found"].any()


def test_match_preserves_injection_rows_and_order():
    inj = _injections([100.0, 200.0, 300.0])
    inj["subclass"] = ["bbh", "blip", "bns"]
    matched = match_injections(_candidates([200.05], [9.0]), inj, window_s=0.5)
    assert list(matched["subclass"]) == ["bbh", "blip", "bns"]
    assert list(matched["found"]) == [False, True, False]


def test_false_alarms_excludes_candidates_near_injections():
    cands = _candidates([100.1, 500.0, 900.2], [5.0, 6.0, 7.0])
    extra = false_alarms(cands, _injections([100.0, 900.0]), window_s=0.5)
    assert list(extra["gpsPeak"]) == [500.0]


def test_false_alarms_with_a_single_injection():
    cands = _candidates([100.1, 500.0], [5.0, 6.0])
    extra = false_alarms(cands, _injections([100.0]), window_s=0.5)
    assert list(extra["gpsPeak"]) == [500.0]


def test_efficiency_is_one_when_everything_is_found():
    inj = _injections(np.arange(10) * 100.0, np.linspace(5, 50, 10))
    matched = match_injections(_candidates(np.arange(10) * 100.0, np.full(10, 8.0)), inj)
    curve = efficiency(matched, bins=np.array([0.0, 100.0]))
    assert curve.loc[0, "efficiency"] == pytest.approx(1.0)
    assert curve.loc[0, "n"] == 10


def test_efficiency_rises_with_injected_snr():
    """Only the loud half is recovered, so the low bin sits below the high bin."""
    inj = _injections(np.arange(20) * 100.0, np.concatenate([np.full(10, 5.0), np.full(10, 40.0)]))
    found_times = (np.arange(20) * 100.0)[10:]
    matched = match_injections(_candidates(found_times, np.full(10, 8.0)), inj)
    curve = efficiency(matched, bins=np.array([0.0, 20.0, 60.0]))
    assert curve.loc[0, "efficiency"] == pytest.approx(0.0)
    assert curve.loc[1, "efficiency"] == pytest.approx(1.0)


def test_efficiency_can_be_grouped_by_class():
    inj = _injections(np.arange(4) * 100.0, np.full(4, 10.0))
    inj["subclass"] = ["bbh", "bbh", "blip", "blip"]
    matched = match_injections(_candidates([0.0, 100.0], [8.0, 8.0]), inj)
    curve = efficiency(matched, bins=np.array([0.0, 20.0]), group_column="subclass")
    by_class = curve.set_index("subclass")["efficiency"]
    assert by_class["bbh"] == pytest.approx(1.0)
    assert by_class["blip"] == pytest.approx(0.0)


def test_a_long_event_is_matched_by_its_extent_not_by_its_time():
    """A chirp's energy sits well before its merger, so an event that spans the
    signal correctly reports a time seconds away from the injection. Matching
    the instant would call that a miss."""
    from wdf.analysis.injections import match_injections

    merger = 1000.0
    event = pd.DataFrame([{
        "gpsPeak": merger - 4.0,
        "gpsCentroid": merger - 4.0,
        "gpsStart": merger - 8.0,
        "duration": 9.0,
        "EnWDF": 12.0,
    }])
    injections = pd.DataFrame([{"gps": merger}])

    matched = match_injections(event, injections, window_s=0.5,
                               candidate_time="gpsCentroid")
    assert bool(matched.found.iloc[0])
    assert matched.recovered_snr.iloc[0] == 12.0


def test_an_event_that_ends_before_the_injection_is_not_a_match():
    from wdf.analysis.injections import match_injections

    event = pd.DataFrame([{
        "gpsPeak": 990.0, "gpsCentroid": 990.0,
        "gpsStart": 988.0, "duration": 2.0, "EnWDF": 12.0,
    }])
    injections = pd.DataFrame([{"gps": 1000.0}])

    assert not bool(match_injections(event, injections, window_s=0.5,
                                     candidate_time="gpsCentroid").found.iloc[0])


def test_a_candidate_with_no_extent_still_matches_on_its_instant():
    """One rule covers both: no extent means the candidate covers its own time."""
    from wdf.analysis.injections import match_injections

    event = pd.DataFrame([{"gpsPeak": 1000.2, "EnWDF": 9.0}])
    injections = pd.DataFrame([{"gps": 1000.0}])

    assert bool(match_injections(event, injections, window_s=0.5).found.iloc[0])
    assert not bool(match_injections(event, injections, window_s=0.1).found.iloc[0])


def test_a_long_event_covering_an_injection_is_not_a_false_alarm():
    from wdf.analysis.injections import false_alarms

    candidates = pd.DataFrame([
        {"gpsPeak": 996.0, "gpsStart": 992.0, "duration": 9.0, "EnWDF": 12.0},
        {"gpsPeak": 500.0, "gpsStart": 499.0, "duration": 1.0, "EnWDF": 7.0},
    ])
    injections = pd.DataFrame([{"gps": 1000.0}])

    remaining = false_alarms(candidates, injections, window_s=0.5)
    assert len(remaining) == 1
    assert float(remaining.gpsPeak.iloc[0]) == 500.0


def _unclaimed_input(times, spans=0.0, statistic=None):
    import pandas as pd
    times = np.asarray(times, dtype=float)
    spans = np.full(times.shape, spans) if np.isscalar(spans) else np.asarray(spans)
    frame = pd.DataFrame(dict(gpsPeak=times, gpsStart=times, duration=spans))
    frame["EnWDF"] = np.arange(len(times), 0, -1) if statistic is None else statistic
    return frame


def test_a_second_candidate_on_our_own_injection_is_not_unexplained():
    """The looser rule: covering the injection is enough, being loudest is not."""
    import pandas as pd
    from wdf.analysis.injections import unclaimed_candidates

    injections = pd.DataFrame(dict(gps=[100.0]))
    # Two candidates on the one injection, and one far away.
    candidates = _unclaimed_input([100.0, 100.2, 500.0])
    left = unclaimed_candidates(candidates, injections, window_s=0.5,
                                statistic="EnWDF")
    assert len(left) == 1
    assert float(left.gpsPeak.iloc[0]) == 500.0


def test_a_long_candidate_is_claimed_by_an_injection_inside_it():
    import pandas as pd
    from wdf.analysis.injections import unclaimed_candidates

    injections = pd.DataFrame(dict(gps=[120.0]))
    candidates = _unclaimed_input([100.0], spans=[50.0])
    assert unclaimed_candidates(candidates, injections, window_s=0.0).empty


def test_the_list_is_ordered_and_can_be_cut():
    import pandas as pd
    from wdf.analysis.injections import unclaimed_candidates

    candidates = _unclaimed_input([10.0, 20.0, 30.0], statistic=[3.0, 9.0, 5.0])
    left = unclaimed_candidates(candidates, pd.DataFrame(dict(gps=[])),
                                statistic="EnWDF", limit=2)
    assert list(left.EnWDF) == [9.0, 5.0]


def test_with_no_injections_nothing_is_claimed():
    import pandas as pd
    from wdf.analysis.injections import unclaimed_candidates

    candidates = _unclaimed_input([10.0, 20.0])
    assert len(unclaimed_candidates(candidates, None)) == 2
    assert len(unclaimed_candidates(candidates, pd.DataFrame())) == 2


def test_an_unknown_statistic_is_refused():
    import pandas as pd
    from wdf.analysis.injections import unclaimed_candidates

    with pytest.raises(KeyError, match="network_enwdf"):
        unclaimed_candidates(_unclaimed_input([1.0]), pd.DataFrame(dict(gps=[])),
                             statistic="network_enwdf")
