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
