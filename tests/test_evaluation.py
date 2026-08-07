"""Comparing ranking statistics at a fixed false-alarm rate, and the split that
has to happen first for a learned statistic to be compared honestly."""
import numpy as np
import pandas as pd
import pytest

from wdf.analysis.evaluation import (compare_statistics, efficiency_at_far,
                                     temporal_split, threshold_at_far)


def _candidates(n, t0=0.0, span=100.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "gps_candidate": np.sort(rng.uniform(t0, t0 + span, n)),
        "network_enwdf": rng.gamma(2.0, 2.0, n) + 4.0,
        "gnn_score": rng.uniform(0.0, 1.0, n),
    })


def test_the_split_cuts_in_time_not_at_random():
    frame = _candidates(200, span=100.0)
    train, test = temporal_split(frame, train_fraction=0.5)

    assert train.sum() + test.sum() == len(frame)
    assert not np.any(train & test)
    assert frame.gps_candidate[train].max() < frame.gps_candidate[test].min()


def test_the_split_divides_the_span_not_the_rows():
    """A quiet stretch contributes few rows, which is the honest behaviour: the
    boundary is a time, not a quantile of the candidates."""
    dense = _candidates(300, t0=0.0, span=10.0, seed=1)
    sparse = _candidates(10, t0=10.0, span=90.0, seed=2)
    frame = pd.concat([dense, sparse], ignore_index=True)

    train, test = temporal_split(frame, train_fraction=0.5)

    assert train.sum() > test.sum()
    assert frame.gps_candidate[train].max() < 50.0


def test_a_degenerate_span_is_refused():
    frame = pd.DataFrame({"gps_candidate": [10.0, 10.0, 10.0]})
    with pytest.raises(ValueError, match="span no time"):
        temporal_split(frame)


def test_a_missing_column_is_named():
    with pytest.raises(ValueError, match="gps_candidate"):
        temporal_split(pd.DataFrame({"t": [1.0, 2.0]}))


def test_the_threshold_admits_the_requested_number_of_background_events():
    background = np.arange(100, dtype=float)   # 0 .. 99
    # ten days of background, one per day -> ten events allowed above threshold
    threshold = threshold_at_far(background, livetime_days=10.0, far_per_day=1.0)
    assert int((background >= threshold).sum()) == 10


def test_a_rate_the_livetime_cannot_resolve_is_reported_as_unmeasurable():
    """Allowing less than one background event says nothing about where the
    threshold is; returning the loudest background instead would look like a
    measurement, and would score every candidate tied at a saturated
    statistic's maximum as recovered."""
    background = np.arange(100, dtype=float)
    assert np.isnan(threshold_at_far(background, livetime_days=1.0, far_per_day=1e-6))

    result = efficiency_at_far(np.array([99.0, 99.0]), background, n_injections=2,
                               livetime_days=1.0, far_per_day=1e-6)
    assert result["measurable"] is False
    assert np.isnan(result["efficiency"])


def test_a_rate_the_background_never_reaches_needs_no_cut():
    background = np.arange(10, dtype=float)
    assert threshold_at_far(background, livetime_days=1.0, far_per_day=100.0) == float("-inf")


def test_efficiency_counts_injections_that_produced_no_candidate():
    """The denominator is the injections made, not the ones recovered: an
    injection that never became a candidate is a miss, not an absentee."""
    background = np.arange(100, dtype=float)
    found = np.array([95.0, 96.0, 97.0])

    result = efficiency_at_far(found, background, n_injections=10,
                               livetime_days=10.0, far_per_day=1.0)

    assert result["n_found"] == 3
    assert result["efficiency"] == pytest.approx(0.3)


def test_a_statistic_that_separates_beats_one_that_does_not():
    rng = np.random.default_rng(5)
    n_injections = 200
    background = pd.DataFrame({
        "separating": rng.normal(0.0, 1.0, 5000),
        "blind": rng.normal(0.0, 1.0, 5000),
    })
    foreground = pd.DataFrame({
        "separating": rng.normal(4.0, 1.0, n_injections),
        "blind": rng.normal(0.0, 1.0, n_injections),
    })

    table = compare_statistics(foreground, background,
                               statistics=("separating", "blind"),
                               n_injections=n_injections, livetime_days=10.0,
                               far_targets=(1.0,))

    separating = table.loc[table.statistic == "separating", "efficiency"].iloc[0]
    blind = table.loc[table.statistic == "blind", "efficiency"].iloc[0]
    assert separating > 0.8
    assert blind < 0.1


def test_a_statistic_missing_from_either_side_is_refused():
    """Foreground and background must be ranked on the same quantity."""
    foreground = pd.DataFrame({"a": [1.0], "b": [1.0]})
    background = pd.DataFrame({"a": [1.0]})
    with pytest.raises(KeyError, match="background"):
        compare_statistics(foreground, background, ("a", "b"),
                           n_injections=1, livetime_days=1.0)
