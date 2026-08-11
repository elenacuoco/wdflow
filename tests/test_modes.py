import numpy as np
import pandas as pd
import pytest

from wdf.analysis.modes import (
    DetectionMode, compare_modes, mode_roc, mode_threshold,
)


def _mode(name, foreground, background, n_injections=100, livetime_days=1.0,
          statistic="stat"):
    return DetectionMode(
        name=name,
        foreground=pd.DataFrame({statistic: np.asarray(foreground, dtype=float)}),
        background=pd.DataFrame({statistic: np.asarray(background, dtype=float)}),
        statistic=statistic,
        n_injections=n_injections,
        livetime_days=livetime_days,
    )


def test_a_mode_producing_more_candidates_earns_nothing_for_them():
    """Extra candidates raise the mode's own threshold, so they do not pay."""
    background = np.arange(1.0, 101.0)
    foreground = np.arange(1.0, 51.0)
    lean = _mode("lean", foreground, background)
    # The same signals and the same livetime, but ten times the background:
    # every candidate the mode adds is a candidate it has to clear.
    noisy = _mode("noisy", foreground, np.repeat(background, 10))

    table = compare_modes([lean, noisy], far_targets=(1.0,)).set_index("mode")
    assert table.loc["noisy", "threshold"] >= table.loc["lean", "threshold"]
    assert table.loc["noisy", "efficiency"] <= table.loc["lean", "efficiency"]


def test_the_denominator_counts_injections_that_produced_nothing():
    """An injection missing from the foreground is missed, not absent."""
    background = np.arange(1.0, 101.0)
    found = np.full(10, 200.0)
    complete = _mode("complete", found, background, n_injections=10)
    partial = _mode("partial", found, background, n_injections=100)

    table = compare_modes([complete, partial], far_targets=(1.0,)).set_index("mode")
    assert table.loc["complete", "efficiency"] == pytest.approx(1.0)
    assert table.loc["partial", "efficiency"] == pytest.approx(0.1)


def test_a_background_too_short_for_the_rate_is_not_a_measurement():
    """Asking for a rate the livetime cannot resolve returns no number."""
    mode = _mode("brief", [10.0], [1.0, 2.0], livetime_days=1e-4)
    row = compare_modes([mode], far_targets=(0.1,)).iloc[0]
    assert not bool(row.measurable)
    assert np.isnan(row.efficiency)


def test_the_roc_only_visits_rates_the_background_reaches():
    """Every point is a threshold some background candidate sits at."""
    background = np.arange(1.0, 51.0)
    mode = _mode("roc", np.arange(1.0, 26.0), background, n_injections=25,
                 livetime_days=2.0)
    curve = mode_roc(mode, n_points=10)

    assert len(curve)
    assert set(curve.threshold).issubset(set(background))
    assert curve.far_per_day.is_monotonic_increasing
    assert curve.efficiency.is_monotonic_increasing
    assert curve.far_per_day.max() <= len(background) / mode.livetime_days


def test_the_threshold_is_the_one_the_rate_implies():
    """The rate and the threshold are two readings of the same background."""
    background = np.arange(1.0, 101.0)
    mode = _mode("t", [1.0], background, livetime_days=1.0)
    cut = mode_threshold(mode, far_per_day=10.0)
    assert np.count_nonzero(background >= cut) == pytest.approx(10, abs=1)


def test_a_missing_statistic_names_the_mode_and_the_side():
    mode = DetectionMode(
        name="network", foreground=pd.DataFrame({"a": [1.0]}),
        background=pd.DataFrame({"b": [1.0]}), statistic="b",
        n_injections=1, livetime_days=1.0)
    with pytest.raises(KeyError, match="network"):
        mode.scores()
