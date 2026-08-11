"""The track an event leaves, and the properties its descriptors must have."""
import numpy as np
import pytest

from wdf.analysis.ridge import (
    RIDGE_FEATURES, event_ridge, event_ridge_features, ridge_features,
)


def _track(f_start, f_end, n=32, duration=1.0, width=0.02, energy=4.0):
    """Tiles lying on a straight sweep in log frequency."""
    t = np.linspace(0.0, duration, n)
    f = np.exp(np.linspace(np.log(f_start), np.log(f_end), n))
    return (t, t + width, f * 0.95, f * 1.05, np.full(n, energy))


def test_a_sweep_is_a_track_and_noise_is_not():
    rng = np.random.default_rng(0)
    sweep = event_ridge_features(*_track(50.0, 400.0))

    n = 32
    t = np.linspace(0.0, 1.0, n)
    f = np.exp(rng.uniform(np.log(30.0), np.log(900.0), n))
    scatter = event_ridge_features(t, t + 0.02, f * 0.95, f * 1.05,
                                   np.full(n, 4.0))

    assert sweep["ridge_scatter"] < 0.1
    assert scatter["ridge_scatter"] > 5.0 * sweep["ridge_scatter"]
    assert sweep["ridge_monotonicity"] > 0.9
    assert scatter["ridge_monotonicity"] < sweep["ridge_monotonicity"]
    assert sweep["ridge_continuity"] < scatter["ridge_continuity"]


def test_a_falling_track_scores_like_a_rising_one():
    """The descriptors must not encode which way a compact binary sweeps."""
    up = event_ridge_features(*_track(50.0, 400.0))
    down = event_ridge_features(*_track(400.0, 50.0))

    assert up["ridge_monotonicity"] == pytest.approx(down["ridge_monotonicity"])
    assert up["ridge_scatter"] == pytest.approx(down["ridge_scatter"], abs=1e-9)
    assert up["ridge_occupancy"] == pytest.approx(down["ridge_occupancy"])
    # The slope keeps its sign, which is a measurement and not a preference.
    assert up["ridge_slope"] == pytest.approx(-down["ridge_slope"], rel=1e-6)


def test_the_slope_is_octaves_per_second():
    """Three octaves over one second, whatever the band they are in."""
    low = event_ridge_features(*_track(25.0, 200.0, duration=1.0))
    high = event_ridge_features(*_track(100.0, 800.0, duration=1.0))
    assert low["ridge_slope"] == pytest.approx(3.0, rel=0.05)
    assert high["ridge_slope"] == pytest.approx(3.0, rel=0.05)


def test_a_gap_lowers_the_occupancy_and_is_not_interpolated():
    t, t_hi, f_lo, f_hi, e = _track(50.0, 400.0, n=32)
    keep = (t < 0.3) | (t > 0.7)
    full = event_ridge(t, t_hi, f_lo, f_hi, e, n_bins=32)
    holed = event_ridge(t[keep], t_hi[keep], f_lo[keep], f_hi[keep], e[keep],
                        n_bins=32)
    assert np.isfinite(full[1]).mean() > np.isfinite(holed[1]).mean()
    assert np.isnan(holed[1]).any()


def test_the_loudest_tile_of_a_bin_is_the_one_taken():
    # Two tiles in the same instant, the quiet one at a different band.
    t = np.array([0.10, 0.10])
    f_lo = np.array([95.0, 400.0])
    energy = np.array([1.0, 25.0])
    _, frequency, loudness = event_ridge(t, t + 0.01, f_lo, f_lo * 1.05,
                                         energy, n_bins=4)
    taken = np.isfinite(frequency)
    assert loudness[taken][0] == 25.0
    assert np.exp(frequency[taken][0]) > 200.0


def test_too_few_tiles_give_no_descriptors_rather_than_a_number():
    out = event_ridge_features([1.0], [1.1], [90.0], [110.0], [4.0])
    assert out["ridge_occupancy"] > 0.0
    for name in ("ridge_slope", "ridge_scatter", "ridge_monotonicity",
                 "ridge_continuity"):
        assert np.isnan(out[name])


def test_no_tiles_at_all_leave_an_occupancy_of_zero_and_nothing_else():
    """Zero bins occupied is a measurement; a slope over no tiles is not."""
    out = event_ridge_features([], [], [], [], [])
    assert set(out) == set(RIDGE_FEATURES)
    assert out["ridge_occupancy"] == 0.0
    assert all(np.isnan(out[name]) for name in RIDGE_FEATURES
               if name != "ridge_occupancy")
