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


def test_the_track_fills_a_hole_and_says_which_bins_it_filled():
    from wdf.analysis.ridge import ridge_track

    log_f = np.log(np.array([100.0, np.nan, np.nan, 800.0]))
    track, measured = ridge_track(np.arange(4.0), log_f)

    assert measured.tolist() == [True, False, False, True]
    assert np.isfinite(track).all()
    # Linear in log frequency: one octave per bin over the three steps.
    assert np.exp(track) == pytest.approx([100.0, 200.0, 400.0, 800.0])


def test_the_track_holds_the_nearest_value_outside_the_occupied_range():
    from wdf.analysis.ridge import ridge_track

    log_f = np.log(np.array([np.nan, 100.0, 200.0, np.nan]))
    track, measured = ridge_track(np.arange(4.0), log_f)

    assert measured.tolist() == [False, True, True, False]
    assert np.exp(track) == pytest.approx([100.0, 100.0, 200.0, 200.0])


def test_the_track_reads_the_same_backwards_as_forwards():
    """Filling must carry no preferred direction of time."""
    from wdf.analysis.ridge import ridge_track

    log_f = np.log(np.array([50.0, np.nan, 150.0, np.nan, np.nan, 400.0]))
    forward, measured = ridge_track(np.arange(6.0), log_f)
    backward, measured_back = ridge_track(np.arange(6.0), log_f[::-1])

    assert np.allclose(forward, backward[::-1])
    assert measured.tolist() == measured_back[::-1].tolist()


def test_the_track_fills_a_falling_sweep_as_it_fills_a_rising_one():
    from wdf.analysis.ridge import ridge_track

    rising = np.log(np.array([50.0, np.nan, 200.0, np.nan, 800.0]))
    falling = np.log(np.array([800.0, np.nan, 200.0, np.nan, 50.0]))
    up, _ = ridge_track(np.arange(5.0), rising)
    down, _ = ridge_track(np.arange(5.0), falling)

    assert np.allclose(np.exp(up), np.exp(down)[::-1])


def test_a_filled_bin_is_not_counted_in_the_occupancy():
    from wdf.analysis.ridge import ridge_track

    t, t_hi, f_lo, f_hi, e = _track(50.0, 400.0, n=32)
    keep = (t < 0.3) | (t > 0.7)
    time, log_f, energy = event_ridge(t[keep], t_hi[keep], f_lo[keep],
                                      f_hi[keep], e[keep], n_bins=32)
    filled, measured = ridge_track(time, log_f)

    holed = ridge_features(time, log_f, energy)
    honest = ridge_features(np.where(measured, time, np.nanmean(time)), filled,
                            np.where(measured, energy, 1.0), measured=measured)
    assert honest["ridge_occupancy"] == pytest.approx(holed["ridge_occupancy"])
    assert honest["ridge_occupancy"] < 1.0


def test_the_existing_callers_of_the_descriptors_are_unchanged():
    """`measured` defaults to None, which is every finite bin, as before."""
    t, t_hi, f_lo, f_hi, e = _track(50.0, 400.0, n=32)
    keep = (t < 0.3) | (t > 0.7)
    ridge = event_ridge(t[keep], t_hi[keep], f_lo[keep], f_hi[keep], e[keep],
                        n_bins=32)
    assert ridge_features(*ridge) == ridge_features(*ridge, measured=None)
    assert ridge_features(*ridge)["ridge_occupancy"] == pytest.approx(
        np.isfinite(ridge[1]).mean())


def test_the_track_selects_the_tiles_it_crosses():
    from wdf.analysis.ridge import ridge_members, ridge_track

    # A track rising one octave per second, read over two seconds.
    times = np.linspace(0.0, 2.0, 5)
    track, _ = ridge_track(times, np.log(np.array([64.0, np.nan, 128.0,
                                                   np.nan, 256.0])), times)

    # Tiles on the track at 0.5 s and 1.5 s, and two off it.
    t_lo = np.array([0.45, 1.45, 0.45, 1.45])
    f_lo = np.array([64.0, 128.0, 512.0, 16.0])
    members = ridge_members(t_lo, t_lo + 0.1, f_lo, f_lo * 2.0, track, times)
    assert members.tolist() == [True, True, False, False]


def test_the_coarsest_band_is_read_as_half_its_upper_edge():
    from wdf.analysis.ridge import ridge_members

    times = np.array([0.0, 1.0])
    track = np.log(np.array([96.0, 96.0]))
    # A tile whose lower edge is zero: its band is read as [64, 128).
    inside = ridge_members([0.4], [0.6], [0.0], [128.0], track, times)
    outside = ridge_members([0.4], [0.6], [0.0], [64.0], track, times)
    assert inside.tolist() == [True]
    assert outside.tolist() == [False]


def test_the_membership_mask_labels_a_cluster():
    """What `ridge_members` returns is what `cluster_events` takes as labels."""
    import pandas as pd

    from wdf.analysis.pixel_graph import build_pixel_graph, cluster_events
    from wdf.analysis.ridge import ridge_members

    pixels = pd.DataFrame(dict(
        trigger_index=[0, 0, 1], ifo=["H1"] * 3, scale=[512.0] * 3,
        fs=[2048.0] * 3, t_lo=[0.0, 0.5, 1.0], t_hi=[0.1, 0.6, 1.1],
        f_lo=[64.0, 128.0, 512.0], f_hi=[128.0, 256.0, 1024.0],
        energy=[9.0, 9.0, 9.0], sigma=[1.0] * 3))
    graph = build_pixel_graph(pixels)
    nodes = graph.nodes
    times = np.array([0.0, 1.0])
    track = np.log(np.array([90.0, 180.0]))
    members = ridge_members(nodes.t_lo, nodes.t_hi, nodes.f_lo, nodes.f_hi,
                            track, times)
    events = cluster_events(graph, labels=members.astype(np.int64))
    assert members.sum() == 2
    assert len(events) == 2
