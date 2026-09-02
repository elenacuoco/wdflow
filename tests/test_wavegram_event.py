"""The event as a wavegram: a connected set of tiles, and what it must report.

The quantities here are the ones every later stage reads, so each test states
the arithmetic it expects rather than comparing against a stored number.
"""
import numpy as np
import pandas as pd
import pytest

from wdf.analysis.cluster_coefficients import score_events_by_reconstruction
from wdf.analysis.event_significance import (EventCalibration,
                                             out_of_sample_significance)
from wdf.analysis.pixel_graph import (PixelGraph, PixelGraphConfig,
                                      build_pixel_graph, cluster_events)
from wdf.analysis.scale import unique_tiles


def tiles(**overrides):
    """Three tiles of one transient and one tile far away from it."""
    frame = pd.DataFrame(dict(
        trigger_index=[0, 0, 1, 2],
        ifo=["H1"] * 4,
        scale=[512.0] * 4,
        fs=[2048.0] * 4,
        t_lo=[0.00, 0.03, 0.06, 10.0],
        t_hi=[0.03, 0.06, 0.09, 10.03],
        f_lo=[64.0, 64.0, 64.0, 128.0],
        f_hi=[128.0, 128.0, 128.0, 256.0],
        energy=[9.0, 16.0, 4.0, 1.0],
        sigma=[1.0, 1.0, 2.0, 1.0]))
    return frame.assign(**overrides)


def events_of(pixels, tolerance=0.0):
    graph = build_pixel_graph(pixels,
                             config=PixelGraphConfig(time_tolerance=tolerance))
    return cluster_events(graph).sort_values("gpsStart").reset_index(drop=True)


def test_the_statistic_is_the_norm_over_the_tiles_on_their_own_noise_scale():
    # Two windows of different noise: 9 and 16 at sigma 1, then 4 at sigma 2.
    events = events_of(tiles())
    assert events.EnWDF[0] == pytest.approx(np.sqrt(9.0 + 16.0 + 1.0))


def test_the_loudest_single_window_is_reported_beside_it():
    # Trigger 0 holds 9 and 16 on sigma 1; trigger 1 holds 4 on sigma 2.
    events = events_of(tiles())
    assert events.EnWDF_window[0] == pytest.approx(5.0)
    assert events.n_triggers[0] == 2


def test_a_tile_with_no_noise_scale_is_left_out_rather_than_counted_as_one():
    unusable = tiles()
    unusable.loc[2, "sigma"] = np.nan
    events = events_of(unusable)
    assert events.EnWDF[0] == pytest.approx(np.sqrt(9.0 + 16.0))


def test_the_frequency_is_the_geometric_moment_of_the_tiles():
    events = events_of(tiles())
    assert events.freqMean[0] == pytest.approx(np.sqrt(64.0 * 128.0))
    assert events.freqMin[0] == 64.0
    assert events.freqMax[0] == 128.0


def test_the_peak_is_the_loudest_tile_and_not_the_first():
    events = events_of(tiles())
    # The 16 sits in the second tile, from 0.03 to 0.06.
    assert events.gpsPeak[0] == pytest.approx(0.045)
    assert events.snrPeak[0] == pytest.approx(4.0)


def test_the_event_spans_its_tiles_and_stops_there():
    events = events_of(tiles())
    assert events.gpsStart[0] == pytest.approx(0.0)
    assert events.duration[0] == pytest.approx(0.09)
    assert len(events) == 2
    assert events.n_pixels.tolist() == [3, 1]


def test_a_tile_two_windows_both_reported_is_counted_once():
    twice = tiles()
    duplicate = twice.iloc[[1]].assign(trigger_index=1, energy=15.0)
    both = pd.concat([twice, duplicate], ignore_index=True)
    assert len(unique_tiles(both)) == len(twice)
    # The larger of the two estimates survives.
    kept = unique_tiles(both)
    assert kept.loc[kept.t_lo == 0.03, "energy"].iloc[0] == 16.0


def test_the_components_of_a_pixel_graph_are_its_connected_sets():
    graph = PixelGraph(pd.DataFrame(dict(a=range(4))),
                       np.array([[0, 1], [2, 3]]), np.zeros((2, 6)))
    assert graph.components().tolist() == [0, 0, 1, 1]


def test_a_size_below_everything_the_background_measured_scores_nothing():
    background = pd.DataFrame(dict(
        EnWDF=np.random.default_rng(0).normal(5.0, 1.0, 3000),
        n_pixels=np.repeat([2, 3, 4], 1000)))
    calibration = EventCalibration.fit(background, statistic="EnWDF")
    asked = pd.DataFrame(dict(EnWDF=[8.0, 8.0], n_pixels=[1, 3]))
    scored = calibration.significance(asked)
    assert np.isnan(scored[0])
    assert np.isfinite(scored[1])


def test_a_size_beyond_the_background_is_pooled_and_says_so():
    background = pd.DataFrame(dict(
        EnWDF=np.random.default_rng(1).normal(5.0, 1.0, 3000),
        n_pixels=np.repeat([1, 2, 3], 1000)))
    calibration = EventCalibration.fit(background, statistic="EnWDF")
    asked = pd.DataFrame(dict(EnWDF=[8.0, 8.0], n_pixels=[3, 50]))
    with pytest.warns(RuntimeWarning, match="larger than any"):
        scored = calibration.significance(asked)
    assert scored[0] == pytest.approx(scored[1])


def test_a_size_that_is_not_a_whole_number_is_refused():
    background = pd.DataFrame(dict(EnWDF=[1.0, 2.0], n_pixels=[1.0, np.nan]))
    with pytest.raises(ValueError, match="no finite"):
        EventCalibration.fit(background, statistic="EnWDF")


def test_the_background_is_scored_by_a_calibration_without_it():
    """Self-scoring caps a background event at the count its own bin can
    express; scored out of fold it reaches the tail candidates are read on."""
    rng = np.random.default_rng(3)
    background = pd.DataFrame(dict(
        EnWDF=rng.chisquare(8, 20000) ** 0.5,
        n_pixels=rng.integers(1, 8, 20000)))
    own = EventCalibration.fit(background, statistic="EnWDF").significance(background)
    out = out_of_sample_significance(background, folds=10, statistic="EnWDF")
    assert np.nanmax(out) > np.nanmax(own)
    # Both remain unit-rate on average, which is what the mapping promises.
    assert np.nanmean(out) == pytest.approx(1.0, abs=0.05)


def test_rescoring_keeps_the_per_window_value_it_is_judged_against():
    events = pd.DataFrame(dict(cluster_id=[0], EnWDF=[10.0], EnWDF_window=[4.0]))
    out = score_events_by_reconstruction(events, {})
    assert out.EnWDF_window[0] == pytest.approx(4.0)


def test_the_event_knows_which_triggers_it_was_assembled_from():
    events = events_of(tiles())
    assert [sorted(m) for m in events.member_indices] == [[0, 1], [2]]


def test_the_ridge_is_measured_on_the_event_own_tiles():
    from wdf.analysis.ridge import RIDGE_FEATURES

    events = events_of(tiles())
    assert set(RIDGE_FEATURES) <= set(events.columns)
    # One tile is no track, so the single-tile event has none.
    assert np.isnan(events.ridge_occupancy[1])
    assert np.isfinite(events.ridge_occupancy[0])


def test_a_tile_two_windows_reported_is_not_counted_twice_by_the_graph():
    from wdf.analysis.pixel_graph import build_pixel_graph, cluster_events

    twice = pd.DataFrame(dict(
        trigger_index=[0, 1, 1], ifo=["H1"] * 3, scale=[512.0] * 3,
        fs=[2048.0] * 3, t_lo=[0.0, 0.0, 0.03], t_hi=[0.03, 0.03, 0.06],
        f_lo=[64.0] * 3, f_hi=[128.0] * 3, energy=[9.0, 9.0, 16.0],
        sigma=[1.0] * 3))
    graph = build_pixel_graph(twice, config=PixelGraphConfig(time_tolerance=0.0))
    events = cluster_events(graph)
    assert events.n_pixels[0] == 2
    assert events.EnWDF[0] == pytest.approx(5.0)


def test_the_map_is_drawn_from_the_cluster_tiles_on_their_own_scale():
    from wdf.analysis.pixel_graph import build_pixel_graph, cluster_wavegrams

    graph = build_pixel_graph(tiles(), config=PixelGraphConfig(time_tolerance=0.0))
    maps = cluster_wavegrams(graph, time_bins=16)
    # 3 and 4 on sigma 1, then 2 on sigma 2: the map carries |c| / sigma.
    assert maps[0].grid.sum() == pytest.approx(3.0 + 4.0 + 1.0)
    assert maps[1].grid.sum() == pytest.approx(1.0)


def test_two_detectors_do_not_share_a_tile():
    both = pd.DataFrame(dict(
        ifo=["H1", "L1"], scale=[512.0] * 2, t_lo=[0.0, 0.0], t_hi=[0.03, 0.03],
        f_lo=[64.0] * 2, f_hi=[128.0] * 2, energy=[9.0, 4.0], sigma=[1.0] * 2))
    assert len(unique_tiles(both)) == 2


def test_two_events_do_not_share_a_tile():
    labelled = pd.DataFrame(dict(
        ifo=["H1"] * 2, cluster_id=[0, 1], scale=[512.0] * 2,
        t_lo=[0.0, 0.0], t_hi=[0.03, 0.03], f_lo=[64.0] * 2, f_hi=[128.0] * 2,
        energy=[9.0, 4.0], sigma=[1.0] * 2))
    assert len(unique_tiles(labelled)) == 2


def test_a_curve_does_not_read_a_threshold_off_a_missing_value():
    from wdf.analysis.modes import DetectionMode, mode_roc

    mode = DetectionMode(
        name="x", statistic="s",
        foreground=pd.DataFrame(dict(s=[9.0, 8.0])),
        background=pd.DataFrame(dict(s=[1.0, np.nan, 3.0, np.nan, 2.0, 5.0])),
        n_injections=2, livetime_days=1.0)
    curve = mode_roc(mode)
    assert np.isfinite(curve.threshold).all()
    assert curve.threshold.iloc[0] == pytest.approx(5.0)


def test_an_empty_background_is_not_a_quiet_one():
    from wdf.analysis.evaluation import threshold_at_far

    assert np.isnan(threshold_at_far([], livetime_days=1.0, far_per_day=1.0))
