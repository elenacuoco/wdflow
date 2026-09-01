"""The event as a wavegram: a connected set of tiles, and what it must report.

The quantities here are the ones every later stage reads, so each test states
the arithmetic it expects rather than comparing against a stored number.
"""
import numpy as np
import pandas as pd
import pytest

from wdf.analysis.cluster_coefficients import score_events_by_reconstruction
from wdf.analysis.event_significance import EventCalibration
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


def test_an_extent_below_everything_the_background_measured_scores_nothing():
    background = pd.DataFrame(dict(
        EnWDF=np.random.default_rng(0).normal(5.0, 1.0, 3000),
        n_triggers=np.repeat([2, 3, 4], 1000)))
    calibration = EventCalibration.fit(background, statistic="EnWDF")
    asked = pd.DataFrame(dict(EnWDF=[8.0, 8.0], n_triggers=[1, 3]))
    scored = calibration.significance(asked)
    assert np.isnan(scored[0])
    assert np.isfinite(scored[1])


def test_an_extent_beyond_the_background_uses_the_pooled_last_bin():
    background = pd.DataFrame(dict(
        EnWDF=np.random.default_rng(1).normal(5.0, 1.0, 3000),
        n_triggers=np.repeat([1, 2, 3], 1000)))
    calibration = EventCalibration.fit(background, statistic="EnWDF")
    asked = pd.DataFrame(dict(EnWDF=[8.0, 8.0], n_triggers=[3, 50]))
    scored = calibration.significance(asked)
    assert scored[0] == pytest.approx(scored[1])


def test_rescoring_keeps_the_per_window_value_it_is_judged_against():
    events = pd.DataFrame(dict(cluster_id=[0], EnWDF=[10.0], EnWDF_window=[4.0]))
    out = score_events_by_reconstruction(events, {})
    assert out.EnWDF_window[0] == pytest.approx(4.0)
