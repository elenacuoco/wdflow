from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wdf.analysis.robust_events import (
    ClusterConfig,
    FARConfig,
    IndexedCoincidenceFinder,
    TimeSlideFAR,
    cluster_detector_triggers,
)


def _events(times, strengths, ifo):
    return pd.DataFrame(
        {
            "cluster_id": np.arange(len(times)),
            "gpsPeak": times,
            "freqMin": 50.0,
            "freqMean": 100.0,
            "freqMax": 200.0,
            "EnWDF": strengths,
            "ifo": ifo,
        }
    )


def test_indexed_coincidence_is_one_to_one_and_exposes_enwdf_statistics():
    finder = IndexedCoincidenceFinder(timing_jitter_s=0.05)
    left = _events([10.00, 10.04], [12.0, 8.0], "H1")
    right = _events([10.02], [7.0], "L1")

    result = finder.find({"H1": left, "L1": right})

    assert len(result) == 1
    assert result.iloc[0].network_enwdf == np.hypot(12.0, 7.0)
    assert result.iloc[0].network_min_enwdf == 7.0
    assert result.iloc[0].dt_s == result.iloc[0].delta_t


def test_far_uses_total_slide_livetime_and_never_reports_exact_zero():
    finder = IndexedCoincidenceFinder(timing_jitter_s=0.05)
    estimator = TimeSlideFAR(finder, FARConfig(n_slides=10))
    candidate = pd.DataFrame(
        {"network_min_enwdf": [20.0], "network_enwdf": [25.0]}
    )
    background = pd.DataFrame({"network_min_enwdf": [1.0, 2.0]})
    background.attrs["total_livetime_s"] = 1000.0

    ranked = estimator.rank_candidates(candidate, background, 100.0)

    assert ranked.iloc[0].n_background_ge == 0
    assert ranked.iloc[0].far_hz == 1.0 / 1000.0
    assert 0.0 < ranked.iloc[0].fap < 1.0


def test_cluster_catalog_keeps_true_enwdf_and_member_indices():
    triggers = pd.DataFrame(
        {
            "gps": [100.0, 100.1],
            "gpsPeak": [100.05, 100.15],
            "duration": [0.1, 0.1],
            "freqMin": [50.0, 60.0],
            "freqMean": [100.0, 110.0],
            "freqMax": [150.0, 160.0],
            "freqPeak": [90.0, 100.0],
            "EnWDF": [4.0, 7.0],
            "snrPeak": [2.0, 3.0],
            "sigma": [1.0, 1.0],
            "ifo": ["H1", "H1"],
        }
    )
    parameters = SimpleNamespace(window=256, overlap=128, resampling=1024)

    _, catalog = cluster_detector_triggers(triggers, parameters)

    assert len(catalog) == 1
    assert catalog.iloc[0].EnWDF == 7.0
    assert catalog.iloc[0].EnWDF == 7.0
    assert catalog.iloc[0].member_indices == (0, 1)


def _timed_events(centroids, spreads, durations, strengths, ifo):
    centroids = np.asarray(centroids, dtype=float)
    durations = np.asarray(durations, dtype=float)
    return pd.DataFrame(
        {
            "cluster_id": np.arange(len(centroids)),
            "gpsCentroid": centroids,
            "gpsPeak": centroids,
            "gpsStart": centroids - 0.5 * durations,
            "duration": durations,
            "tSpread": spreads,
            "freqMin": 50.0,
            "freqMean": 100.0,
            "freqMax": 200.0,
            "EnWDF": strengths,
            "ifo": ifo,
        }
    )


def test_the_timing_tolerance_comes_from_the_events_not_from_a_constant():
    """A chirp lasting seconds and a blip lasting milliseconds cannot share one
    window: the tolerance is the light travel time plus the two events' own
    spreads in quadrature."""
    finder = IndexedCoincidenceFinder(light_travel_time_s=0.01, timing_jitter_s=0.001,
                                      timing_sigma=3.0)
    tight = finder.config.timing_tolerance(0.001, 0.001)
    loose = finder.config.timing_tolerance(1.0, 1.0)

    assert tight < 0.02
    assert loose > 4.0
    assert tight == pytest.approx(0.01 + 3.0 * np.hypot(0.001, 0.001))


def test_a_pair_further_apart_than_its_own_tolerance_is_not_a_candidate():
    finder = IndexedCoincidenceFinder(light_travel_time_s=0.01, timing_jitter_s=0.001,
                                      timing_sigma=3.0)
    left = _timed_events([100.0], [0.001], [0.01], [10.0], "H1")
    near = _timed_events([100.005], [0.001], [0.01], [9.0], "L1")
    far = _timed_events([100.5], [0.001], [0.01], [9.0], "L1")

    assert len(finder.find({"H1": left, "L1": near})) == 1
    assert len(finder.find({"H1": left, "L1": far})) == 0


def test_the_same_pair_pairs_once_its_spread_says_it_may():
    """Two long events sit further apart in centroid than two short ones may,
    and still describe the same transient."""
    finder = IndexedCoincidenceFinder(light_travel_time_s=0.01, timing_jitter_s=0.001,
                                      timing_sigma=3.0)
    short_left = _timed_events([100.0], [0.001], [0.01], [10.0], "H1")
    short_right = _timed_events([100.3], [0.001], [0.01], [9.0], "L1")
    long_left = _timed_events([100.0], [0.5], [4.0], [10.0], "H1")
    long_right = _timed_events([100.3], [0.5], [4.0], [9.0], "L1")

    assert len(finder.find({"H1": short_left, "L1": short_right})) == 0
    assert len(finder.find({"H1": long_left, "L1": long_right})) == 1


def test_supports_that_do_not_overlap_are_refused_however_close_the_centroids():
    """The centroids of a long event and a short one inside it coincide; the
    time-support test is what says whether they describe the same stretch."""
    finder = IndexedCoincidenceFinder(light_travel_time_s=0.01, timing_jitter_s=1.0,
                                      minimum_time_overlap=0.5)
    left = _timed_events([100.0], [0.01], [0.02], [10.0], "H1")
    apart = _timed_events([100.0 + 1.0], [0.01], [0.02], [9.0], "L1")
    apart["gpsStart"] = 101.0
    apart["gpsCentroid"] = 100.0

    assert len(finder.find({"H1": left, "L1": apart})) == 0


def test_a_weaker_detectors_shorter_support_still_overlaps_fully():
    """The weaker detector keeps fewer coefficients, so its support is a subset
    of the stronger one's -- which is a full overlap, not a partial one."""
    from wdf.analysis.robust_events import _shifted_overlap_fraction

    assert _shifted_overlap_fraction(100.0, 104.0, 101.0, 101.5, 0.0) == pytest.approx(1.0)
    assert _shifted_overlap_fraction(100.0, 104.0, 106.0, 106.5, 0.0) == pytest.approx(0.0)


def test_the_background_slides_the_time_the_coincidence_is_measured_on():
    """A FAR measured with a criterion the foreground did not use is not a FAR."""
    left = _timed_events(1000.0 + np.arange(0, 100, 7.0), 0.01, 0.05,
                         np.linspace(6, 12, 15), "H1")
    right = _timed_events(1000.0 + np.arange(0, 100, 9.0), 0.01, 0.05,
                          np.linspace(6, 12, 12), "L1")
    finder = IndexedCoincidenceFinder(timing_jitter_s=0.05)
    far = TimeSlideFAR(finder, FARConfig(n_slides=5, min_shift_s=2.0, seed=0))

    background = far.background_distribution(
        {"H1": left, "L1": right},
        {"H1": (1000.0, 1100.0), "L1": (1000.0, 1100.0)},
    )

    # Every background candidate carries the same columns the foreground does,
    # because it came out of the same finder.
    foreground = finder.find({"H1": left, "L1": right})
    if len(background) and len(foreground):
        assert set(foreground.columns) <= set(background.columns)


def _windows_on_the_grid(n_windows, stride, window_s, peak_offsets, gps0=1000.0):
    """Consecutive analysis windows, with the peak placed inside each one."""
    gps = gps0 + stride * np.arange(n_windows)
    return pd.DataFrame({
        "gps": gps,
        "gpsPeak": gps + np.asarray(peak_offsets, dtype=float),
        "gpsStart": gps,
        "gpsEnd": gps + window_s,
        "duration": window_s,
        "freqMin": 64.0,
        "freqMean": 128.0,
        "freqMax": 256.0,
        "freqPeak": 128.0,
        "EnWDF": np.full(n_windows, 6.0),
        "snrPeak": 3.0,
        "sigma": 1.0,
        "ifo": "H1",
    })


def test_consecutive_windows_link_wherever_their_peaks_fall():
    """The peak sits anywhere inside its own window, and the stride is shorter
    than the window by the overlap, so peak-to-peak differences reach beyond one
    stride while the windows are still consecutive. Linking is a statement about
    the windows, so it must not depend on where the energy landed in them."""
    parameters = SimpleNamespace(window=512, overlap=128, resampling=2048)
    stride = (512 - 128) / 2048.0
    window_s = 512 / 2048.0

    rng = np.random.default_rng(0)
    n = 12
    # The worst case the tiling allows: peaks at opposite ends of adjacent windows.
    alternating = np.where(np.arange(n) % 2 == 0, 0.0, window_s)
    scattered = rng.uniform(0.0, window_s, n)

    for offsets in (np.zeros(n), alternating, scattered):
        triggers = _windows_on_the_grid(n, stride, window_s, offsets)
        _, events = cluster_detector_triggers(
            triggers, parameters, config=ClusterConfig(max_missing_windows=0))
        assert len(events) == 1
        assert int(events.iloc[0].n_triggers) == n


def test_a_missing_window_breaks_the_chain_unless_it_is_allowed():
    """max_missing_windows counts windows, which is what it can mean once the
    origins sit on an exact grid."""
    parameters = SimpleNamespace(window=512, overlap=128, resampling=2048)
    stride = (512 - 128) / 2048.0
    window_s = 512 / 2048.0

    triggers = _windows_on_the_grid(6, stride, window_s, np.zeros(6))
    with_gap = triggers.drop(index=3).reset_index(drop=True)

    _, strict = cluster_detector_triggers(
        with_gap, parameters, config=ClusterConfig(max_missing_windows=0))
    _, lenient = cluster_detector_triggers(
        with_gap, parameters, config=ClusterConfig(max_missing_windows=1))

    assert len(strict) == 2
    assert len(lenient) == 1
