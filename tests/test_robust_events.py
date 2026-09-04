from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wdf.analysis.robust_events import (
    ClusterConfig,
    CoincidenceConfig,
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
    # The default floor is set by the extents of a real run; a synthetic
    # segment of a hundred seconds states its own, above these events.
    estimator = TimeSlideFAR(finder, FARConfig(n_slides=10,
                                              min_shift_s=2.0))
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


def test_the_tolerance_follows_the_events_up_to_what_a_signal_can_produce():
    """It is the light travel time plus the two events' own spreads in
    quadrature, and never more than a signal can produce however uncertain the
    events are about their own centroids."""
    finder = IndexedCoincidenceFinder(light_travel_time_s=0.01, timing_jitter_s=0.001,
                                      timing_sigma=3.0, maximum_tolerance_s=0.025)
    tight = finder.config.timing_tolerance(0.001, 0.001)
    loose = finder.config.timing_tolerance(1.0, 1.0)

    assert tight == pytest.approx(0.01 + 3.0 * np.hypot(0.001, 0.001))
    assert tight < loose <= 0.025
    assert loose == pytest.approx(0.025)


def test_a_pair_further_apart_than_its_own_tolerance_is_not_a_candidate():
    finder = IndexedCoincidenceFinder(light_travel_time_s=0.01, timing_jitter_s=0.001,
                                      timing_sigma=3.0)
    left = _timed_events([100.0], [0.001], [0.01], [10.0], "H1")
    near = _timed_events([100.005], [0.001], [0.01], [9.0], "L1")
    far = _timed_events([100.5], [0.001], [0.01], [9.0], "L1")

    assert len(finder.find({"H1": left, "L1": near})) == 1
    assert len(finder.find({"H1": left, "L1": far})) == 0


def test_a_long_pair_is_admitted_on_its_extent_and_ranked_on_its_instant():
    """A transient longer than one analysis window is assembled as several
    events, and the two detectors do not keep the same one: their instants
    then differ by far more than the light travel time although both belong to
    one signal. Gating on that difference would take evidence away from a
    candidate the detectors did assemble, so the test is that the stretches of
    time they cover meet --- and the difference ranks the survivors."""
    finder = IndexedCoincidenceFinder(light_travel_time_s=0.01, timing_jitter_s=0.001,
                                      timing_sigma=3.0, maximum_tolerance_s=0.025)
    # Four seconds long each and overlapping almost entirely, their instants
    # three hundred milliseconds apart: one long transient, two fragments.
    long_left = _timed_events([100.0], [0.5], [4.0], [10.0], "H1")
    long_right = _timed_events([100.3], [0.5], [4.0], [9.0], "L1")
    assert len(finder.find({"H1": long_left, "L1": long_right})) == 1

    # Extents that never meet are refused however long the events are.
    elsewhere = _timed_events([120.0], [0.5], [4.0], [9.0], "L1")
    assert len(finder.find({"H1": long_left, "L1": elsewhere})) == 0


def test_two_instants_are_still_held_to_the_light_travel_time():
    """Events of no measured extent have nothing but their instant, so the
    shift that has to reconcile them is the light travel time plus what each is
    uncertain about its own timing --- the extent test does not loosen that."""
    finder = IndexedCoincidenceFinder(light_travel_time_s=0.01, timing_jitter_s=0.001,
                                      timing_sigma=3.0)
    left = _timed_events([100.0], [0.001], [0.0], [10.0], "H1")
    near = _timed_events([100.005], [0.001], [0.0], [9.0], "L1")
    far = _timed_events([100.5], [0.001], [0.0], [9.0], "L1")

    assert len(finder.find({"H1": left, "L1": near})) == 1
    assert len(finder.find({"H1": left, "L1": far})) == 0


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



def test_a_time_slide_moves_an_event_without_tearing_it_apart():
    """An event that straddles the seam of a circular slide must arrive whole:
    wrapping each of its times on its own leaves the start after the end and
    the extent as long as the segment."""
    import numpy as np
    import pandas as pd
    from wdf.analysis.robust_events import FARConfig, TimeSlideFAR

    start, span = 1000.0, 100.0
    events = pd.DataFrame(dict(
        cluster_id=[0], ifo=["L1"], gps=[start + span - 0.5],
        gpsStart=[start + span - 0.5], gpsCentroid=[start + span - 0.4],
        gpsPeak=[start + span - 0.4], gpsEnd=[start + span - 0.2],
        tSpread=[0.01], duration=[0.3], freqMin=[20.0], freqMean=[100.0],
        freqMax=[400.0], EnWDF=[10.0], sigma=[1.0]))

    seen = []

    class Recorder:
        def find(self, events_by_ifo):
            seen.append(events_by_ifo["L1"].copy())
            return pd.DataFrame()

    # A synthetic hundred-second segment states its own floor; the default
    # is set by the extents an event of a real run reaches.
    slider = TimeSlideFAR(Recorder(),
                          config=FARConfig(n_slides=20, seed=3,
                                           min_shift_s=2.0))
    slider.background_distribution({"H1": events.assign(ifo="H1"), "L1": events},
                                   segment_bounds={"H1": (start, start + span),
                                    "L1": (start, start + span)})

    assert seen, "the slide must reach the finder"
    for frame in seen:
        extent = float(frame.gpsEnd.iloc[0] - frame.gpsStart.iloc[0])
        assert extent == pytest.approx(0.3), "the event arrived torn apart"


def test_an_empty_background_keeps_its_columns():
    """A background with no candidates still has to say what a candidate is.
    A DataFrame with neither rows nor columns reports a missing background as a
    missing statistic, which sends the reader looking for the wrong fault."""
    import numpy as np
    import pandas as pd
    from wdf.analysis.robust_events import FARConfig, TimeSlideFAR

    columns = ["candidate_id", "gps_candidate", "network_enwdf",
               "network_min_enwdf"]

    class NeverFinds:
        def find(self, events_by_ifo):
            return pd.DataFrame(columns=columns)

    events = pd.DataFrame(dict(
        cluster_id=[0], ifo=["L1"], gps=[1000.0], gpsStart=[1000.0],
        gpsCentroid=[1000.0], gpsPeak=[1000.0], tSpread=[0.01], duration=[0.1],
        freqMin=[20.0], freqMean=[100.0], freqMax=[400.0], EnWDF=[10.0],
        sigma=[1.0]))

    background = TimeSlideFAR(
        NeverFinds(),
        FARConfig(n_slides=5, seed=0,
                  min_shift_s=2.0)).background_distribution(
        {"H1": events.assign(ifo="H1"), "L1": events},
        segment_bounds={"H1": (900.0, 1100.0), "L1": (900.0, 1100.0)})

    assert len(background) == 0
    assert list(background.columns) == columns


def test_time_slides_displace_every_detector_but_the_reference():
    """With three detectors no pair may keep its real coincidences.

    The reference stays put and every other detector is displaced, so the pairs
    that include the reference are decorrelated by its own shift. The pair that
    excludes it is decorrelated only by the *difference* of two shifts, which is
    the case a two-detector implementation never has to consider.
    """
    slides = TimeSlideFAR(IndexedCoincidenceFinder(CoincidenceConfig()),
                          FARConfig(n_slides=25, seed=3, min_shift_s=5.0))
    grid = slides._shift_grid(np.random.default_rng(0), 2, span=600.0,
                              least=5.0)
    assert len(grid) >= 1
    for drawn in grid:
        assert np.abs(drawn).min() >= 5.0
        # One step apart by construction; the comparison carries the rounding
        # of a product of floats, not a shortfall.
        assert abs(drawn[0] - drawn[1]) >= 5.0 - 1e-9
    # Enumerated, so no configuration is formed twice and none reaches the
    # span, where the wrap would make it the identity.
    assert len({tuple(row) for row in grid}) == len(grid)
    assert grid.max() < 600.0


def test_time_slides_refuse_a_single_detector():
    slides = TimeSlideFAR(IndexedCoincidenceFinder(CoincidenceConfig()))
    with pytest.raises(ValueError, match="at least two"):
        slides.background_distribution({"H1": pd.DataFrame()},
                                       {"H1": (0.0, 100.0)})


def test_a_slide_shorter_than_the_admission_window_is_refused():
    """A pair is admitted when the two instants differ by no more than the
    coincidence's tolerance, so a displacement below that window leaves a real
    coincidence admissible and the signal joins the background it is measured
    against. The window is read from the coincidence, not assumed of it, and a
    floor below it is refused."""
    import pandas as pd
    import pytest

    from wdf.analysis.robust_events import (CoincidenceConfig, FARConfig,
                                            IndexedCoincidenceFinder,
                                            TimeSlideFAR)

    config = CoincidenceConfig(light_travel_time_s=0.01,
                               maximum_tolerance_s=0.025)
    finder = IndexedCoincidenceFinder(config)

    def events(ifo, gps):
        return pd.DataFrame(dict(
            cluster_id=[0], ifo=[ifo], gps=[gps], gpsStart=[gps],
            gpsCentroid=[gps], gpsPeak=[gps], tSpread=[0.001], duration=[0.1],
            freqMin=[20.0], freqMean=[100.0], freqMax=[400.0], EnWDF=[10.0],
            sigma=[1.0]))

    span = 600.0
    catalogues = {"H1": events("H1", 100.0), "L1": events("L1", 300.0)}
    bounds = {"H1": (0.0, span), "L1": (0.0, span)}

    with pytest.raises(ValueError, match="keeps real coincidences"):
        TimeSlideFAR(finder, FARConfig(n_slides=4, min_shift_s=0.005)
                     ).background_distribution(catalogues, segment_bounds=bounds)

    # Asked rather than told --- `min_shift_s=None` --- the step is the wider
    # of two measured scales: the window a pair is admitted in, and the length
    # the trigger stream is correlated over, since two lags closer than that
    # re-use the same clusters. The released analyses state a constant instead.
    drawn = TimeSlideFAR(
        finder, FARConfig(n_slides=4, min_shift_s=None)
    ).background_distribution(catalogues, segment_bounds=bounds)
    assert drawn.attrs["admission_window_s"] == pytest.approx(0.025)
    assert drawn.attrs["correlated_length_s"] == pytest.approx(0.1)
    assert drawn.attrs["min_shift_s"] == pytest.approx(0.1)


def test_the_lags_are_enumerated_and_the_livetime_is_what_the_span_held():
    """A span that cannot hold the number of lags asked for yields fewer, and
    the livetime credited is what it held. Drawing at random instead would
    return the number asked for, with the shortfall made up of configurations
    already formed --- correlated rows in a distribution a threshold reads as
    an order statistic."""
    import pandas as pd

    from wdf.analysis.robust_events import FARConfig, TimeSlideFAR

    class NeverFinds:
        def find(self, slid):
            return pd.DataFrame()

    def events(ifo):
        return pd.DataFrame(dict(
            cluster_id=[0], ifo=[ifo], gps=[500.0], gpsStart=[500.0],
            gpsCentroid=[500.0], gpsPeak=[500.0], tSpread=[0.01],
            duration=[0.1], freqMin=[20.0], freqMean=[100.0], freqMax=[400.0],
            EnWDF=[10.0], sigma=[1.0]))

    # A thousand lags of at least a hundred seconds do not fit in an hour.
    span = 3600.0
    background = TimeSlideFAR(
        NeverFinds(), FARConfig(n_slides=1000, seed=0, min_shift_s=100.0)
    ).background_distribution(
        {"H1": events("H1"), "L1": events("L1")},
        segment_bounds={"H1": (0.0, span), "L1": (0.0, span)})

    held = background.attrs["n_slides"]
    assert background.attrs["n_slides_requested"] == 1000
    assert 1 <= held <= int(span // 100.0)
    assert background.attrs["total_livetime_s"] == held * span

    # Asked instead of told, the window is the coincidence's own tolerance --
    # milliseconds -- and the same hour then holds every lag that was asked
    # for: a floor above the window buys nothing and costs configurations the
    # stretch had.
    from wdf.analysis.robust_events import (CoincidenceConfig,
                                            IndexedCoincidenceFinder)
    measured = TimeSlideFAR(
        IndexedCoincidenceFinder(CoincidenceConfig(light_travel_time_s=0.01)),
        FARConfig(n_slides=1000, seed=0,
                  min_shift_s=None)).background_distribution(
        {"H1": events("H1"), "L1": events("L1")},
        segment_bounds={"H1": (0.0, span), "L1": (0.0, span)})
    assert measured.attrs["min_shift_s"] < 1.0
    assert measured.attrs["n_slides"] > held
