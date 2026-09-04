"""The network stage: what an edge between two detectors carries."""
import numpy as np
import pytest

from wdf.analysis.network_graph import TriggerGraphBuilder


def test_a_different_event_set_is_prepared_again():
    """A slide moves events and leaves the set intact, so one preparation
    serves every slide of it. A background slid stretch by stretch is several
    sets, and each is its own preparation."""
    import pandas as pd

    whole = {"H1": pd.DataFrame(dict(cluster_id=[0, 1, 2])),
             "L1": pd.DataFrame(dict(cluster_id=[0, 1]))}
    stretch = {"H1": whole["H1"].iloc[:2], "L1": whole["L1"]}

    order = TriggerGraphBuilder.event_order(whole, ["H1", "L1"])
    assert order == [("H1", 0), ("H1", 1), ("H1", 2), ("L1", 0), ("L1", 1)]
    assert TriggerGraphBuilder.event_order(stretch, ["H1", "L1"]) != order
    # The detector order is part of it: the arrays are laid out that way.
    assert TriggerGraphBuilder.event_order(whole, ["L1", "H1"]) != order


def test_the_edge_carries_the_difference_of_two_node_instants():
    """An edge's arrival-time difference says whether the pair is causally
    possible and how much of its tolerance it consumed. That is a difference of
    two quantities each event owns, so a time slide carries it and nothing is
    measured per pair."""
    import pandas as pd
    from wdf.analysis.robust_events import INSTANT_COLUMNS

    # The order states the preference: the instant read on the event's own
    # reconstruction, then the tile centre, then the energy centroid.
    assert INSTANT_COLUMNS == ("gpsEnvelope", "gpsPeak", "gpsCentroid")

    events = pd.DataFrame({"gpsPeak": [1000.020, 1000.000],
                           "gpsEnvelope": [1000.013, 1000.011]})
    from wdf.analysis.robust_events import _numeric
    read = _numeric(events, INSTANT_COLUMNS)
    assert read[0] - read[1] == pytest.approx(0.002)
    # Without the refined column the tile centre answers, and the difference is
    # the one the tiling can express.
    coarse = _numeric(events.drop(columns=["gpsEnvelope"]), INSTANT_COLUMNS)
    assert coarse[0] - coarse[1] == pytest.approx(0.020)


def test_a_slide_carries_the_refined_instant():
    """A displacement that moved the tile centre and left the refined instant
    behind would put the two on different clocks, and the difference of two
    events of one slide would carry the displacement."""
    import inspect

    from wdf.analysis import robust_events

    source = inspect.getsource(robust_events.TimeSlideFAR)
    shifted = source[source.index('for column in ("gpsMax"'):]
    shifted = shifted[:shifted.index(")")]
    for column in ("gpsPeak", "gpsEnvelope", "gpsCentroid", "gpsStart"):
        assert column in shifted, f"a slide leaves {column} behind"


def test_the_graph_no_longer_measures_a_pair_on_its_waveforms():
    """The estimator stays in `wdf.analysis.timing` for the candidates that
    earn it; the graph must not reach for it once per pair."""
    import inspect

    from wdf.analysis import network_graph

    source = inspect.getsource(network_graph)
    assert "arrival_time_difference" not in source
    assert "_timed_on_reconstruction" not in source
    assert not hasattr(network_graph.TriggerGraphBuilder, "_timed_on_reconstruction")


def test_a_graph_asked_only_for_candidates_builds_no_neighbourhood():
    """The same-detector edges give a node its local context and the learned
    ranking reads them. A caller that wants the candidate table does not, and
    on a slid background they are millions of rows built and discarded once per
    shift. Asking for them or not must not change a single candidate."""
    import pandas as pd

    from wdf.analysis.network_graph import TriggerGraphBuilder

    class Rendered:
        """The smallest thing `prepare` accepts: a map and its tiles."""
        bin_seconds = 1.0
        tiles = None

        def __init__(self, seed):
            rng = np.random.default_rng(seed)
            self.grid = rng.random((4, 8))

        def wavegram(self, n_time_bins):
            return self.grid

    def events(ifo, times):
        return pd.DataFrame({
            "cluster_id": np.arange(len(times)),
            "gpsPeak": times, "gpsStart": times - 0.05,
            "duration": np.full(len(times), 0.1),
            "tSpread": np.full(len(times), 0.002),
            "freqMin": np.full(len(times), 40.0),
            "freqMax": np.full(len(times), 300.0),
            "EnWDF": np.linspace(8.0, 20.0, len(times)),
            "n_coeff": np.full(len(times), 512),
            "fs": np.full(len(times), 2048.0),
        })

    times = np.array([100.0, 100.5, 101.0, 101.5])
    clustered = {"H1": events("H1", times), "L1": events("L1", times + 0.003)}
    coefficients = {ifo: {k: Rendered(k) for k in range(len(times))}
                    for ifo in clustered}

    builder = TriggerGraphBuilder(ifos=["H1", "L1"])
    prepared = builder.prepare(clustered, coefficients)
    full = builder.build_from_prepared(clustered, prepared)
    bare = builder.build_from_prepared(clustered, prepared,
                                       with_neighbours=False)

    assert len(bare.intra_edges) == 0
    assert len(full.intra_edges) > 0
    assert np.array_equal(full.cross_edges, bare.cross_edges)
    assert np.allclose(full.cross_edge_features, bare.cross_edge_features,
                       equal_nan=True)
    pd.testing.assert_frame_equal(full.candidate_table(), bare.candidate_table())


def test_a_displaced_event_carries_its_tiles():
    """The coherent statistic is summed over tiles, and a tile carries an
    absolute time. The node-side arrays are prepared once and reused for every
    slide, so a slide that moves an event and not its tiles would compare the
    pair at the place it used to occupy: no slid pair would share a tile, the
    statistic would be zero on the whole accidental population by construction,
    and the threshold it is read at would be zero at every rate.

    Two events of identical morphology brought together by a displacement must
    therefore carry the coherent energy they carry when they coincide without
    one."""
    import pandas as pd

    from wdf.analysis.network_graph import TriggerGraphBuilder

    SPACING = 30.0
    # One tile, 8 ms long, in one band, of unit amplitude, placed on the
    # event's own instant. Two such events overlapping in time and band share
    # it; two separated by the spacing do not.
    def tiles_at(instant):
        return (np.array([instant - 0.004]), np.array([instant + 0.004]),
                np.array([64.0]), np.array([128.0]),
                np.array([9.0]), np.array([3.0]))

    class Rendered:
        bin_seconds = 1.0

        def __init__(self, instant):
            self.grid = np.ones((4, 8))
            self.tiles = tiles_at(instant)

        def wavegram(self, n_time_bins):
            return self.grid

    def events(times):
        return pd.DataFrame({
            "cluster_id": np.arange(len(times)),
            "gpsPeak": times, "gpsStart": times - 0.05,
            "gpsCentroid": times,
            "duration": np.full(len(times), 0.1),
            "tSpread": np.full(len(times), 0.002),
            "freqMin": np.full(len(times), 40.0),
            "freqMax": np.full(len(times), 300.0),
            "EnWDF": np.full(len(times), 12.0),
            "n_coeff": np.full(len(times), 512),
            "fs": np.full(len(times), 2048.0),
        })

    base = 1000.0
    h1_times = base + np.array([0.0, SPACING, 2 * SPACING])
    l1_times = base + np.array([0.0])
    clustered = {"H1": events(h1_times), "L1": events(l1_times)}
    coefficients = {"H1": {k: Rendered(t) for k, t in enumerate(h1_times)},
                    "L1": {0: Rendered(l1_times[0])}}

    builder = TriggerGraphBuilder(ifos=["H1", "L1"])
    prepared = builder.prepare(clustered, coefficients)

    def coherence(shift):
        moved = clustered["L1"].copy()
        for column in ("gpsPeak", "gpsStart", "gpsCentroid"):
            moved[column] = moved[column] + shift
        graph = builder.build_from_prepared({"H1": clustered["H1"],
                                             "L1": moved}, prepared,
                                            with_neighbours=False)
        table = graph.candidate_table()
        assert len(table) == 1, "one H1 event coincides with the displaced L1"
        return float(table.tile_coherence.iloc[0])

    unshifted = coherence(0.0)
    assert unshifted != 0.0, "coinciding events of one morphology share a tile"
    # Identical morphologies, so the displaced pair carries the same energy.
    assert coherence(SPACING) == pytest.approx(unshifted)
    assert coherence(2 * SPACING) == pytest.approx(unshifted)


def test_a_comparison_rendering_is_prepared_beside_the_assembly_one():
    """`prepare` may be given a second rendering of the same events, the one
    two detectors are compared on, at a resolution of its own. It is stacked by
    the same routine as the assembly rendering, so the two must be unpacked
    alike --- a caller that renders both takes this path, and a test suite that
    never does leaves it unread."""
    import pandas as pd

    from wdf.analysis.network_graph import TriggerGraphBuilder

    class Rendered:
        bin_seconds = 1.0
        tiles = None
        block_tiles = None

        def __init__(self, seed):
            self.grid = np.random.default_rng(seed).random((4, 8))

        def wavegram(self, n_time_bins):
            return self.grid

    def events(times):
        return pd.DataFrame({
            "cluster_id": np.arange(len(times)),
            "gpsPeak": times, "gpsStart": times - 0.05,
            "duration": np.full(len(times), 0.1),
            "tSpread": np.full(len(times), 0.002),
            "freqMin": np.full(len(times), 40.0),
            "freqMax": np.full(len(times), 300.0),
            "EnWDF": np.full(len(times), 12.0),
            "n_coeff": np.full(len(times), 512),
            "fs": np.full(len(times), 2048.0),
        })

    times = np.array([100.0, 100.5, 101.0])
    clustered = {"H1": events(times), "L1": events(times + 0.003)}
    assembly = {ifo: {k: Rendered(k) for k in range(len(times))} for ifo in clustered}
    comparison = {ifo: {k: Rendered(10 + k) for k in range(len(times))}
                  for ifo in clustered}

    builder = TriggerGraphBuilder(ifos=["H1", "L1"])
    prepared = builder.prepare(clustered, assembly, comparison=comparison)
    graph = builder.build_from_prepared(clustered, prepared, with_neighbours=False)
    assert len(graph.cross_edges) > 0


def test_a_finder_refuses_an_event_set_it_was_not_prepared_for():
    """A preparation records the instant each event held when its tiles were
    laid out, and every later call places the tiles by the difference from it.
    Preparing again inside a slide loop would take that instant from frames the
    loop had already displaced, so the tiles would move by the difference
    between two slides instead of by the slide and no displaced pair would
    share one --- a failure that reads as a background of unnatural
    cleanliness rather than as an error. A caller that changes the event set
    prepares for it itself."""
    import pandas as pd
    import pytest

    from wdf.analysis.network_graph import (TriggerGraphBuilder,
                                            WavegramCoincidenceFinder)

    class Rendered:
        bin_seconds = 1.0
        tiles = None
        block_tiles = None

        def __init__(self, seed):
            self.grid = np.random.default_rng(seed).random((4, 8))

        def wavegram(self, n_time_bins):
            return self.grid

    def events(times):
        return pd.DataFrame({
            "cluster_id": np.arange(len(times)),
            "gpsPeak": times, "gpsStart": times - 0.05,
            "gpsCentroid": times, "gpsEnvelope": times,
            "duration": np.full(len(times), 0.1),
            "tSpread": np.full(len(times), 0.002),
            "freqMin": np.full(len(times), 40.0),
            "freqMax": np.full(len(times), 300.0),
            "EnWDF": np.full(len(times), 12.0),
            "n_coeff": np.full(len(times), 512),
            "fs": np.full(len(times), 2048.0),
        })

    times = 1000.0 + np.array([0.0, 0.5, 1.0, 1.5])
    whole = {"H1": events(times), "L1": events(times + 0.003)}
    coefficients = {ifo: {k: Rendered(k) for k in range(len(times))}
                    for ifo in whole}

    builder = TriggerGraphBuilder(ifos=["H1", "L1"])
    finder = WavegramCoincidenceFinder(
        None, builder, coefficients,
        prepared=builder.prepare(whole, coefficients))

    # The set it was prepared for is described without complaint.
    assert len(finder.find(whole))

    # A subset is a different set, and the finder says so instead of preparing
    # for it from whatever frames arrived.
    part = {ifo: frame.iloc[:2] for ifo, frame in whole.items()}
    with pytest.raises(ValueError, match="prepare a finder for the set"):
        finder.find(part)


def test_the_comparison_bin_resolves_the_delay_and_the_shortest_tile():
    """The bin the two detectors are compared on is not the node grid's.

    It is bounded by the two things the comparison has to represent: the light
    travel time, which a coarser bin cannot place a pair inside, and the
    shortest tile the ladder holds, which a coarser bin sums signed
    coefficients of one detector across before either detector meets the
    other. Neither bound is a preference, and the column width of whichever
    rendering arrived is neither of them.
    """
    import pandas as pd

    from wdf.analysis.detectors import network_light_travel_time
    from wdf.analysis.network_graph import TriggerGraphBuilder

    ladder = np.array([[64.0, 128.0], [128.0, 256.0], [256.0, 512.0]])

    class Rendered:
        # Far coarser than either bound, and deliberately so: a node grid is a
        # fixed-size feature and has no reason to resolve a millisecond.
        bin_seconds = 0.05
        block_tiles = None
        bands = ladder

        def __init__(self, gps):
            lo = np.array([gps - 0.004, gps, gps + 0.002])
            self.tiles = (lo, lo + np.array([1.0 / 128, 1.0 / 256, 1.0 / 512]),
                          ladder[:, 0], ladder[:, 1],
                          np.array([4.0, 9.0, 1.0]), np.array([2.0, -3.0, 1.0]))

        def wavegram(self, n_time_bins):
            return np.zeros((4, 8))

    def events(times):
        return pd.DataFrame({
            "cluster_id": np.arange(len(times)),
            "gpsPeak": times, "gpsStart": times - 0.05,
            "duration": np.full(len(times), 0.1),
            "tSpread": np.full(len(times), 0.002),
            "freqMin": np.full(len(times), 64.0),
            "freqMax": np.full(len(times), 512.0),
            "EnWDF": np.full(len(times), 12.0),
            "n_coeff": np.full(len(times), 512),
        })

    times = {"H1": np.array([100.0, 100.4]), "L1": np.array([100.003, 100.402])}
    clustered = {ifo: events(t) for ifo, t in times.items()}
    comparison = {ifo: {int(k): Rendered(t) for k, t in enumerate(times[ifo])}
                  for ifo in times}
    builder = TriggerGraphBuilder(ifos=["H1", "L1"])
    prepared = builder.prepare(clustered, comparison, comparison=comparison)

    travel = network_light_travel_time(("H1", "L1"))
    shortest = 1.0 / ladder[:, 1].max()
    assert prepared["profile_bin"] <= travel
    assert prepared["profile_bin"] <= shortest
    assert prepared["profile_bin"] < Rendered.bin_seconds

    # And the axis it implies resolves the delay rather than collapsing on it:
    # the tolerance spans several lags, not one.
    graph = builder.build_from_prepared(clustered, prepared)
    assert len(graph.cross_edge_lags) > 2
    step = np.diff(graph.cross_edge_lags)
    assert np.allclose(step, prepared["profile_bin"])


def test_the_match_is_asked_only_of_a_pair_already_coincident_in_time():
    """A pair admitted on its extents is not thereby coincident in its instants.

    Admission is on the events' stretches of time, so two long events that
    overlap are candidates even when their instants are a second apart --- a
    transient longer than one analysis window is assembled as several events
    and the two detectors need not keep the same one. No displacement the
    tolerance permits brings those two instants together, so a search over
    those displacements would report the agreement between the tail of one and
    the head of the other and pay a trials factor for it. The pair keeps its
    edge and every statistic that needs no displacement; its wavegram match is
    no agreement.
    """
    import pandas as pd

    from wdf.analysis.network_graph import TriggerGraphBuilder

    ladder = np.array([[64.0, 128.0], [128.0, 256.0]])

    class Rendered:
        bin_seconds = 0.05
        block_tiles = None
        bands = ladder

        def __init__(self, gps):
            lo = np.array([gps - 0.002, gps])
            self.tiles = (lo, lo + np.array([1.0 / 128, 1.0 / 256]),
                          ladder[:, 0], ladder[:, 1],
                          np.array([4.0, 9.0]), np.array([2.0, -3.0]))

        def wavegram(self, n_time_bins):
            return np.zeros((4, 8))

    def events(times):
        return pd.DataFrame({
            "cluster_id": np.arange(len(times)),
            # Long enough that the two detectors' stretches overlap whatever
            # their instants are, which is what makes both pairs candidates.
            "gpsPeak": times, "gpsStart": times - 1.0,
            "duration": np.full(len(times), 2.0),
            "tSpread": np.full(len(times), 0.002),
            "freqMin": np.full(len(times), 64.0),
            "freqMax": np.full(len(times), 256.0),
            "EnWDF": np.full(len(times), 12.0),
            "n_coeff": np.full(len(times), 512),
        })

    # One pair coincident to within the light travel time, one a second apart.
    times = {"H1": np.array([100.0, 400.0]), "L1": np.array([100.003, 401.0])}
    clustered = {ifo: events(t) for ifo, t in times.items()}
    comparison = {ifo: {int(k): Rendered(t) for k, t in enumerate(times[ifo])}
                  for ifo in times}
    builder = TriggerGraphBuilder(ifos=["H1", "L1"])
    graph = builder.build(clustered, comparison, comparison=comparison)
    table = graph.candidate_table()

    close = table[np.abs(table.dt_s) < 0.05]
    far = table[np.abs(table.dt_s) > 0.5]
    assert len(close) == 1 and len(far) == 1
    assert float(close.network_wavegram_match.iloc[0]) > 0.0
    assert bool(close.network_wavegram_matched.iloc[0])
    assert float(far.network_wavegram_match.iloc[0]) == 0.0
    # And it says so: a pair never compared has no displacement to report, and
    # the first point of the lag axis is not one. Reporting it would give every
    # candidate a displacement and make the distribution of the arrival-time
    # difference a picture of the grid instead of of the sky.
    assert not bool(far.network_wavegram_matched.iloc[0])
    assert not np.isfinite(float(far.network_wavegram_match_dt.iloc[0]))
    assert np.isfinite(float(close.network_wavegram_match_dt.iloc[0]))
    # The distant pair keeps its edge and the statistics that need no
    # displacement: the match is withheld, the candidate is not.
    assert float(far.network_morphology.iloc[0]) >= 0.0
    assert float(far.network_min_enwdf.iloc[0]) > 0.0


def test_a_long_event_does_not_fix_the_grid_of_a_short_pair():
    """The grid a pair is compared on comes from that pair, not from the run.

    What events a run holds is not known in advance: a binary neutron star can
    last minutes and a black-hole merger a fraction of a second. Rendering
    every event on the widest one would make the long transient fix the grid of
    every short pair --- and at a bin of the shortest tile, that array does not
    exist. The measurement of a pair must therefore not move when an unrelated
    long event joins the catalogue.
    """
    import pandas as pd

    from wdf.analysis.network_graph import TriggerGraphBuilder

    ladder = np.array([[64.0, 128.0], [128.0, 256.0]])

    class Rendered:
        bin_seconds = 0.05
        block_tiles = None
        bands = ladder

        def __init__(self, gps, seconds):
            steps = np.arange(0.0, seconds, 1.0 / 256)
            lo = np.repeat(gps + steps, 2)
            self.tiles = (lo, lo + np.tile([1.0 / 128, 1.0 / 256], len(steps)),
                          np.tile(ladder[:, 0], len(steps)),
                          np.tile(ladder[:, 1], len(steps)),
                          np.full(2 * len(steps), 4.0),
                          np.tile([2.0, -3.0], len(steps)))

        def wavegram(self, n_time_bins):
            return np.zeros((4, 8))

    def events(times, seconds):
        return pd.DataFrame({
            "cluster_id": np.arange(len(times)),
            "gpsPeak": times, "gpsStart": times,
            "duration": np.asarray(seconds, dtype=float),
            "tSpread": np.full(len(times), 0.002),
            "freqMin": np.full(len(times), 64.0),
            "freqMax": np.full(len(times), 256.0),
            "EnWDF": np.full(len(times), 12.0),
            "n_coeff": np.full(len(times), 512),
        })

    def match_of_the_short_pair(times, seconds):
        clustered = {ifo: events(np.asarray(times[ifo]), seconds[ifo])
                     for ifo in times}
        comparison = {
            ifo: {int(k): Rendered(t, s)
                  for k, (t, s) in enumerate(zip(times[ifo], seconds[ifo]))}
            for ifo in times}
        table = TriggerGraphBuilder(ifos=["H1", "L1"]).build(
            clustered, comparison, comparison=comparison).candidate_table()
        close = table[np.abs(table.dt_s) < 0.05]
        assert len(close) == 1
        return float(close.network_wavegram_match.iloc[0])

    alone = match_of_the_short_pair(
        {"H1": [100.0], "L1": [100.003]}, {"H1": [0.05], "L1": [0.05]})
    # The same pair, with a transient a thousand times longer elsewhere in the
    # run. Two hundred seconds is a binary neutron star, not a pathology.
    together = match_of_the_short_pair(
        {"H1": [100.0, 900.0], "L1": [100.003]},
        {"H1": [0.05, 200.0], "L1": [0.05]})
    assert alone > 0.0
    assert together == alone


def test_a_builder_asked_not_to_match_says_so_rather_than_reporting_zero():
    """The comparison of two renderings is a baseline and enters no ranking.

    It costs a correlation per candidate, so a study that does not read it need
    not pay for it. What must not happen is a zero standing in for a
    measurement: a candidate whose renderings were never compared has to be
    distinguishable from one compared and found to agree in nothing.
    """
    import pandas as pd

    from wdf.analysis.network_graph import TriggerGraphBuilder

    ladder = np.array([[64.0, 128.0], [128.0, 256.0]])

    class Rendered:
        bin_seconds = 0.05
        block_tiles = None
        bands = ladder

        def __init__(self, gps):
            lo = np.array([gps - 0.002, gps])
            self.tiles = (lo, lo + np.array([1.0 / 128, 1.0 / 256]),
                          ladder[:, 0], ladder[:, 1],
                          np.array([4.0, 9.0]), np.array([2.0, -3.0]))

        def wavegram(self, n_time_bins):
            return np.zeros((4, 8))

    times = {"H1": np.array([100.0]), "L1": np.array([100.003])}
    clustered = {ifo: pd.DataFrame({
        "cluster_id": np.arange(len(t)), "gpsPeak": t, "gpsStart": t - 0.05,
        "duration": np.full(len(t), 0.1), "tSpread": np.full(len(t), 0.002),
        "freqMin": np.full(len(t), 64.0), "freqMax": np.full(len(t), 256.0),
        "EnWDF": np.full(len(t), 12.0), "n_coeff": np.full(len(t), 512),
    }) for ifo, t in times.items()}
    comparison = {ifo: {int(k): Rendered(t) for k, t in enumerate(times[ifo])}
                  for ifo in times}

    def table(match):
        return TriggerGraphBuilder(ifos=["H1", "L1"], match_wavegrams=match).build(
            clustered, comparison, comparison=comparison).candidate_table()

    on, off = table(True), table(False)
    assert len(on) == len(off) == 1
    assert bool(on.network_wavegram_matched.iloc[0])
    assert float(on.network_wavegram_match.iloc[0]) > 0.0
    assert not bool(off.network_wavegram_matched.iloc[0])
    assert not np.isfinite(float(off.network_wavegram_match_dt.iloc[0]))
    # Everything that does not go through the comparison is untouched: the
    # candidate set, and the statistic the coincidence is read on.
    assert float(on.network_morphology.iloc[0]) == float(off.network_morphology.iloc[0])
    assert float(on.network_enwdf.iloc[0]) == float(off.network_enwdf.iloc[0])


def test_the_wavegram_finder_scores_what_it_forms():
    """A ranking that reads the graph is applied where the graph still exists.

    The finder returns the candidate table and keeps no reference to the graph,
    so a caller cannot attach such a ranking afterwards. Without a scorer of
    its own the finder would return an unscored table whatever it was handed,
    and a background built through it would carry no column for that ranking
    while the foreground did --- which is a rate for a population that was
    never measured, arrived at in silence.
    """
    import pandas as pd

    from wdf.analysis.network_graph import (TriggerGraphBuilder,
                                            WavegramCoincidenceFinder)
    from wdf.analysis.robust_events import (CoincidenceConfig,
                                            IndexedCoincidenceFinder)

    ladder = np.array([[64.0, 128.0], [128.0, 256.0]])

    class Rendered:
        bin_seconds = 0.05
        block_tiles = None
        bands = ladder

        def __init__(self, gps):
            lo = np.array([gps - 0.002, gps])
            self.tiles = (lo, lo + np.array([1.0 / 128, 1.0 / 256]),
                          ladder[:, 0], ladder[:, 1],
                          np.array([4.0, 9.0]), np.array([2.0, -3.0]))

        def wavegram(self, n_time_bins):
            return np.zeros((4, 8))

    class Ranking:
        """Stands for the learned one: it reads the graph and names a column."""

        def score(self, graph):
            table = graph.candidate_table()
            table["ranking"] = np.arange(len(table), dtype=float)
            return table

    times = {"H1": np.array([100.0]), "L1": np.array([100.003])}
    clustered = {ifo: pd.DataFrame({
        "cluster_id": np.arange(len(t)), "gpsPeak": t, "gpsStart": t - 0.05,
        "duration": np.full(len(t), 0.1), "tSpread": np.full(len(t), 0.002),
        "freqMin": np.full(len(t), 64.0), "freqMax": np.full(len(t), 256.0),
        "EnWDF": np.full(len(t), 12.0), "n_coeff": np.full(len(t), 512),
    }) for ifo, t in times.items()}
    maps = {ifo: {int(k): Rendered(t) for k, t in enumerate(times[ifo])}
            for ifo in times}

    coincidence = CoincidenceConfig(minimum_frequency_overlap=0.0,
                                    minimum_time_overlap=0.0)
    builder = TriggerGraphBuilder(ifos=["H1", "L1"], coincidence=coincidence)

    plain = WavegramCoincidenceFinder(
        IndexedCoincidenceFinder(coincidence), builder, maps,
        comparison=maps, events=clustered).find(clustered)
    scored = WavegramCoincidenceFinder(
        IndexedCoincidenceFinder(coincidence), builder, maps,
        comparison=maps, events=clustered, scorer=Ranking()).find(clustered)

    assert len(plain) == len(scored) == 1
    assert "ranking" not in plain
    assert "ranking" in scored
    # The scorer widens the table and changes nothing else in it.
    for column in plain.columns:
        assert column in scored
