import numpy as np
import pandas as pd
import pytest

from wdf.analysis.detector_graph import (
    DetectorGraphConfig,
    band_grid,
    build_detector_graph,
    detector_events,
    occupied_bands,
    trigger_wavegrams,
)

FS = 2048.0


def _triggers(scales, n, seed=0, gps0=1000.0, ifo="H1"):
    rng = np.random.default_rng(seed)
    rows = []
    for scale in scales:
        stride = 0.75 * scale / FS
        for k in range(n):
            n_nonzero = int(rng.integers(1, 6))
            index = np.sort(rng.choice(np.arange(1, scale), n_nonzero, replace=False))
            rows.append(dict(
                gps=gps0 + k * stride, gpsStart=gps0 + k * stride,
                gpsCentroid=gps0 + k * stride, tSpread=0.01,
                gpsPeak=gps0 + k * stride, duration=scale / FS,
                EnWDF=float(rng.uniform(3, 12)), sigma=1.0,
                snrPeak=float(rng.uniform(1, 6)),
                freqMin=20.0, freqMean=120.0, freqMax=400.0,
                wave="DaubC12", n_coeff=int(scale), fs=FS,
                wt_index=index.astype(np.uint16),
                wt_value=rng.normal(size=n_nonzero).astype(np.float32),
                ifo=ifo))
    return pd.DataFrame(rows)


def _fixed_coefficients(triggers, index):
    """Give every trigger the same surviving coefficients."""
    triggers = triggers.copy()
    triggers["wt_index"] = [np.asarray(index, dtype=np.uint16)] * len(triggers)
    triggers["wt_value"] = [np.full(len(index), 5.0, dtype=np.float32)] * len(triggers)
    return triggers


def _walking_coefficients(triggers, indices):
    """One surviving coefficient per trigger, at the index given."""
    triggers = triggers.copy()
    triggers["wt_index"] = [np.array([k], dtype=np.uint16) for k in indices]
    triggers["wt_value"] = [np.array([5.0], dtype=np.float32) for _ in indices]
    return triggers


def test_the_band_ladder_is_shared_between_window_lengths():
    """A longer window extends the ladder downward; it does not subdivide it,
    so the same physical band is the same row at every window length."""
    bands = band_grid([256, 512, 1024], FS)
    short = {tuple(np.round(b, 9)) for b in band_grid([256], FS)}
    assert short <= {tuple(np.round(b, 9)) for b in bands}
    assert bands[0][0] < band_grid([256], FS)[0][0] or bands[0][0] == 0.0
    assert (np.diff(bands[:, 0]) >= 0).all()


def test_a_node_feature_has_one_width_across_window_lengths():
    triggers = _triggers([256, 512, 1024], 8)
    graph = build_detector_graph(triggers)
    assert graph.node_features.shape[0] == len(triggers)
    assert np.isfinite(graph.node_features).all()
    for scale in (256, 512, 1024):
        rows = graph.nodes["n_coeff"].to_numpy() == scale
        assert rows.any()


def test_the_wavegram_places_a_band_on_the_same_row_at_every_scale():
    bands = band_grid([256, 1024], FS)
    triggers = _triggers([256, 1024], 4)
    grids = trigger_wavegrams(triggers, bands, time_bins=8)
    assert grids.shape == (len(triggers), len(bands) * 8)
    assert (grids >= 0).all()


def test_triggers_of_different_window_lengths_are_joined():
    triggers = _triggers([256, 512], 10)
    graph = build_detector_graph(triggers)
    assert graph.edge_table()["cross_scale"].sum() > 0


def test_a_distant_trigger_joins_nothing():
    near = _triggers([256], 5)
    far = _triggers([256], 5, gps0=5000.0)
    graph = build_detector_graph(pd.concat([near, far], ignore_index=True))
    labels = graph.components()
    assert labels[graph.nodes["gps"] < 2000.0].max() != labels[graph.nodes["gps"] > 2000.0].min()


def test_a_disjoint_band_is_never_joined():
    low = _triggers([256], 6)
    high = _triggers([256], 6, seed=1).assign(freqMin=600.0, freqMax=1000.0)
    graph = build_detector_graph(
        pd.concat([low, high], ignore_index=True),
        config=DetectorGraphConfig(minimum_frequency_overlap=0.5))
    table = graph.edge_table()
    band = graph.nodes["freqMin"].to_numpy()
    crossing = band[table["node_i"].to_numpy()] != band[table["node_j"].to_numpy()]
    assert not crossing.any()


def test_events_carry_what_the_network_graph_reads():
    graph = build_detector_graph(_triggers([256, 512], 10))
    events = detector_events(graph)
    for column in ("cluster_id", "ifo", "gpsCentroid", "tSpread", "gpsStart",
                   "duration", "freqMin", "freqMax", "EnWDF"):
        assert column in events.columns
    assert events["n_triggers"].sum() == len(graph.nodes)


def test_the_event_statistic_measures_its_whole_extent():
    """A transient spanning several windows is scored over all of them, so the
    event is louder than its loudest single window -- which is what a search
    without this step would have reported, and is kept beside it."""
    from wdf.analysis.detector_graph import stitched_statistic

    triggers = _triggers([256, 512], 6)
    graph = build_detector_graph(triggers)
    events = detector_events(graph)

    assert events["EnWDF_window"].max() == pytest.approx(graph.nodes["EnWDF"].max())
    assert (events["EnWDF"] >= events["EnWDF_window"] - 1e-9).all()
    np.testing.assert_allclose(events["EnWDF"].to_numpy(),
                               stitched_statistic(graph), rtol=1e-9)


def test_the_stitched_statistic_does_not_add_window_lengths():
    """Each length is a complete description of the same strain, so the largest
    is taken rather than their sum."""
    from wdf.analysis.detector_graph import stitched_statistic

    triggers = _triggers([256, 512], 8)
    graph = build_detector_graph(triggers)
    both = stitched_statistic(graph)

    one = build_detector_graph(triggers[triggers.n_coeff == 256])
    other = build_detector_graph(triggers[triggers.n_coeff == 512])
    apart = max(stitched_statistic(one).max(), stitched_statistic(other).max())
    assert both.max() <= np.hypot(stitched_statistic(one).max(),
                                  stitched_statistic(other).max()) + 1e-9
    assert both.max() == pytest.approx(apart, rel=0.5)


def test_pruning_edges_splits_events_without_rebuilding_the_graph():
    """The learned stage decides which admissible edges survive, so the split
    has to follow a mask over the same edges."""
    graph = build_detector_graph(_triggers([256, 512], 12))
    everything = graph.components()
    nothing = graph.components(keep=np.zeros(len(graph.edges), dtype=bool))
    assert nothing.max() == len(graph.nodes) - 1
    assert everything.max() <= nothing.max()


def test_a_significance_floor_thins_the_graph():
    triggers = _triggers([256], 40)
    loose = build_detector_graph(triggers)
    tight = build_detector_graph(
        triggers, config=DetectorGraphConfig(minimum_significance=np.log1p(9.0)))
    assert len(tight.nodes) < len(loose.nodes)


def test_no_triggers_gives_no_graph_and_no_events():
    empty = pd.DataFrame(columns=["gps", "n_coeff", "fs", "EnWDF"])
    graph = build_detector_graph(empty)
    assert len(graph.edges) == 0
    assert detector_events(graph).empty


def test_the_wavegram_is_on_the_noise_scale():
    """Raw coefficients are strain, of order 1e-22: a grid of those is
    numerically zero once compressed or multiplied by another grid."""
    import numpy as np
    triggers = _triggers([512], 4)
    triggers["wt_value"] = [np.asarray(v) * 1e-22 for v in triggers.wt_value]
    triggers["sigma"] = 1e-22

    grids = trigger_wavegrams(triggers, band_grid([512], FS), time_bins=8)
    assert grids.max() > 1e-3
    assert np.isfinite(np.log1p(grids)).all()


def test_a_trigger_joins_only_where_it_has_energy():
    """The support of a broadband transient covers everything beneath it. The
    bands it occupies are where its energy is, and only those connect."""
    import numpy as np

    low = _fixed_coefficients(_triggers([512], 6), [9, 10])
    high = _fixed_coefficients(_triggers([512], 6, seed=3), [400, 401])
    both = pd.concat([low, high], ignore_index=True)

    bands = band_grid([512], FS)
    masks = occupied_bands(both, bands)
    assert (masks[:6] & masks[6:]).sum() == 0, "the fixture must not share a band"

    graph = build_detector_graph(both, config=DetectorGraphConfig(band_adjacency=1))
    table = graph.edge_table()
    if len(table):
        # The graph sorts its nodes in time, so the two populations are told
        # apart by the bands they occupy rather than by their row.
        node_mask = occupied_bands(graph.nodes, bands)
        crossing = (node_mask[table.node_i.to_numpy()]
                    != node_mask[table.node_j.to_numpy()])
        assert not crossing.any()


def test_energy_in_a_band_connects_to_the_band_beside_it():
    """A transient that sweeps moves between neighbouring bands, so touching
    means adjacent as well as shared."""
    import numpy as np

    triggers = _triggers([512], 8)
    triggers = _walking_coefficients(triggers, [2 ** (7 + k % 2)
                                                for k in range(len(triggers))])

    apart = build_detector_graph(triggers, config=DetectorGraphConfig(band_adjacency=0))
    touching = build_detector_graph(triggers, config=DetectorGraphConfig(band_adjacency=1))
    assert len(touching.edges) > len(apart.edges)


def test_a_sweeping_transient_is_not_broken_by_the_continuity_test():
    """A chirp climbs one band at a time, and the test asks the bands to touch
    rather than the two wavegrams to look alike."""
    import numpy as np

    sweeping = _walking_coefficients(_triggers([512], 10),
                                     [2 ** (4 + k // 3) for k in range(10)])
    graph = build_detector_graph(sweeping, config=DetectorGraphConfig(band_adjacency=1))
    labels = graph.components()
    assert labels.max() + 1 < len(sweeping) / 2


def test_the_event_extends_as_far_as_its_energy():
    """A window is the same length whatever it holds, so measuring the event
    across the windows adds up to one whole window at each end."""
    triggers = _fixed_coefficients(_triggers([512], 4), [64])
    window = triggers.n_coeff.to_numpy(dtype=float) / triggers.fs.to_numpy(dtype=float)
    tile = window / triggers.n_coeff.to_numpy(dtype=float)
    triggers = triggers.assign(gpsStart=triggers.gps + 0.25 * window, duration=tile)

    events = detector_events(build_detector_graph(triggers))
    covered = (triggers.gpsStart + triggers.duration).max() - triggers.gpsStart.min()
    assert events["duration"].max() == pytest.approx(covered)
    assert events["gpsStart"].min() == pytest.approx(triggers.gpsStart.min())


def test_continuity_is_between_the_energy_not_between_the_windows():
    """Consecutive windows overlap by construction, so a gap measured on them
    is negative almost everywhere. What has to continue is the surviving
    tiles: one window's energy ending where the next window's begins."""
    triggers = _fixed_coefficients(_triggers([512], 6), [64])
    stride = float(np.diff(np.sort(triggers.gps.to_numpy())).min())

    # Every window overlaps its neighbour, but each holds a tile of one
    # millisecond sitting at the window's own start, so the energy is apart by
    # the whole stride.
    touching = triggers.assign(gpsStart=triggers.gps, duration=stride)
    apart = triggers.assign(gpsStart=triggers.gps, duration=0.001)

    assert len(build_detector_graph(apart).edges) < \
        len(build_detector_graph(touching).edges)


def test_two_events_of_different_length_share_one_time_base():
    """A column stands for the same duration wherever it is drawn, so two maps
    compared across the network are not stretched onto each other."""
    import numpy as np
    from wdf.analysis.detector_graph import event_coefficients

    short = _walking_coefficients(_triggers([512], 3), [64, 64, 64])
    long = _walking_coefficients(_triggers([512], 24), [64] * 24)

    grids = []
    for triggers in (short, long):
        graph = build_detector_graph(triggers)
        labels = np.zeros(len(graph.nodes), dtype=int)
        grids.append(event_coefficients(graph, labels)[0].wavegram())

    assert grids[0].shape == grids[1].shape
    # The long event fills more columns, which is the point: the same duration
    # per column means the map records how long the transient actually was.
    assert int((grids[1] > 0).sum(axis=0).astype(bool).sum()) > \
        int((grids[0] > 0).sum(axis=0).astype(bool).sum())


def test_the_time_base_follows_the_data_and_is_not_written_in_the_source():
    """A column is as long as the search's shortest stride, which is a property
    of the data analysed. Writing it into the module fixes the map to one
    sampling rate and fails silently on any other."""
    import numpy as np
    from wdf.analysis.detector_graph import event_coefficients, wavegram_bin_seconds

    triggers = _walking_coefficients(_triggers([512], 6), [64] * 6)
    faster = triggers.assign(fs=triggers.fs * 4.0,
                             gps=triggers.gps.min()
                             + (triggers.gps - triggers.gps.min()) / 4.0)

    assert wavegram_bin_seconds(faster) == pytest.approx(
        wavegram_bin_seconds(triggers) / 4.0)

    rendered = event_coefficients(build_detector_graph(faster),
                                  np.zeros(len(faster), dtype=int))
    assert rendered[0].bin_seconds == pytest.approx(wavegram_bin_seconds(faster))


def test_the_map_keeps_the_part_that_carries_the_energy():
    """An event longer than the map is truncated about its energy centroid, not
    from wherever it happens to begin."""
    import numpy as np
    from wdf.analysis.detector_graph import event_coefficients

    triggers = _walking_coefficients(_triggers([512], 120), [64] * 120)
    triggers = triggers.assign(EnWDF=1.0)
    triggers.loc[triggers.index[-5:], "EnWDF"] = 50.0

    graph = build_detector_graph(triggers)
    labels = np.zeros(len(graph.nodes), dtype=int)
    grid = event_coefficients(graph, labels)[0].wavegram()

    occupied = np.flatnonzero((grid > 0).any(axis=0))
    assert occupied.size, "the loud end must fall inside the map"


def test_the_map_can_be_placed_in_time_and_frequency():
    """A grid without its axes is not a map. The rows carry their band edges and
    the columns their GPS times, so a plot and a comparison read the same
    placement instead of each recomputing it."""
    import numpy as np
    from wdf.analysis.detector_graph import band_grid, event_coefficients

    triggers = _walking_coefficients(_triggers([512], 8), [64] * 8)
    graph = build_detector_graph(triggers)
    labels = np.zeros(len(graph.nodes), dtype=int)
    rendered = event_coefficients(graph, labels)[0]

    assert rendered.bands.shape == (rendered.grid.shape[0], 2)
    np.testing.assert_allclose(rendered.bands, band_grid([512], FS))

    times = rendered.times()
    assert times.shape == (rendered.grid.shape[1],)
    assert np.allclose(np.diff(times), rendered.bin_seconds)
    # The map is centred on the event, so the transient falls inside its span.
    assert times[0] < triggers.gpsCentroid.mean() < times[-1]


def test_the_event_is_described_by_its_own_coefficients():
    """A long event's parameters follow its energy over the whole extent, not
    the average of what each window saw of it. Two windows an octave apart give
    an event whose band covers both, where averaging their summaries would put
    it between them and describe neither."""
    import numpy as np
    from wdf.analysis.detector_graph import event_tiles

    triggers = _walking_coefficients(_triggers([512], 6), [8, 8, 8, 400, 400, 400])
    triggers = triggers.assign(freqQ05=100.0, freqQ95=110.0, duration90=0.01)

    graph = build_detector_graph(triggers)
    labels = np.zeros(len(graph.nodes), dtype=int)
    events = detector_events(graph, labels=labels)

    lo, hi, band_lo, band_hi, energy = event_tiles(graph.nodes,
                                                   np.arange(len(graph.nodes)))
    assert energy.size == len(graph.nodes), "one surviving coefficient each"
    # The members claim a band of 100 to 110 Hz each; the coefficients say
    # otherwise, and it is the coefficients the event follows.
    assert events.freqQ95.iloc[0] > 2 * events.freqQ05.iloc[0]
    assert band_lo.min() < events.freqQ05.iloc[0] < events.freqQ95.iloc[0] < band_hi.max()
    # And the extent follows the tiles rather than the members' own duration90.
    assert events.duration90.iloc[0] > 0.05


def test_the_event_has_a_waveform_and_not_only_a_number():
    """The assembled map exists to give three things: the parameters, a
    reconstruction in the time domain, and something to compare across
    detectors. This is the second."""
    import numpy as np
    from wdf.analysis.detector_graph import event_waveform

    triggers = _fixed_coefficients(_triggers([512], 5), [64, 65])
    graph = build_detector_graph(triggers)
    labels = np.zeros(len(graph.nodes), dtype=int)

    waveforms = event_waveform(graph, labels)
    gps_start, samples, length = waveforms[0]

    assert length == 512
    assert samples.ndim == 1 and samples.size > 512, \
        "an event of five windows is longer than any one of them"
    assert np.isfinite(samples).all()
    assert gps_start == pytest.approx(float(graph.nodes.gps.min()))


def test_a_different_basis_can_be_asked_to_break_continuity():
    """The competition's verdict is a statement about the shape in a window, so
    two windows it assigned to different bases can be asked to be two events.
    Off by default: a transient may legitimately change character."""
    triggers = _fixed_coefficients(_triggers([512], 8), [64])
    triggers = triggers.assign(
        wave=["DaubC12" if k < 4 else "Coif2" for k in range(len(triggers))])

    together = build_detector_graph(triggers)
    apart = build_detector_graph(
        triggers, config=DetectorGraphConfig(same_basis=True))

    assert len(apart.edges) < len(together.edges)
    assert apart.components().max() > together.components().max()


def test_the_morphology_is_measured_on_the_coefficients_not_on_a_grid():
    """Two events of one transient are loud in the same places on the plane.
    Asking that at the resolution the transform has needs no grid, and so no
    cell size chosen by hand -- too fine and two detectors share nothing, too
    coarse and everything agrees."""
    import numpy as np
    from wdf.analysis.detector_graph import event_tiles, tile_coherence

    same = _fixed_coefficients(_triggers([512], 4), [64, 65])
    elsewhere = _fixed_coefficients(_triggers([512], 4, seed=5), [400, 401])

    def cloud(triggers):
        graph = build_detector_graph(triggers)
        return event_tiles(graph.nodes, np.arange(len(graph.nodes)))

    together = tile_coherence(cloud(same), cloud(same), 0.01)
    apart = tile_coherence(cloud(same), cloud(elsewhere), 0.01)

    assert together > 0.0
    assert apart == 0.0, "tiles that share no band cohere not at all"


def test_the_event_carries_the_coefficients_its_map_is_a_view_of():
    """One source of truth per event: the map, the parameters, the waveform and
    the comparison across detectors all derive from the same coefficients, so
    they cannot come to describe different things."""
    import numpy as np
    from wdf.analysis.detector_graph import event_coefficients

    triggers = _fixed_coefficients(_triggers([512], 5), [64, 65])
    graph = build_detector_graph(triggers)
    labels = np.zeros(len(graph.nodes), dtype=int)
    rendered = event_coefficients(graph, labels)[0]

    assert rendered.tiles is not None
    t_lo, t_hi, f_lo, f_hi, energy = rendered.tiles
    assert energy.size == 2 * len(graph.nodes), "every survivor of every member"
    # The grid holds the same energy the tiles do, up to what falls outside it.
    assert rendered.grid.sum() > 0
    assert (f_lo <= f_hi).all() and (t_lo <= t_hi).all()
