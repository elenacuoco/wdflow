import numpy as np
import pandas as pd
import pytest

from wdf.analysis.detector_graph import (
    DetectorGraphConfig,
    band_grid,
    build_detector_graph,
    detector_events,
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
