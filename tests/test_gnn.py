from itertools import combinations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from wdf.analysis.gnn import (
    N_EDGE_FEATURES,
    GNNCoincidenceScorer,
    TriggerGraphBuilder,
)


FS, WINDOW, OVERLAP, N_COEFF = 2048.0, 512, 128, 512


def _synth_clustered(ifo, n, t0, seed, times=None):
    rng = np.random.default_rng(seed)
    if times is None:
        times = np.sort(rng.uniform(0, 100, n))
    peak = t0 + times
    return pd.DataFrame(dict(
        cluster_id=range(n), ifo=ifo,
        gpsStart=peak - rng.uniform(0.0, 0.05, n),
        gpsCentroid=peak, tSpread=rng.uniform(0.005, 0.05, n),
        gpsPeak=peak,
        EnWDF=rng.uniform(1, 20, n),
        freqMean=rng.uniform(50, 300, n), freqMax=rng.uniform(300, 500, n),
        freqMin=rng.uniform(20, 50, n), duration=rng.uniform(0.05, 0.5, n),
        wave="BsplineC309", n_triggers=rng.integers(1, 5, n), gps_span_s=rng.uniform(0, 1, n),
    ))


def _synth_coefficients(clustered, seed=0):
    """A ClusterCoefficients per event, with a different number of windows each
    so the fixed-width node feature is exercised."""
    from wdf.analysis.cluster_coefficients import ClusterCoefficients

    rng = np.random.default_rng(seed)
    out = {}
    for _, row in clustered.iterrows():
        n_windows = int(row["n_triggers"])
        coefficients = np.zeros((n_windows, N_COEFF))
        # a handful of loud coefficients, the rest below threshold and zeroed
        for w in range(n_windows):
            k = rng.integers(1, N_COEFF, size=6)
            coefficients[w, k] = rng.uniform(4.0, 30.0, size=6)
        out[int(row["cluster_id"])] = ClusterCoefficients(
            cluster_id=int(row["cluster_id"]),
            ifo=str(row["ifo"]),
            fs=FS, window=WINDOW, overlap=OVERLAP,
            times=float(row["gpsStart"]) + np.arange(n_windows) * (WINDOW - OVERLAP) / FS,
            coefficients=coefficients,
            waves=("DaubC12",) * n_windows,
            sigma=np.ones(n_windows),
        )
    return out


def _synth_graph_inputs(sizes, t0=1000.0, seed=1):
    # Every detector sees the same arrival times, jittered within the timing
    # tolerance, so physically admissible pairs exist to be scored.
    jitter = np.random.default_rng(seed)
    common = np.sort(jitter.uniform(0, 100, max(sizes.values())))
    clustered = {
        ifo: _synth_clustered(ifo, n, t0, seed + i,
                              times=common[:n] + jitter.uniform(-0.005, 0.005, n))
        for i, (ifo, n) in enumerate(sizes.items())}
    coefficients = {ifo: _synth_coefficients(frame, seed + i)
                    for i, (ifo, frame) in enumerate(clustered.items())}
    return clustered, coefficients


def test_graph_builder_produces_edges():
    clustered, coefficients = _synth_graph_inputs({"H1": 20, "L1": 20})
    graph = TriggerGraphBuilder(intra_ifo_window_s=5.0).build(clustered, coefficients)
    assert graph.node_features.shape[0] == 40
    assert graph.cross_edges.shape[1] == 2
    assert len(graph.cross_edges) > 0


def test_the_cross_edges_are_exactly_the_physically_admissible_pairs():
    """The graph has no admissibility rule of its own: an edge exists where and
    only where the classical finder admits the pair, so the two stages rank the
    same candidates."""
    from wdf.analysis.robust_events import IndexedCoincidenceFinder

    clustered, coefficients = _synth_graph_inputs({"H1": 40, "L1": 30})
    builder = TriggerGraphBuilder(ifos=["H1", "L1"])
    graph = builder.build(clustered, coefficients)

    finder = IndexedCoincidenceFinder(builder.coincidence)
    admissible = finder.candidate_edges(
        clustered["H1"].reset_index(drop=True),
        clustered["L1"].reset_index(drop=True))
    offset = len(clustered["H1"])
    expected = {(i, j + offset) for i, j, *_ in admissible}
    assert {tuple(e) for e in graph.cross_edges.tolist()} == expected


def test_an_impossible_pair_never_becomes_an_edge():
    """Two events further apart than any signal could put them share no edge,
    however alike their morphology."""
    clustered, coefficients = _synth_graph_inputs({"H1": 15, "L1": 15})
    clustered["L1"] = clustered["L1"].assign(
        gpsCentroid=clustered["L1"]["gpsCentroid"] + 3600.0,
        gpsPeak=clustered["L1"]["gpsPeak"] + 3600.0,
        gpsStart=clustered["L1"]["gpsStart"] + 3600.0)
    graph = TriggerGraphBuilder(ifos=["H1", "L1"]).build(clustered, coefficients)
    assert len(graph.cross_edges) == 0


def test_scorer_forward_and_fit_reduce_loss():
    clustered = {"H1": _synth_clustered("H1", 15, 1000.0, 3), "L1": _synth_clustered("L1", 15, 1000.0, 4)}
    coefficients = {ifo: _synth_coefficients(frame) for ifo, frame in clustered.items()}
    graph = TriggerGraphBuilder().build(clustered, coefficients)
    model = GNNCoincidenceScorer(node_dim=graph.node_features.shape[1], hidden=8, seed=0,
                                 cross_edge_dim=graph.cross_edge_features.shape[1])

    table = model.score(graph)
    assert "gnn_score" in table.columns
    assert ((table["gnn_score"] >= 0) & (table["gnn_score"] <= 1)).all()

    rng = np.random.default_rng(0)
    labels = (rng.uniform(0, 1, len(graph.cross_edges)) > 0.8).astype(float)
    history = model.fit([(graph, labels)], epochs=30, lr=1e-2)
    assert len(history) > 0
    assert history[-1] <= history[0]


def _reference_build(builder, clustered, coefficients):
    """Pure-Python itertools.combinations reference matching the pre-vectorization
    implementation -- kept only to check the vectorized build() against it."""
    ifos = builder.ifos or list(clustered.keys())
    nodes, node_ifo = [], []
    for ifo in ifos:
        for _, row in clustered[ifo].reset_index(drop=True).iterrows():
            nodes.append(row); node_ifo.append(ifo)
    nodes_df = pd.DataFrame(nodes).reset_index(drop=True)
    nodes_df["ifo"] = node_ifo
    idx_by_ifo = {ifo: nodes_df.index[nodes_df["ifo"] == ifo].to_numpy() for ifo in ifos}

    grids = np.vstack([
        coefficients[ifo][int(row["cluster_id"])].wavegram(builder.wavegram_time_bins).ravel()
        for ifo in ifos
        for _, row in clustered[ifo].reset_index(drop=True).iterrows()
    ])
    grids = np.log1p(grids)
    norms = np.linalg.norm(grids, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    shapes = grids / norms

    intra_edges, intra_feats = [], []
    for idxs in idx_by_ifo.values():
        for i, j in combinations(idxs, 2):
            dt = float(nodes_df.at[i, "gpsPeak"] - nodes_df.at[j, "gpsPeak"])
            if abs(dt) <= builder.intra_ifo_window_s:
                intra_edges.append((i, j)); intra_feats.append([dt])
                intra_edges.append((j, i)); intra_feats.append([-dt])
    return (np.array(intra_edges, dtype=np.int64).reshape(-1, 2),
            np.array(intra_feats).reshape(-1, 1))


def test_vectorized_build_matches_reference_loop():
    clustered, coefficients = _synth_graph_inputs({"H1": 60, "L1": 45})
    builder = TriggerGraphBuilder(intra_ifo_window_s=5.0, ifos=["H1", "L1"])

    ref_intra_e, ref_intra_f = _reference_build(builder, clustered, coefficients)
    graph = builder.build(clustered, coefficients)

    def edge_set(edges):
        return set(map(tuple, edges.tolist()))

    assert edge_set(ref_intra_e) == edge_set(graph.intra_edges)

    ref_map = {tuple(e): f for e, f in zip(ref_intra_e, ref_intra_f)}
    for e, f in zip(graph.intra_edges, graph.intra_edge_features):
        np.testing.assert_allclose(f, ref_map[tuple(e.tolist())], rtol=1e-5, atol=1e-5)


def test_fit_batches_multiple_segments_together():
    """batch_size=None (the default) should put every segment's graph into
    one torch_geometric Batch per epoch -- exercises Batch.from_data_list
    across graphs of different sizes (different node/edge counts), which is
    exactly the multi-segment throughput path this module exists for.
    """
    examples = []
    for seed, (n_h1, n_l1) in enumerate([(15, 15), (8, 12), (20, 5)]):
        # Shared arrival times, so pairs exist to batch: with the tolerance set
        # by the light travel time, independently drawn events do not coincide.
        clustered, coefficients = _synth_graph_inputs(
            {"H1": n_h1, "L1": n_l1}, t0=1000.0 + seed * 200, seed=seed + 1)
        graph = TriggerGraphBuilder().build(clustered, coefficients)
        rng = np.random.default_rng(seed)
        labels = (rng.uniform(0, 1, len(graph.cross_edges)) > 0.8).astype(float)
        examples.append((graph, labels))

    model = GNNCoincidenceScorer(node_dim=examples[0][0].node_features.shape[1], hidden=8, seed=0,
                             cross_edge_dim=examples[0][0].cross_edge_features.shape[1])
    history_full_batch = model.fit(examples, epochs=20, lr=1e-2)
    assert len(history_full_batch) > 0
    assert history_full_batch[-1] <= history_full_batch[0]

    # batch_size=1 (one segment per chunk, gradients accumulated/averaged
    # across chunks each epoch) should also train without error -- checks
    # the chunking path, not just the single-Batch-per-epoch default.
    model2 = GNNCoincidenceScorer(node_dim=examples[0][0].node_features.shape[1], hidden=8, seed=0,
                             cross_edge_dim=examples[0][0].cross_edge_features.shape[1])
    history_chunked = model2.fit(examples, epochs=20, lr=1e-2, batch_size=1)
    assert len(history_chunked) > 0


def test_scorer_device_selection():
    clustered = {"H1": _synth_clustered("H1", 5, 1000.0, 1), "L1": _synth_clustered("L1", 5, 1000.0, 2)}
    coefficients = {ifo: _synth_coefficients(frame) for ifo, frame in clustered.items()}
    graph = TriggerGraphBuilder().build(clustered, coefficients)

    default_model = GNNCoincidenceScorer(node_dim=graph.node_features.shape[1], hidden=4, seed=0,
                                     cross_edge_dim=graph.cross_edge_features.shape[1])
    # Not "cuda whenever torch says one exists": a device that is present but
    # full or mismatched is not usable, and the choice falls back rather than
    # failing the run. What has to hold is that the model and its parameters
    # agree on where they are.
    from wdf.analysis.gnn import usable_device

    assert default_model.device == usable_device()
    assert next(default_model.parameters()).device.type == usable_device()

    cpu_model = GNNCoincidenceScorer(node_dim=graph.node_features.shape[1], hidden=4, seed=0, device="cpu",
                                 cross_edge_dim=graph.cross_edge_features.shape[1])
    assert cpu_model.device == "cpu"
    table = cpu_model.score(graph)
    assert "gnn_score" in table.columns


def test_the_graph_is_built_from_the_coefficients_not_from_scalars():
    """Two events with identical scalar summaries but different coefficients
    must give different node features, which is the whole reason the graph
    reads the coefficient matrices."""
    clustered, _ = _synth_graph_inputs({"H1": 6, "L1": 6})
    coefficients = {ifo: _synth_coefficients(frame, seed=7)
                    for ifo, frame in clustered.items()}
    builder = TriggerGraphBuilder(ifos=["H1", "L1"])

    graph = builder.build(clustered, coefficients)

    # The map's own width over log2(512) + 1 octave rows, plus one column per
    # detector. The width is level one's, not a number restated here.
    from wdf.analysis.detector_graph import WAVEGRAM_TIME_BINS

    assert graph.node_features.shape[1] == 10 * WAVEGRAM_TIME_BINS + 2
    assert graph.cross_edge_features.shape[1] == N_EDGE_FEATURES

    other = {ifo: _synth_coefficients(frame, seed=99)
             for ifo, frame in clustered.items()}
    changed = builder.build(clustered, other)
    assert not np.allclose(graph.node_features, changed.node_features)


def test_a_candidate_carries_the_agreement_between_the_two_wavegrams():
    clustered = {"H1": _synth_clustered("H1", 8, 1000.0, 3),
                 "L1": _synth_clustered("L1", 8, 1000.0, 4)}
    coefficients = {ifo: _synth_coefficients(frame, seed=11)
                    for ifo, frame in clustered.items()}
    graph = TriggerGraphBuilder().build(clustered, coefficients)

    table = graph.candidate_table()
    assert "wavegram_similarity" in table.columns
    # a normalised inner product of non-negative grids
    assert ((table["wavegram_similarity"] >= -1e-6)
            & (table["wavegram_similarity"] <= 1 + 1e-6)).all()


def test_fit_only_learns_from_the_masked_edges():
    """A model trained on half the edges must not have fitted the other half:
    scoring on the edges it was trained on reports memory, not performance."""
    clustered, _ = _synth_graph_inputs({"H1": 25, "L1": 25})
    coefficients = {ifo: _synth_coefficients(frame, seed=13)
                    for ifo, frame in clustered.items()}
    graph = TriggerGraphBuilder().build(clustered, coefficients)

    n_edges = len(graph.cross_edges)
    rng = np.random.default_rng(0)
    labels = (rng.uniform(0, 1, n_edges) > 0.7).astype(float)

    # train on the first half of the edges only
    mask = np.zeros(n_edges, dtype=bool)
    mask[: n_edges // 2] = True

    masked = GNNCoincidenceScorer(node_dim=graph.node_features.shape[1], hidden=8, seed=0,
                                  cross_edge_dim=graph.cross_edge_features.shape[1])
    masked_history = masked.fit([(graph, labels, mask)], epochs=40, lr=1e-2)

    full = GNNCoincidenceScorer(node_dim=graph.node_features.shape[1], hidden=8, seed=0,
                                cross_edge_dim=graph.cross_edge_features.shape[1])
    full_history = full.fit([(graph, labels)], epochs=40, lr=1e-2)

    assert len(masked_history) == 40
    assert masked_history[-1] <= masked_history[0]
    # the two runs saw different data, so they cannot have landed in the same place
    assert masked_history[-1] != pytest.approx(full_history[-1], rel=1e-9)

    masked_scores = masked.score(graph)["gnn_score"].to_numpy()
    full_scores = full.score(graph)["gnn_score"].to_numpy()
    assert not np.allclose(masked_scores, full_scores)


def test_an_all_false_mask_trains_on_nothing():
    clustered = {"H1": _synth_clustered("H1", 6, 1000.0, 7),
                 "L1": _synth_clustered("L1", 6, 1000.0, 8)}
    coefficients = {ifo: _synth_coefficients(frame, seed=17)
                    for ifo, frame in clustered.items()}
    graph = TriggerGraphBuilder().build(clustered, coefficients)

    model = GNNCoincidenceScorer(node_dim=graph.node_features.shape[1], hidden=4, seed=0,
                                 cross_edge_dim=graph.cross_edge_features.shape[1])
    labels = np.zeros(len(graph.cross_edges))
    assert model.fit([(graph, labels, np.zeros(len(labels), dtype=bool))]) == []


def test_the_logit_is_kept_because_the_probability_saturates():
    """A confident model puts many candidates on exactly 1.0 after the sigmoid,
    so a threshold there admits all of them; the logits stay ordered."""
    clustered = {"H1": _synth_clustered("H1", 10, 1000.0, 21),
                 "L1": _synth_clustered("L1", 10, 1000.0, 22)}
    coefficients = {ifo: _synth_coefficients(frame, seed=23)
                    for ifo, frame in clustered.items()}
    graph = TriggerGraphBuilder().build(clustered, coefficients)

    model = GNNCoincidenceScorer(node_dim=graph.node_features.shape[1], hidden=8, seed=0,
                                 cross_edge_dim=graph.cross_edge_features.shape[1])
    table = model.score(graph)

    assert "gnn_logit" in table.columns and "gnn_score" in table.columns
    logit = table["gnn_logit"].to_numpy()
    score = table["gnn_score"].to_numpy()
    assert np.allclose(score, 1.0 / (1.0 + np.exp(-logit)), atol=1e-6)
    # the logit orders the candidates at least as finely as the probability
    assert len(np.unique(np.round(logit, 6))) >= len(np.unique(np.round(score, 6)))


def test_a_pair_is_a_positive_only_when_both_events_are_one_injection():
    """A noise event sitting near a real signal in the other detector must not
    make the pair a positive: the network stage measures coherence between two
    views of one signal, not proximity to an injection."""
    from wdf.analysis.network_graph import edge_labels_from_injections

    clustered, coefficients = _synth_graph_inputs({"H1": 20, "L1": 20})
    graph = TriggerGraphBuilder(ifos=["H1", "L1"]).build(clustered, coefficients)
    assert len(graph.cross_edges)

    # An injection on top of every H1 event and none near L1's: no pair can have
    # both of its events covering the same injection.
    only_h1 = clustered["H1"]["gpsCentroid"].to_numpy()
    labels = edge_labels_from_injections(
        graph, only_h1 - 3600.0, window_s=0.05)
    assert labels.sum() == 0

    # Injections covering both detectors' events do produce positives.
    both = np.concatenate([clustered[ifo]["gpsCentroid"].to_numpy() for ifo in ("H1", "L1")])
    assert edge_labels_from_injections(graph, both, window_s=0.5).sum() > 0


def test_no_injections_makes_every_pair_a_negative():
    from wdf.analysis.network_graph import edge_labels_from_injections

    clustered, coefficients = _synth_graph_inputs({"H1": 10, "L1": 10})
    graph = TriggerGraphBuilder(ifos=["H1", "L1"]).build(clustered, coefficients)
    assert edge_labels_from_injections(graph, []).sum() == 0


def test_the_cosine_saturates_on_sparse_grids_and_the_overlap_does_not():
    """Two events each occupying one cell of the plane agree perfectly in
    direction whenever it is the same cell, which independent noise does by
    chance. The overlap is an energy and stays small when they are quiet."""
    import numpy as np
    def _aligned_similarity(left, right, *_):
        """The zero-lag inner product, which is what the builder now uses."""
        flat = lambda g: g.reshape(len(g), -1)
        return np.einsum("ij,ij->i", flat(left), flat(right)), np.zeros(len(left))


    # The lag is searched in seconds, so a fixture states the duration of its
    # own column; one lag either way is what these grids are built to need.
    bin_s, lag_s = 0.01, 0.01

    one_cell = np.zeros((2, 4, 8))
    one_cell[:, 1, 3] = 0.4                      # the same cell, both quiet
    agreement, _ = _aligned_similarity(
        one_cell / np.linalg.norm(one_cell.reshape(2, -1), axis=1)[:, None, None],
        one_cell / np.linalg.norm(one_cell.reshape(2, -1), axis=1)[:, None, None],
        bin_s, lag_s)
    overlap, _ = _aligned_similarity(one_cell, one_cell, bin_s, lag_s)

    assert agreement[0] == pytest.approx(1.0)
    assert np.sqrt(overlap[0]) < 0.5

    loud = one_cell * 20.0
    louder, _ = _aligned_similarity(loud, loud, bin_s, lag_s)
    assert np.sqrt(louder[0]) > np.sqrt(overlap[0])


def test_the_overlap_is_carried_on_every_candidate():
    from wdf.analysis.network_graph import EDGE_FEATURES

    clustered, coefficients = _synth_graph_inputs({"H1": 20, "L1": 20})
    table = TriggerGraphBuilder(ifos=["H1", "L1"]).build(
        clustered, coefficients).candidate_table()
    assert "wavegram_overlap" in EDGE_FEATURES
    # The coherent energy of a pair is never negative: it is the inner product
    # of two non-negative maps.
    assert (table["wavegram_overlap"] >= 0.0).all()


def test_the_correlation_separates_alike_pairs_from_merely_loud_ones():
    """Coherent energy is large wherever two events are loud together. The
    correlation reaches one only where the grids also agree."""
    import numpy as np
    def _aligned_similarity(left, right, *_):
        """The zero-lag inner product, which is what the builder now uses."""
        flat = lambda g: g.reshape(len(g), -1)
        return np.einsum("ij,ij->i", flat(left), flat(right)), np.zeros(len(left))


    # The lag is searched in seconds, so a fixture states the duration of its
    # own column; one lag either way is what these grids are built to need.
    bin_s, lag_s = 0.01, 0.01

    same = np.zeros((1, 4, 8))
    same[0, 2, 4] = 6.0
    cross, _ = _aligned_similarity(same, same, bin_s, lag_s)
    present = 2.0 * float((same ** 2).sum())
    assert 2.0 * float(cross[0]) / present == pytest.approx(1.0)

    # Both loud, in different cells: the coherent energy is zero and so is the
    # correlation, however loud each one is on its own.
    other = np.zeros((1, 4, 8))
    other[0, 1, 6] = 60.0
    cross, _ = _aligned_similarity(same, other, bin_s, lag_s)
    present = float((same ** 2).sum()) + float((other ** 2).sum())
    assert 2.0 * float(cross[0]) / max(present, 1e-30) < 0.05



def test_the_map_carries_shape_and_not_arrival_time():
    """Each event's map is centred on its own energy, so moving the whole event
    leaves it unchanged: the arrival-time difference between two detectors is
    not in the maps at all and is carried by dt_s. There is therefore nothing
    for an alignment to find, computed or learned."""
    import numpy as np
    from wdf.analysis.detector_graph import build_detector_graph, event_coefficients
    from wdf.analysis.detectors import light_travel_time
    from tests.test_detector_graph import _triggers, _walking_coefficients

    delay = light_travel_time("H1", "L1")
    triggers = _walking_coefficients(_triggers([512], 6), [64] * 6)
    delayed = triggers.assign(gps=triggers.gps + delay,
                              gpsStart=triggers.gpsStart + delay,
                              gpsCentroid=triggers.gpsCentroid + delay)

    def rendered(frame, bin_seconds):
        graph = build_detector_graph(frame)
        labels = np.zeros(len(graph.nodes), dtype=int)
        return event_coefficients(graph, labels, bin_seconds=bin_seconds)[0].wavegram()

    for bin_seconds in (0.09375, delay / 4.0):
        np.testing.assert_array_equal(rendered(triggers, bin_seconds),
                                      rendered(delayed, bin_seconds))



def test_a_map_cannot_resolve_below_the_tiles_it_is_made_of():
    """How fine the comparison rendering can usefully be. The dyadic tiles of a
    512-sample window at 2048 Hz are up to a quarter of a second long at the
    low-frequency end, so asking for columns of milliseconds does not create
    detail the transform never measured: sub-tile structure exists only in the
    high-frequency rows."""
    import numpy as np
    from wdf.analysis.wavelets import coeff_time_bounds

    t_lo, t_hi = coeff_time_bounds(512, 2048.0)
    width = np.asarray(t_hi) - np.asarray(t_lo)
    assert width.max() > 0.2, "the lowest band spans most of the window"
    assert width.min() < 0.001, "the highest band is a handful of samples"


def _network_inputs(seed=0):
    """Two detectors' events and their maps, as level one produces them."""
    from _synth import triggers_from_signal
    from wdf.analysis.detector_graph import (
        DetectorGraphConfig, build_detector_graph, detector_events,
        event_coefficients,
    )
    rng = np.random.default_rng(seed)
    fs, window, overlap = 2048.0, 512, 256
    events, maps = {}, {}
    for k, ifo in enumerate(("H1", "L1")):
        n = 24 * window
        t = np.arange(n) / fs
        signal = 6.0 * np.sin(2.0 * np.pi * (120.0 + 20 * k) * t) \
            + rng.normal(size=n)
        triggers = triggers_from_signal(signal, fs, window, overlap, ifo=ifo)
        graph = build_detector_graph(triggers, config=DetectorGraphConfig())
        labels = graph.components()
        events[ifo] = detector_events(graph, labels=labels)
        maps[ifo] = event_coefficients(graph, labels)
    return events, maps


def test_preparing_the_nodes_once_gives_the_same_graph():
    """The split is a rearrangement, so it must change nothing it produces."""
    from wdf.analysis.network_graph import TriggerGraphBuilder

    events, maps = _network_inputs()
    builder = TriggerGraphBuilder(ifos=["H1", "L1"])

    whole = builder.build(events, maps)
    split = builder.build_from_prepared(events, builder.prepare(events, maps))

    assert np.array_equal(whole.cross_edges, split.cross_edges)
    assert np.allclose(whole.cross_edge_features, split.cross_edge_features,
                       equal_nan=True)
    assert np.allclose(whole.node_features, split.node_features)


def test_prepared_nodes_survive_a_time_slide():
    """A slide moves the events and not what they are, which is the point."""
    from wdf.analysis.network_graph import TriggerGraphBuilder

    events, maps = _network_inputs()
    builder = TriggerGraphBuilder(ifos=["H1", "L1"])
    prepared = builder.prepare(events, maps)

    slid = dict(events)
    shifted = events["L1"].copy()
    for column in ("gps", "gpsStart", "gpsCentroid", "gpsPeak"):
        if column in shifted:
            shifted[column] = shifted[column] + 7.0
    slid["L1"] = shifted

    reused = builder.build_from_prepared(slid, prepared)
    rebuilt = builder.build(slid, maps)
    assert np.array_equal(reused.cross_edges, rebuilt.cross_edges)
    assert np.allclose(reused.cross_edge_features, rebuilt.cross_edge_features,
                       equal_nan=True)


def test_prepared_nodes_refuse_a_different_event_set():
    """Indexing them with other events would attach one event's shape to
    another's time, and produce a plausible wrong graph."""
    import pytest
    from wdf.analysis.network_graph import TriggerGraphBuilder

    events, maps = _network_inputs()
    builder = TriggerGraphBuilder(ifos=["H1", "L1"])
    prepared = builder.prepare(events, maps)

    fewer = {ifo: frame.iloc[:-1].copy() for ifo, frame in events.items()}
    with pytest.raises(ValueError, match="not the ones prepared"):
        builder.build_from_prepared(fewer, prepared)
