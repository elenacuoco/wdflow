from itertools import combinations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from wdf.analysis.gnn import TriggerGraphBuilder, GNNCoincidenceScorer


FS, WINDOW, OVERLAP, N_COEFF = 2048.0, 512, 128, 512


def _synth_clustered(ifo, n, t0, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(dict(
        cluster_id=range(n), ifo=ifo,
        gpsStart=t0 + np.sort(rng.uniform(0, 100, n)),
        gpsPeak=t0 + np.sort(rng.uniform(0, 100, n)),
        snrMean=rng.uniform(1, 10, n), EnWDF=rng.uniform(1, 20, n),
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
    clustered = {ifo: _synth_clustered(ifo, n, t0, seed + i)
                 for i, (ifo, n) in enumerate(sizes.items())}
    coefficients = {ifo: _synth_coefficients(frame, seed + i)
                    for i, (ifo, frame) in enumerate(clustered.items())}
    return clustered, coefficients


def test_graph_builder_produces_edges():
    clustered = {"H1": _synth_clustered("H1", 20, 1000.0, 1), "L1": _synth_clustered("L1", 20, 1000.0, 2)}
    coefficients = {ifo: _synth_coefficients(frame) for ifo, frame in clustered.items()}
    graph = TriggerGraphBuilder(intra_ifo_window_s=5.0, cross_ifo_window_s=2.0).build(clustered, coefficients)
    assert graph.node_features.shape[0] == 40
    assert graph.cross_edges.shape[1] == 2


def test_scorer_forward_and_fit_reduce_loss():
    clustered = {"H1": _synth_clustered("H1", 15, 1000.0, 3), "L1": _synth_clustered("L1", 15, 1000.0, 4)}
    coefficients = {ifo: _synth_coefficients(frame) for ifo, frame in clustered.items()}
    graph = TriggerGraphBuilder(cross_ifo_window_s=5.0).build(clustered, coefficients)
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
    cross_edges, cross_feats = [], []
    for ifo_a, ifo_b in combinations(ifos, 2):
        for i in idx_by_ifo[ifo_a]:
            for j in idx_by_ifo[ifo_b]:
                dt = float(nodes_df.at[i, "gpsPeak"] - nodes_df.at[j, "gpsPeak"])
                if abs(dt) <= builder.cross_ifo_window_s:
                    similarity = float(shapes[i] @ shapes[j])
                    cross_edges.append((i, j)); cross_feats.append([dt, similarity])
    return (np.array(intra_edges, dtype=np.int64).reshape(-1, 2),
            np.array(intra_feats).reshape(-1, 1),
            np.array(cross_edges, dtype=np.int64).reshape(-1, 2),
            np.array(cross_feats).reshape(-1, 2))


def test_vectorized_build_matches_reference_loop():
    clustered = {"H1": _synth_clustered("H1", 60, 1000.0, 1), "L1": _synth_clustered("L1", 45, 1000.0, 2)}
    coefficients = {ifo: _synth_coefficients(frame) for ifo, frame in clustered.items()}
    builder = TriggerGraphBuilder(intra_ifo_window_s=5.0, cross_ifo_window_s=2.0, ifos=["H1", "L1"])

    ref_intra_e, ref_intra_f, ref_cross_e, ref_cross_f = _reference_build(
        builder, clustered, coefficients)
    graph = builder.build(clustered, coefficients)

    def edge_set(edges):
        return set(map(tuple, edges.tolist()))

    assert edge_set(ref_intra_e) == edge_set(graph.intra_edges)
    assert edge_set(ref_cross_e) == edge_set(graph.cross_edges)

    ref_map = {tuple(e): f for e, f in zip(ref_intra_e, ref_intra_f)}
    for e, f in zip(graph.intra_edges, graph.intra_edge_features):
        np.testing.assert_allclose(f, ref_map[tuple(e.tolist())], rtol=1e-5, atol=1e-5)
    ref_map = {tuple(e): f for e, f in zip(ref_cross_e, ref_cross_f)}
    for e, f in zip(graph.cross_edges, graph.cross_edge_features):
        np.testing.assert_allclose(f, ref_map[tuple(e.tolist())], rtol=1e-5, atol=1e-5)


def test_fit_batches_multiple_segments_together():
    """batch_size=None (the default) should put every segment's graph into
    one torch_geometric Batch per epoch -- exercises Batch.from_data_list
    across graphs of different sizes (different node/edge counts), which is
    exactly the multi-segment throughput path this module exists for.
    """
    examples = []
    for seed, (n_h1, n_l1) in enumerate([(15, 15), (8, 12), (20, 5)]):
        clustered = {
            "H1": _synth_clustered("H1", n_h1, 1000.0 + seed * 200, seed * 2),
            "L1": _synth_clustered("L1", n_l1, 1000.0 + seed * 200, seed * 2 + 1),
        }
        coefficients = {ifo: _synth_coefficients(frame) for ifo, frame in clustered.items()}
        graph = TriggerGraphBuilder(cross_ifo_window_s=5.0).build(clustered, coefficients)
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
    graph = TriggerGraphBuilder(cross_ifo_window_s=5.0).build(clustered, coefficients)

    default_model = GNNCoincidenceScorer(node_dim=graph.node_features.shape[1], hidden=4, seed=0,
                                     cross_edge_dim=graph.cross_edge_features.shape[1])
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert default_model.device == expected
    assert next(default_model.parameters()).device.type == expected

    cpu_model = GNNCoincidenceScorer(node_dim=graph.node_features.shape[1], hidden=4, seed=0, device="cpu",
                                 cross_edge_dim=graph.cross_edge_features.shape[1])
    assert cpu_model.device == "cpu"
    table = cpu_model.score(graph)
    assert "gnn_score" in table.columns


def test_the_graph_is_built_from_the_coefficients_not_from_scalars():
    """Two events with identical scalar summaries but different coefficients
    must give different node features, which is the whole reason the graph
    reads the coefficient matrices."""
    clustered = {"H1": _synth_clustered("H1", 6, 1000.0, 1),
                 "L1": _synth_clustered("L1", 6, 1000.0, 2)}
    coefficients = {ifo: _synth_coefficients(frame, seed=7)
                    for ifo, frame in clustered.items()}
    builder = TriggerGraphBuilder(cross_ifo_window_s=5.0, ifos=["H1", "L1"])

    graph = builder.build(clustered, coefficients)

    # 32 time bins over log2(512) + 1 octave rows, plus one column per detector
    assert graph.node_features.shape[1] == 10 * 32 + 2
    assert graph.cross_edge_features.shape[1] == 2

    other = {ifo: _synth_coefficients(frame, seed=99)
             for ifo, frame in clustered.items()}
    changed = builder.build(clustered, other)
    assert not np.allclose(graph.node_features, changed.node_features)


def test_a_candidate_carries_the_agreement_between_the_two_wavegrams():
    clustered = {"H1": _synth_clustered("H1", 8, 1000.0, 3),
                 "L1": _synth_clustered("L1", 8, 1000.0, 4)}
    coefficients = {ifo: _synth_coefficients(frame, seed=11)
                    for ifo, frame in clustered.items()}
    graph = TriggerGraphBuilder(cross_ifo_window_s=5.0).build(clustered, coefficients)

    table = graph.candidate_table()
    assert "wavegram_similarity" in table.columns
    # a normalised inner product of non-negative grids
    assert ((table["wavegram_similarity"] >= -1e-6)
            & (table["wavegram_similarity"] <= 1 + 1e-6)).all()


def test_fit_only_learns_from_the_masked_edges():
    """A model trained on half the edges must not have fitted the other half:
    scoring on the edges it was trained on reports memory, not performance."""
    clustered = {"H1": _synth_clustered("H1", 25, 1000.0, 5),
                 "L1": _synth_clustered("L1", 25, 1000.0, 6)}
    coefficients = {ifo: _synth_coefficients(frame, seed=13)
                    for ifo, frame in clustered.items()}
    graph = TriggerGraphBuilder(cross_ifo_window_s=5.0).build(clustered, coefficients)

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
    graph = TriggerGraphBuilder(cross_ifo_window_s=5.0).build(clustered, coefficients)

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
    graph = TriggerGraphBuilder(cross_ifo_window_s=5.0).build(clustered, coefficients)

    model = GNNCoincidenceScorer(node_dim=graph.node_features.shape[1], hidden=8, seed=0,
                                 cross_edge_dim=graph.cross_edge_features.shape[1])
    table = model.score(graph)

    assert "gnn_logit" in table.columns and "gnn_score" in table.columns
    logit = table["gnn_logit"].to_numpy()
    score = table["gnn_score"].to_numpy()
    assert np.allclose(score, 1.0 / (1.0 + np.exp(-logit)), atol=1e-6)
    # the logit orders the candidates at least as finely as the probability
    assert len(np.unique(np.round(logit, 6))) >= len(np.unique(np.round(score, 6)))
