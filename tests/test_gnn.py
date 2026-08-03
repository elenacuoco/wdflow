from itertools import combinations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from wdf.analysis.gnn import TriggerGraphBuilder, GNNCoincidenceScorer


def _synth_clustered(ifo, n, t0, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(dict(
        cluster_id=range(n), ifo=ifo,
        gpsStart=t0 + np.sort(rng.uniform(0, 100, n)),
        gpsMax=t0 + np.sort(rng.uniform(0, 100, n)),
        snrMean=rng.uniform(1, 10, n), snrMax=rng.uniform(1, 20, n),
        freqMean=rng.uniform(50, 300, n), freqMax=rng.uniform(300, 500, n),
        freqMin=rng.uniform(20, 50, n), duration=rng.uniform(0.05, 0.5, n),
        wave="BsplineC309", n_triggers=rng.integers(1, 5, n), gps_span_s=rng.uniform(0, 1, n),
    ))


def test_graph_builder_produces_edges():
    clustered = {"H1": _synth_clustered("H1", 20, 1000.0, 1), "L1": _synth_clustered("L1", 20, 1000.0, 2)}
    graph = TriggerGraphBuilder(intra_ifo_window_s=5.0, cross_ifo_window_s=2.0).build(clustered)
    assert graph.node_features.shape[0] == 40
    assert graph.cross_edges.shape[1] == 2


def test_scorer_forward_and_fit_reduce_loss():
    clustered = {"H1": _synth_clustered("H1", 15, 1000.0, 3), "L1": _synth_clustered("L1", 15, 1000.0, 4)}
    graph = TriggerGraphBuilder(cross_ifo_window_s=5.0).build(clustered)
    model = GNNCoincidenceScorer(node_dim=graph.node_features.shape[1], hidden=8, seed=0)

    table = model.score(graph)
    assert "gnn_score" in table.columns
    assert ((table["gnn_score"] >= 0) & (table["gnn_score"] <= 1)).all()

    rng = np.random.default_rng(0)
    labels = (rng.uniform(0, 1, len(graph.cross_edges)) > 0.8).astype(float)
    history = model.fit([(graph, labels)], epochs=30, lr=1e-2)
    assert len(history) > 0
    assert history[-1] <= history[0]


def _reference_build(builder, clustered):
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

    intra_edges, intra_feats = [], []
    for idxs in idx_by_ifo.values():
        for i, j in combinations(idxs, 2):
            dt = float(nodes_df.at[i, "gpsMax"] - nodes_df.at[j, "gpsMax"])
            if abs(dt) <= builder.intra_ifo_window_s:
                intra_edges.append((i, j)); intra_feats.append([dt])
                intra_edges.append((j, i)); intra_feats.append([-dt])
    cross_edges, cross_feats = [], []
    for ifo_a, ifo_b in combinations(ifos, 2):
        for i in idx_by_ifo[ifo_a]:
            for j in idx_by_ifo[ifo_b]:
                dt = float(nodes_df.at[i, "gpsMax"] - nodes_df.at[j, "gpsMax"])
                if abs(dt) <= builder.cross_ifo_window_s:
                    dfreq = float(nodes_df.at[i, "freqMean"] - nodes_df.at[j, "freqMean"])
                    dsnr = float(nodes_df.at[i, "snrMax"] - nodes_df.at[j, "snrMax"])
                    cross_edges.append((i, j)); cross_feats.append([dt, dfreq, dsnr])
    return (np.array(intra_edges, dtype=np.int64).reshape(-1, 2),
            np.array(intra_feats).reshape(-1, 1),
            np.array(cross_edges, dtype=np.int64).reshape(-1, 2),
            np.array(cross_feats).reshape(-1, 3))


def test_vectorized_build_matches_reference_loop():
    clustered = {"H1": _synth_clustered("H1", 60, 1000.0, 1), "L1": _synth_clustered("L1", 45, 1000.0, 2)}
    builder = TriggerGraphBuilder(intra_ifo_window_s=5.0, cross_ifo_window_s=2.0, ifos=["H1", "L1"])

    ref_intra_e, ref_intra_f, ref_cross_e, ref_cross_f = _reference_build(builder, clustered)
    graph = builder.build(clustered)

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
        graph = TriggerGraphBuilder(cross_ifo_window_s=5.0).build(clustered)
        rng = np.random.default_rng(seed)
        labels = (rng.uniform(0, 1, len(graph.cross_edges)) > 0.8).astype(float)
        examples.append((graph, labels))

    model = GNNCoincidenceScorer(node_dim=examples[0][0].node_features.shape[1], hidden=8, seed=0)
    history_full_batch = model.fit(examples, epochs=20, lr=1e-2)
    assert len(history_full_batch) > 0
    assert history_full_batch[-1] <= history_full_batch[0]

    # batch_size=1 (one segment per chunk, gradients accumulated/averaged
    # across chunks each epoch) should also train without error -- checks
    # the chunking path, not just the single-Batch-per-epoch default.
    model2 = GNNCoincidenceScorer(node_dim=examples[0][0].node_features.shape[1], hidden=8, seed=0)
    history_chunked = model2.fit(examples, epochs=20, lr=1e-2, batch_size=1)
    assert len(history_chunked) > 0


def test_scorer_device_selection():
    clustered = {"H1": _synth_clustered("H1", 5, 1000.0, 1), "L1": _synth_clustered("L1", 5, 1000.0, 2)}
    graph = TriggerGraphBuilder(cross_ifo_window_s=5.0).build(clustered)

    default_model = GNNCoincidenceScorer(node_dim=graph.node_features.shape[1], hidden=4, seed=0)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert default_model.device == expected
    assert next(default_model.parameters()).device.type == expected

    cpu_model = GNNCoincidenceScorer(node_dim=graph.node_features.shape[1], hidden=4, seed=0, device="cpu")
    assert cpu_model.device == "cpu"
    table = cpu_model.score(graph)
    assert "gnn_score" in table.columns
