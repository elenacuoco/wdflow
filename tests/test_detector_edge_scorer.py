import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from wdf.analysis.detector_graph import build_detector_graph, detector_events
from wdf.analysis.gnn import DetectorEdgeScorer, edge_labels_from_injections
from tests.test_detector_graph import _triggers


def _graph(scales=(256, 512), n=20, seed=0):
    return build_detector_graph(_triggers(list(scales), n, seed=seed))


def _model(graph, **kwargs):
    return DetectorEdgeScorer(node_dim=graph.node_features.shape[1],
                              edge_dim=graph.edge_features.shape[1],
                              hidden=8, **kwargs)


def test_an_edge_is_a_positive_when_both_triggers_share_an_injection():
    graph = _graph()
    labels = edge_labels_from_injections(graph, [1000.5])
    assert labels.shape == (len(graph.edges),)
    assert set(np.unique(labels)) <= {0.0, 1.0}
    assert labels.sum() > 0


def test_no_injection_makes_every_edge_a_negative():
    graph = _graph()
    assert edge_labels_from_injections(graph, []).sum() == 0
    assert edge_labels_from_injections(graph, [50000.0]).sum() == 0


def test_fitting_reduces_the_loss():
    graph = _graph()
    labels = edge_labels_from_injections(graph, [1000.5, 1001.2])
    history = _model(graph).fit([(graph, labels)], epochs=30)
    assert len(history) == 30
    assert history[-1] < history[0]


def test_pruning_edges_splits_what_geometry_had_merged():
    """Connected components over every admissible edge chain noise together;
    the model exists to decide which of them survive."""
    graph = _graph()
    labels = edge_labels_from_injections(graph, [1000.5, 1001.2])
    model = _model(graph)
    model.fit([(graph, labels)], epochs=40)
    scored = model.score(graph)
    kept = scored["edge_score"].to_numpy() > 0.5
    assert detector_events(graph, labels=graph.components(kept)).shape[0] >= \
        detector_events(graph).shape[0]


def test_scoring_keeps_the_edge_table_s_own_columns():
    graph = _graph()
    scored = _model(graph).score(graph)
    assert {"node_i", "node_j", "dt_s", "edge_logit", "edge_score"} <= set(scored.columns)
    assert len(scored) == len(graph.edges)


def test_the_scaling_is_carried_with_the_model_not_measured_per_graph():
    """A scaling measured on each graph would have the model read two stretches
    on two different scales, which is the comparison it exists to make."""
    graph = _graph()
    model = _model(graph)
    assert torch.allclose(model.feature_scale, torch.ones_like(model.feature_scale))
    model.fit([(graph, edge_labels_from_injections(graph, [1000.5]))], epochs=2)
    assert not torch.allclose(model.feature_scale, torch.ones_like(model.feature_scale))


def test_a_mask_keeps_the_untrained_edges_out_of_the_loss():
    graph = _graph()
    labels = edge_labels_from_injections(graph, [1000.5, 1001.2])
    mask = np.zeros(len(labels), dtype=bool)
    mask[: len(labels) // 2] = True
    masked = _model(graph, seed=1).fit([(graph, labels, mask)], epochs=20)
    every = _model(graph, seed=1).fit([(graph, labels)], epochs=20)
    assert masked[-1] != pytest.approx(every[-1], rel=1e-9)


def test_an_empty_mask_trains_on_nothing():
    graph = _graph()
    labels = edge_labels_from_injections(graph, [1000.5])
    history = _model(graph).fit(
        [(graph, labels, np.zeros(len(labels), dtype=bool))], epochs=5)
    assert all(value == 0.0 for value in history)


def test_a_graph_without_edges_is_not_fitted():
    empty = build_detector_graph(pd.DataFrame(columns=["gps", "n_coeff", "fs", "EnWDF"]))
    assert DetectorEdgeScorer(node_dim=1, edge_dim=9, hidden=4).fit([(empty, [])]) == []
