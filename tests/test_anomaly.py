"""The network stage as an anomaly detector fitted on time slides."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wdf.analysis.anomaly import BackgroundAnomalyScorer
from wdf.analysis.network_graph import TriggerGraphBuilder
from tests.test_gnn import _synth_graph_inputs


def _graph(sizes, seed):
    clustered, coefficients = _synth_graph_inputs(sizes, seed=seed)
    return TriggerGraphBuilder(ifos=["H1", "L1"]).build(clustered, coefficients)


def _fitted(graphs, epochs=60):
    model = BackgroundAnomalyScorer(
        node_dim=graphs[0].node_features.shape[1],
        cross_edge_dim=graphs[0].cross_edge_features.shape[1],
        hidden=8, seed=0)
    return model, model.fit(graphs, epochs=epochs, lr=5e-2)


def test_it_learns_the_accidentals_it_was_shown():
    graphs = [_graph({"H1": 20, "L1": 20}, seed=s) for s in (1, 2, 3)]
    _, history = _fitted(graphs)
    assert history[-1] < history[0]


def test_nothing_but_the_background_is_ever_read():
    """No label and no injection enters the fit: it takes graphs and nothing
    else, which is what keeps the search un-modelled."""
    import inspect

    parameters = inspect.signature(BackgroundAnomalyScorer.fit).parameters
    assert set(parameters) == {"self", "graphs", "epochs", "lr"}


def test_a_candidate_unlike_the_accidentals_scores_higher():
    """What the ranking rests on. Fitted on accidental coincidences, the model
    reconstructs those; a population it never saw it reconstructs worse."""
    graphs = [_graph({"H1": 25, "L1": 25}, seed=s) for s in (1, 2, 3)]
    model, _ = _fitted(graphs, epochs=150)

    ordinary = model.score(graphs[0])["anomaly_score"].to_numpy()

    # The same graph with edge features displaced well outside the range the
    # accidentals occupy.
    strange = _graph({"H1": 25, "L1": 25}, seed=1)
    strange.cross_edge_features = (strange.cross_edge_features
                                   + 10.0 * strange.cross_edge_features.std(axis=0)
                                   + 1.0).astype(np.float32)
    unusual = model.score(strange)["anomaly_score"].to_numpy()

    assert np.median(unusual) > np.median(ordinary)


def test_no_candidate_gives_an_empty_table_and_no_crash():
    graph = _graph({"H1": 15, "L1": 15}, seed=4)
    graph.cross_edges = np.zeros((0, 2), dtype=np.int64)
    graph.cross_edge_features = np.zeros((0, len(graph.cross_edge_features[0])),
                                         dtype=np.float32)
    model = BackgroundAnomalyScorer(node_dim=graph.node_features.shape[1],
                                    cross_edge_dim=graph.cross_edge_features.shape[1],
                                    hidden=8, seed=0)
    assert model.score(graph).empty


def test_the_pair_has_no_preferred_order():
    """Which detector the graph builder happened to name first is not physics,
    so the score must not depend on it."""
    graphs = [_graph({"H1": 20, "L1": 20}, seed=s) for s in (1, 2)]
    model, _ = _fitted(graphs)

    graph = graphs[0]
    swapped = _graph({"H1": 20, "L1": 20}, seed=1)
    swapped.cross_edges = swapped.cross_edges[:, ::-1].copy()

    np.testing.assert_allclose(model.score(graph)["anomaly_score"].to_numpy(),
                               model.score(swapped)["anomaly_score"].to_numpy(),
                               rtol=1e-5, atol=1e-6)
