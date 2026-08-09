"""Graph-neural-network coincidence scorer, run ALONGSIDE (not replacing)
CoincidenceFinder's classical time-window method: WDF's own detection stays
single-detector and blind, cross-detector combination is a separate,
comparable-but-independent step, and a learned combiner should be validated
against the classical one directly rather than replacing it outright.

Built on `torch_geometric`: each segment's graph is tiny (tens of clustered
events per detector, at most a few hundred candidate edges), but a pipeline
running over thousands of segments needs many such small graphs trained/
scored together, not one Python-level forward pass per segment. `torch_geometric`
gives batched sparse message passing (`Batch.from_data_list`) for that, plus a
maintained scatter/aggregation backend, instead of a hand-rolled per-graph loop.

Training positives are real catalogued events in the analyzed segment(s);
negatives are accidental/background cross-detector edges (including
time-slide background from significance.BackgroundEstimator). With only a
handful of real positive events even across several long continuous segments,
`fit` is still, numerically, closer to a proof-of-concept calibration than a
large-scale production run -- a synthetic-injection extension (bilby/pycbc)
to get a larger, better-balanced training set is noted as future work.

Requires `pip install -e ".[gnn]"` (torch + torch_geometric). Importing this
module without them installed raises ImportError with that instruction -- the
rest of wdfLib works without either.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wdf.analysis.network_graph import (
    EDGE_FEATURES,
    N_EDGE_FEATURES,
    WAVEGRAM_TIME_BINS,
    TriggerGraph,
    TriggerGraphBuilder,
)

try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import MessagePassing
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "wdfLib.gnn requires torch and torch_geometric. Install with: pip install -e '.[gnn]'"
    ) from _e

class _CandidateData(Data):
    """`Data` with a second, separate edge set (`cross_edge_index`/
    `cross_edge_attr`) for the cross-IFO candidate edges being scored --
    kept apart from `edge_index` (the intra-IFO message-passing edges)
    since the two play different roles and shouldn't be convolved together.
    `Batch.from_data_list` needs to know `cross_edge_index` holds node
    indices too (so they get offset per-graph like `edge_index` does), which
    is what `__inc__` below tells it.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "cross_edge_index":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "cross_edge_index":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def _to_pyg_data(graph: TriggerGraph, labels: np.ndarray | None = None,
                 mask: np.ndarray | None = None) -> _CandidateData:
    data = _CandidateData(
        x=torch.from_numpy(graph.node_features),
        edge_index=torch.from_numpy(graph.intra_edges.T).long().reshape(2, -1),
        edge_attr=torch.from_numpy(graph.intra_edge_features),
        cross_edge_index=torch.from_numpy(graph.cross_edges.T).long().reshape(2, -1),
        cross_edge_attr=torch.from_numpy(graph.cross_edge_features),
        num_nodes=graph.node_features.shape[0],
    )
    if labels is not None:
        data.y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    # A mask selects which edges contribute to the loss. It travels with the
    # graph so `Batch.from_data_list` concatenates it like the labels.
    n_edges = graph.cross_edges.shape[0]
    keep = np.ones(n_edges, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    data.train_mask = torch.from_numpy(keep)
    return data


class _IntraMessagePassing(MessagePassing):
    """One round of intra-detector message passing: each cluster's embedding
    absorbs its temporally-close same-detector neighbors (mean-aggregated,
    matching `aggr="mean"`'s behavior of returning 0 for a node with no
    incoming edges -- "no nearby context" is a real, informative state here,
    not a missing value), i.e. 'is this cluster sitting in a noisy
    neighborhood?'.
    """

    def __init__(self, hidden: int):
        super().__init__(aggr="mean")
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.update_mlp = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())

    def forward(self, h, edge_index, edge_attr):
        agg = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return self.update_mlp(torch.cat([h, agg], dim=1))

    def message(self, h_j, edge_attr):
        return self.message_mlp(torch.cat([h_j, edge_attr], dim=1))


class GNNCoincidenceScorer(nn.Module):
    """Intra-detector message passing (`_IntraMessagePassing`), then an
    edge-classification head scores cross-detector candidate edges."""

    def __init__(self, node_dim: int, hidden: int = 16, seed: int = 0,
                 device: str | None = None, cross_edge_dim: int = 2):
        """
        :type node_dim: int
        :param node_dim: width of a node's feature vector, i.e.
            `TriggerGraph.node_features.shape[1]`.
        :type hidden: int
        :param hidden: width of the hidden representation.
        :type seed: int
        :param seed: torch seed, for a reproducible initialisation.
        :type device: str or None
        :param device: torch device; CUDA when available if None.
        :type cross_edge_dim: int
        :param cross_edge_dim: number of features on a candidate edge, i.e.
            `TriggerGraph.cross_edge_features.shape[1]`.
        """
        super().__init__()
        torch.manual_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.intra_mp = _IntraMessagePassing(hidden)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 2 + cross_edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        # The feature scaling is part of the fitted model, not of a graph. A
        # scaling measured on each graph separately would have the model read
        # the foreground and the background on two different scales, which is
        # exactly the comparison it exists to make.
        self.register_buffer("feature_mean", torch.zeros(node_dim))
        self.register_buffer("feature_scale", torch.ones(node_dim))
        self.to(self.device)

    def _edge_logits_from_data(self, data) -> "torch.Tensor":
        """`data` may be a single `_CandidateData` or a `Batch` of several --
        `edge_index`/`cross_edge_index` already carry per-graph node-index
        offsets from `Batch.from_data_list`, so the same code path scores
        one segment or many at once with no special-casing.
        """
        x = data.x.to(self.device)
        x = (x - self.feature_mean) / self.feature_scale
        h = self.encoder(x)
        h = self.intra_mp(h, data.edge_index.to(self.device), data.edge_attr.to(self.device))
        if data.cross_edge_index.numel() == 0:
            return torch.zeros(0, device=self.device)
        i, j = data.cross_edge_index[0].to(self.device), data.cross_edge_index[1].to(self.device)
        ef = data.cross_edge_attr.to(self.device)
        return self.edge_head(torch.cat([h[i], h[j], ef], dim=1)).squeeze(-1)

    def edge_logits(self, graph: TriggerGraph) -> "torch.Tensor":
        return self._edge_logits_from_data(_to_pyg_data(graph))

    def score(self, graph: TriggerGraph) -> pd.DataFrame:
        """Cross-IFO candidate table (`TriggerGraph.candidate_table` schema)
        with the model's output on each candidate edge.

        Two columns are added. `gnn_score` is the sigmoid probability that the
        edge is a real astrophysical coincidence, which is what to read when a
        probability is wanted. `gnn_logit` is the same quantity before the
        sigmoid, and it is what to rank and threshold on: a confident model
        saturates the sigmoid, so many candidates land on exactly 1.0 and a
        threshold there admits every one of them, while the logits stay
        ordered.

        :type graph: TriggerGraph
        :param graph: the graph to score.
        :return: pandas.DataFrame -- the candidate table with `gnn_logit` and
            `gnn_score`.
        """
        with torch.no_grad():
            logits = self.edge_logits(graph)
            if logits.numel():
                raw = logits.cpu().numpy()
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                raw = probs = np.array([])
        table = graph.candidate_table()
        table["gnn_logit"] = raw
        table["gnn_score"] = probs
        return table

    def _fit_feature_scaling(self, graphs) -> None:
        """Measure the node-feature scaling from the graphs being fitted on."""
        features = np.vstack([g.node_features for g in graphs if len(g.node_features)])
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale == 0.0] = 1.0
        device = self.feature_mean.device
        self.feature_mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.feature_scale = torch.as_tensor(scale, dtype=torch.float32, device=device)

    def fit(
        self,
        examples: list[tuple],
        epochs: int = 100,
        lr: float = 1e-2,
        batch_size: int | None = None,
    ) -> list[float]:
        """examples: `(graph, labels)` or `(graph, labels, mask)` per segment,
        labels a 0/1 array aligned with `graph.cross_edges` (1 = real
        astrophysical coincidence, 0 = accidental/background).

        `mask` selects the edges that contribute to the loss. Supply one
        whenever the same graph is later scored, so that training and
        evaluation use disjoint edges: a model scored on the edges it was
        fitted on reports its own memory rather than its performance. Edges
        share nodes, so the split should follow the segment's time
        (`wdf.analysis.evaluation.temporal_split`) rather than be drawn at
        random.

        `batch_size` groups segments into `torch_geometric` `Batch`es for a
        single sparse forward/backward pass each, instead of one Python-level
        call per segment -- the throughput path for training across many
        segments at once. Default (None) puts every segment in one batch per
        epoch (full-batch training), the common case when `examples` is the
        whole training set for one run, not a huge out-of-core dataset.
        Returns the per-epoch loss history.
        """
        datas = [_to_pyg_data(example[0], example[1],
                              example[2] if len(example) > 2 else None)
                 for example in examples if example[0].cross_edges.size]
        datas = [d for d in datas if bool(d.train_mask.any())]
        if not datas:
            return []
        self._fit_feature_scaling([example[0] for example in examples])
        chunk_size = batch_size or len(datas)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        history = []
        for _ in range(epochs):
            opt.zero_grad()
            total, n_chunks = torch.tensor(0.0, device=self.device), 0
            for start in range(0, len(datas), chunk_size):
                batch = Batch.from_data_list(datas[start:start + chunk_size])
                logits = self._edge_logits_from_data(batch)
                keep = batch.train_mask.to(self.device)
                total = total + loss_fn(logits[keep], batch.y.to(self.device)[keep])
                n_chunks += 1
            loss = total / n_chunks
            loss.backward()
            opt.step()
            history.append(float(loss.item()))
        return history
