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

from itertools import combinations

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import MessagePassing
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "wdfLib.gnn requires torch and torch_geometric. Install with: pip install -e '.[gnn]'"
    ) from _e

NODE_FEATURE_COLUMNS = ["snrMax", "freqMean", "freqMax", "duration", "n_triggers"]


class TriggerGraph:
    """Node = clustered per-IFO event. Intra-IFO edges = temporally-close
    same-detector clusters (local-density context). Cross-IFO edges =
    candidate coincidence pairs across detectors (both true and accidental,
    deliberately, since the GNN needs negative examples to learn from).

    Plain numpy/pandas container -- the public shape callers (this module's
    own GNNCoincidenceScorer, but also notebooks) build/inspect directly.
    `GNNCoincidenceScorer` converts it to a `torch_geometric.data.Data`
    internally; nothing about this class depends on torch_geometric.
    """

    def __init__(
        self, nodes, node_features, intra_edges, intra_edge_features,
        cross_edges, cross_edge_features, ifos,
    ):
        self.nodes = nodes
        self.node_features = node_features
        self.intra_edges = intra_edges
        self.intra_edge_features = intra_edge_features
        self.cross_edges = cross_edges
        self.cross_edge_features = cross_edge_features
        self.ifos = ifos

    def candidate_table(self) -> pd.DataFrame:
        """Cross-IFO candidate edges as a DataFrame, schema-compatible with
        CoincidenceFinder.find's output (gps_candidate/dt_s/network_snr/
        n_ifos in common), so ROCCurve can be run on either interchangeably.
        """
        rows = []
        for k, (i, j) in enumerate(self.cross_edges):
            ni, nj = self.nodes.iloc[int(i)], self.nodes.iloc[int(j)]
            dt, dfreq, dsnr = self.cross_edge_features[k]
            rows.append(dict(
                candidate_id=k,
                gps_candidate=float((ni["gpsMax"] + nj["gpsMax"]) / 2),
                ifos_involved=f"{ni['ifo']},{nj['ifo']}",
                dt_s=float(dt),
                network_snr=float(np.sqrt(ni["snrMax"] ** 2 + nj["snrMax"] ** 2)),
                n_ifos=2,
                node_i=int(i),
                node_j=int(j),
            ))
        return pd.DataFrame(rows)


class TriggerGraphBuilder:
    def __init__(
        self,
        intra_ifo_window_s: float = 5.0,
        cross_ifo_window_s: float = 0.5,
        ifos: list[str] | None = None,
    ):
        self.intra_ifo_window_s = intra_ifo_window_s
        self.cross_ifo_window_s = cross_ifo_window_s
        self.ifos = ifos

    def build(self, clustered: dict[str, pd.DataFrame]) -> TriggerGraph:
        ifos = self.ifos or list(clustered.keys())
        nodes, node_ifo = [], []
        for ifo in ifos:
            for _, row in clustered[ifo].reset_index(drop=True).iterrows():
                nodes.append(row)
                node_ifo.append(ifo)
        nodes_df = pd.DataFrame(nodes).reset_index(drop=True)
        nodes_df["ifo"] = node_ifo

        feats = nodes_df[NODE_FEATURE_COLUMNS].to_numpy(dtype=float)
        log_snr = np.log10(np.clip(nodes_df["snrMax"].to_numpy(dtype=float), 1e-6, None))[:, None]
        onehot = pd.get_dummies(nodes_df["ifo"]).reindex(columns=ifos, fill_value=0).to_numpy(dtype=float)
        X = np.hstack([feats, log_snr, onehot])
        mu, sigma = X.mean(axis=0), X.std(axis=0)
        sigma[sigma == 0] = 1.0
        X = ((X - mu) / sigma).astype(np.float32)

        idx_by_ifo = {ifo: nodes_df.index[nodes_df["ifo"] == ifo].to_numpy() for ifo in ifos}
        gps = nodes_df["gpsMax"].to_numpy(dtype=float)
        freq = nodes_df["freqMean"].to_numpy(dtype=float)
        snr = nodes_df["snrMax"].to_numpy(dtype=float)

        # Vectorized replacement for a pure-Python itertools.combinations /
        # nested-for-loop pairwise scan: with clustered_events keeping every
        # DBSCAN singleton as its own node, real segments can have thousands
        # of nodes per IFO, so an O(n^2) Python loop (here, and worse, inside
        # a 100s-of-time-slides background loop that rebuilds the graph from
        # scratch each slide) dominates runtime. A numpy pairwise-distance
        # matrix does the same O(n^2) comparisons in C, not Python.
        intra_edges, intra_feats = [], []
        for idxs in idx_by_ifo.values():
            n = len(idxs)
            if n < 2:
                continue
            dt_mat = gps[idxs][:, None] - gps[idxs][None, :]
            iu, ju = np.triu_indices(n, k=1)
            dt = dt_mat[iu, ju]
            keep = np.abs(dt) <= self.intra_ifo_window_s
            i_sel, j_sel, dt_sel = idxs[iu[keep]], idxs[ju[keep]], dt[keep]
            intra_edges.append(np.column_stack([i_sel, j_sel]))
            intra_edges.append(np.column_stack([j_sel, i_sel]))
            intra_feats.append(dt_sel[:, None])
            intra_feats.append(-dt_sel[:, None])

        cross_edges, cross_feats = [], []
        for ifo_a, ifo_b in combinations(ifos, 2):
            idx_a, idx_b = idx_by_ifo[ifo_a], idx_by_ifo[ifo_b]
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue
            dt_mat = gps[idx_a][:, None] - gps[idx_b][None, :]
            keep = np.abs(dt_mat) <= self.cross_ifo_window_s
            ia, jb = np.nonzero(keep)  # row-major -> same (i outer, j inner) order as the old nested loop
            i_sel, j_sel = idx_a[ia], idx_b[jb]
            dt_sel = dt_mat[ia, jb]
            dfreq_sel = freq[idx_a][ia] - freq[idx_b][jb]
            dsnr_sel = snr[idx_a][ia] - snr[idx_b][jb]
            cross_edges.append(np.column_stack([i_sel, j_sel]))
            cross_feats.append(np.column_stack([dt_sel, dfreq_sel, dsnr_sel]))

        intra_edges = np.concatenate(intra_edges) if intra_edges else np.zeros((0, 2), dtype=np.int64)
        intra_feats = np.concatenate(intra_feats) if intra_feats else np.zeros((0, 1), dtype=np.float32)
        cross_edges = np.concatenate(cross_edges) if cross_edges else np.zeros((0, 2), dtype=np.int64)
        cross_feats = np.concatenate(cross_feats) if cross_feats else np.zeros((0, 3), dtype=np.float32)

        return TriggerGraph(
            nodes=nodes_df,
            node_features=X,
            intra_edges=intra_edges.astype(np.int64).reshape(-1, 2),
            intra_edge_features=intra_feats.astype(np.float32).reshape(-1, 1),
            cross_edges=cross_edges.astype(np.int64).reshape(-1, 2),
            cross_edge_features=cross_feats.astype(np.float32).reshape(-1, 3),
            ifos=ifos,
        )


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


def _to_pyg_data(graph: TriggerGraph, labels: np.ndarray | None = None) -> _CandidateData:
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

    def __init__(self, node_dim: int, hidden: int = 16, seed: int = 0, device: str | None = None):
        super().__init__()
        torch.manual_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.intra_mp = _IntraMessagePassing(hidden)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.to(self.device)

    def _edge_logits_from_data(self, data) -> "torch.Tensor":
        """`data` may be a single `_CandidateData` or a `Batch` of several --
        `edge_index`/`cross_edge_index` already carry per-graph node-index
        offsets from `Batch.from_data_list`, so the same code path scores
        one segment or many at once with no special-casing.
        """
        x = data.x.to(self.device)
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
        """Cross-IFO candidate table (TriggerGraph.candidate_table schema)
        with an added `gnn_score` column: sigmoid probability that the edge
        is a real astrophysical coincidence rather than accidental."""
        with torch.no_grad():
            logits = self.edge_logits(graph)
            probs = torch.sigmoid(logits).cpu().numpy() if logits.numel() else np.array([])
        table = graph.candidate_table()
        table["gnn_score"] = probs
        return table

    def fit(
        self,
        examples: list[tuple[TriggerGraph, np.ndarray]],
        epochs: int = 100,
        lr: float = 1e-2,
        batch_size: int | None = None,
    ) -> list[float]:
        """examples: (graph, labels) pairs, one per segment, labels a 0/1
        array aligned with graph.cross_edges (1 = real astrophysical
        coincidence, 0 = accidental/background).

        `batch_size` groups segments into `torch_geometric` `Batch`es for a
        single sparse forward/backward pass each, instead of one Python-level
        call per segment -- the throughput path for training across many
        segments at once. Default (None) puts every segment in one batch per
        epoch (full-batch training), the common case when `examples` is the
        whole training set for one run, not a huge out-of-core dataset.
        Returns the per-epoch loss history.
        """
        datas = [_to_pyg_data(g, labels) for g, labels in examples if g.cross_edges.size]
        if not datas:
            return []
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
                total = total + loss_fn(logits, batch.y.to(self.device))
                n_chunks += 1
            loss = total / n_chunks
            loss.backward()
            opt.step()
            history.append(float(loss.item()))
        return history
