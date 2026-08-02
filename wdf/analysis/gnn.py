"""Graph-neural-network coincidence scorer, run ALONGSIDE (not replacing)
CoincidenceFinder's classical time-window method -- the two are meant to be
compared directly in the notebook, per explicit instruction, not to replace
one another.

Graphs here are tiny (per segment: tens of clustered events per detector, at
most a few hundred candidate edges), so this deliberately avoids a full graph
library (torch_geometric) in favor of a minimal hand-rolled message-passing
layer in plain PyTorch -- one dependency (torch, via the `gnn` extra), not
two.

Training positives are real catalogued events in the analyzed segment;
negatives are accidental/background cross-detector edges (including
time-slide background from significance.BackgroundEstimator). With only a
handful of real positive events even in one long continuous segment, `fit`
here is explicitly a proof-of-concept calibration, not a production training
run -- see the project notebook for the small-N caveat and the
synthetic-injection extension (bilby/pycbc) noted as future work, not
required for v1.

Requires `pip install -e ".[gnn]"` (torch). Importing this module without
torch installed raises ImportError with that instruction -- the rest of
wdfLib works without torch.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "wdfLib.gnn requires torch. Install with: pip install -e '.[gnn]'"
    ) from _e

NODE_FEATURE_COLUMNS = ["snrMax", "freqMean", "freqMax", "duration", "n_triggers"]


class TriggerGraph:
    """Node = clustered per-IFO event. Intra-IFO edges = temporally-close
    same-detector clusters (local-density context). Cross-IFO edges =
    candidate coincidence pairs across detectors (both true and accidental,
    deliberately, since the GNN needs negative examples to learn from).
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


class _EdgeMLP(nn.Module):
    def __init__(self, in_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, x):
        return self.net(x)


class GNNCoincidenceScorer(nn.Module):
    """One round of intra-detector message passing (each cluster's
    embedding absorbs its temporally-close same-detector neighbors, i.e.
    'is this cluster sitting in a noisy neighborhood?'), then an
    edge-classification head scores cross-detector candidate edges."""

    def __init__(self, node_dim: int, hidden: int = 16, seed: int = 0, device: str | None = None):
        super().__init__()
        torch.manual_seed(seed)
        # Graphs here are tiny (see module docstring), so a GPU rarely beats CPU once
        # you count host<->device transfer overhead -- but if one's available, use it
        # rather than second-guess every caller's hardware.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.message = _EdgeMLP(hidden + 1, hidden)  # neighbor embedding + dt
        self.update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.to(self.device)

    def _propagate(self, graph: TriggerGraph) -> "torch.Tensor":
        x = torch.from_numpy(graph.node_features).to(self.device)
        h = self.encoder(x)
        n = h.shape[0]
        agg = torch.zeros_like(h)
        count = torch.zeros(n, 1, device=self.device)
        if graph.intra_edges.size:
            neighbor = torch.from_numpy(graph.intra_edges[:, 1]).long().to(self.device)
            receiver = torch.from_numpy(graph.intra_edges[:, 0]).long().to(self.device)
            edge_feat = torch.from_numpy(graph.intra_edge_features).to(self.device)
            msg = self.message(torch.cat([h[neighbor], edge_feat], dim=1))
            agg.index_add_(0, receiver, msg)
            count.index_add_(0, receiver, torch.ones(len(receiver), 1, device=self.device))
        agg = agg / count.clamp(min=1)
        return self.update(torch.cat([h, agg], dim=1))

    def edge_logits(self, graph: TriggerGraph) -> "torch.Tensor":
        h = self._propagate(graph)
        if graph.cross_edges.size == 0:
            return torch.zeros(0, device=self.device)
        i = torch.from_numpy(graph.cross_edges[:, 0]).long().to(self.device)
        j = torch.from_numpy(graph.cross_edges[:, 1]).long().to(self.device)
        ef = torch.from_numpy(graph.cross_edge_features).to(self.device)
        return self.edge_head(torch.cat([h[i], h[j], ef], dim=1)).squeeze(-1)

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
    ) -> list[float]:
        """examples: (graph, labels) pairs, labels a 0/1 array aligned with
        graph.cross_edges (1 = real astrophysical coincidence, 0 =
        accidental/background). Returns the per-epoch loss history."""
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        history = []
        for _ in range(epochs):
            opt.zero_grad()
            total, n_batches = torch.tensor(0.0, device=self.device), 0
            for graph, labels in examples:
                if graph.cross_edges.size == 0:
                    continue
                logits = self.edge_logits(graph)
                y = torch.from_numpy(labels.astype(np.float32)).to(self.device)
                total = total + loss_fn(logits, y)
                n_batches += 1
            if n_batches == 0:
                break
            loss = total / n_batches
            loss.backward()
            opt.step()
            history.append(float(loss.item()))
        return history
