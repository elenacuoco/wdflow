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
        # One row per cross edge, flattened over band and lag. The width is
        # stated rather than inferred: a graph with no edge and no lag has no
        # element for `-1` to solve for, and reshaping it is refused.
        cross_edge_profile=torch.from_numpy(graph.cross_edge_profiles.reshape(
            len(graph.cross_edges),
            int(np.prod(graph.cross_edge_profiles.shape[1:])))),
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
                 device: str | None = None, cross_edge_dim: int = 2, profile_dim: int = 0):
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
        :param cross_edge_dim: number of scalar features on a candidate edge.
        :type profile_dim: int
        :param profile_dim: flattened width of the signed correlation profile; zero
            loads legacy models without this feature.
        """
        super().__init__()
        torch.manual_seed(seed)
        self.device = usable_device(device)
        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.intra_mp = _IntraMessagePassing(hidden)
        self.profile_dim = int(profile_dim)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 2 + cross_edge_dim + self.profile_dim, hidden),
            nn.ReLU(), nn.Linear(hidden, 1)
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
        # Two statements would hold two copies of the node features at once;
        # on a slid background this matrix is gigabytes and the call is made
        # once per shift. The subtraction already owns its result, so the
        # scaling is applied to it in place and `data.x` is left untouched.
        x = (x - self.feature_mean).div_(self.feature_scale)
        h = self.encoder(x)
        h = self.intra_mp(h, data.edge_index.to(self.device), data.edge_attr.to(self.device))
        if data.cross_edge_index.numel() == 0:
            return torch.zeros(0, device=self.device)
        i, j = data.cross_edge_index[0].to(self.device), data.cross_edge_index[1].to(self.device)
        ef = data.cross_edge_attr.to(self.device)
        if self.profile_dim:
            profile = data.cross_edge_profile.to(self.device)
            if profile.shape[1] != self.profile_dim:
                raise ValueError(f"correlation profile width {profile.shape[1]} "
                                 f"does not match model {self.profile_dim}")
            ef = torch.cat([ef, profile], dim=1)
        return self.edge_head(torch.cat([h[i], h[j], ef], dim=1)).squeeze(-1)

    def edge_logits(self, graph: TriggerGraph) -> "torch.Tensor":
        return self._edge_logits_from_data(_to_pyg_data(graph))

    def save(self, path: str) -> None:
        """Write the fitted model, with the widths needed to rebuild it.

        The feature scaling is written with the weights because it is part of
        the fitted model. A scaling re-measured on whatever graph the model is
        later shown would have it read two populations on two different scales,
        which is the comparison it exists to make.

        :type path: str
        :param path: file to write.
        :return: None
        """
        hidden = int(self.encoder[0].out_features)
        torch.save({
            "state_dict": self.state_dict(),
            "node_dim": int(self.encoder[0].in_features),
            "hidden": hidden,
            "cross_edge_dim": int(self.edge_head[0].in_features - 2 * hidden - self.profile_dim),
            "profile_dim": self.profile_dim,
        }, path)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        """Rebuild a model written by `save`, in evaluation mode.

        :type path: str
        :param path: file to read.
        :type device: str or None
        :param device: torch device; CUDA when available if None.
        :return: GNNCoincidenceScorer -- ready to `score`. Fitting it further
            continues from these weights and is no longer a fresh model.
        :raises KeyError: if the file was not written by `save`.
        """
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(node_dim=blob["node_dim"], hidden=blob["hidden"],
                    cross_edge_dim=blob["cross_edge_dim"],
                    profile_dim=blob.get("profile_dim", 0), device=device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model

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


class _EdgeMessagePassing(MessagePassing):
    """One round of message passing over an edge set of any feature width."""

    def __init__(self, hidden: int, edge_dim: int):
        super().__init__(aggr="mean")
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden + edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.update_mlp = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())

    def forward(self, h, edge_index, edge_attr):
        return self.update_mlp(
            torch.cat([h, self.propagate(edge_index, h=h, edge_attr=edge_attr)], dim=1))

    def message(self, h_j, edge_attr):
        return self.message_mlp(torch.cat([h_j, edge_attr], dim=1))


class DetectorEdgeScorer(nn.Module):
    """The detector stage: which of a detector's admissible edges join one transient.

    The graph's edges are what geometry allows; connected components over all
    of them chain noise triggers together, because triggers close in time and
    band are common in noise. This decides which of those edges survive, and
    the components of the survivors are the detector's events.

    The morphology reaches the model: a node carries its trigger's wavegram on
    a band-by-time grid, so what is compared is how two triggers look in the
    plane and not only how close they are.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 32,
                 layers: int = 2, seed: int = 0, device: str | None = None):
        """
        :type node_dim: int
        :param node_dim: width of `DetectorGraph.node_features`.
        :type edge_dim: int
        :param edge_dim: width of `DetectorGraph.edge_features`.
        :type hidden: int
        :param hidden: width of the hidden representation.
        :type layers: int
        :param layers: rounds of message passing.
        :type seed: int
        :param seed: torch seed, for a reproducible initialisation.
        :type device: str or None
        :param device: torch device; CUDA when available if None.
        """
        super().__init__()
        torch.manual_seed(seed)
        self.device = usable_device(device)
        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.rounds = nn.ModuleList(
            [_EdgeMessagePassing(hidden, edge_dim) for _ in range(layers)])
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 2 + edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # Carried with the model, so that the stretch it was fitted on and any
        # stretch it later scores are read on one scale.
        self.register_buffer("feature_mean", torch.zeros(node_dim))
        self.register_buffer("feature_scale", torch.ones(node_dim))
        self.to(self.device)

    def upload(self, graph):
        """Move a graph onto the device once, for reuse across epochs.

        Rebuilding the tensors inside the training loop copies the whole node
        feature matrix from host to device on every step, which at realistic
        graph sizes is most of the training time and none of the arithmetic.

        :param graph: a `wdf.analysis.detector_graph.DetectorGraph`.
        :return: tuple -- the device tensors `_logits` reads.
        """
        x = torch.as_tensor(graph.node_features, dtype=torch.float32, device=self.device)
        edge_attr = torch.as_tensor(graph.edge_features, dtype=torch.float32,
                                    device=self.device)
        if not len(graph.edges):
            return x, edge_attr, None, None, None
        edges = torch.as_tensor(np.ascontiguousarray(graph.edges.T),
                                dtype=torch.long, device=self.device)
        # Messages travel both ways: the relation between two triggers is
        # symmetric, and a directed edge set would make the first of a pair
        # blind to the second.
        both = torch.cat([edges, edges.flip(0)], dim=1)
        both_attr = torch.cat([edge_attr, edge_attr], dim=0)
        return x, edge_attr, edges, both, both_attr

    def _logits(self, uploaded):
        x, edge_attr, edges, both, both_attr = uploaded
        if edges is None:
            return torch.zeros(0, device=self.device)
        h = self.encoder((x - self.feature_mean) / self.feature_scale)
        for round_ in self.rounds:
            h = round_(h, both, both_attr)
        return self.edge_head(
            torch.cat([h[edges[0]], h[edges[1]], edge_attr], dim=1)).squeeze(-1)

    def score(self, graph) -> pd.DataFrame:
        """The model's opinion on each admissible edge.

        :param graph: a `wdf.analysis.detector_graph.DetectorGraph`.
        :return: pandas.DataFrame -- the edge table with `edge_logit` and
            `edge_score`, the probability that the two triggers are one
            transient.
        """
        with torch.no_grad():
            logits = self._logits(self.upload(graph))
            raw = logits.cpu().numpy() if logits.numel() else np.array([])
            probability = torch.sigmoid(logits).cpu().numpy() if logits.numel() \
                else np.array([])
        table = graph.edge_table()
        table["edge_logit"] = raw
        table["edge_score"] = probability
        return table

    def fit(self, examples: list[tuple], epochs: int = 100, lr: float = 1e-2) -> list[float]:
        """Fit on labelled edges.

        :param examples: `(graph, labels)` or `(graph, labels, mask)` per
            stretch, labels aligned with `graph.edges`, 1 where the two
            triggers belong to one transient.
        :type epochs: int
        :param epochs: gradient steps.
        :type lr: float
        :param lr: learning rate.
        :return: list[float] -- the loss per epoch.
        """
        usable = [e for e in examples if len(e[0].edges)]
        if not usable:
            return []
        features = np.vstack([e[0].node_features for e in usable])
        mean, scale = features.mean(axis=0), features.std(axis=0)
        scale[scale == 0.0] = 1.0
        self.feature_mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        self.feature_scale = torch.as_tensor(scale, dtype=torch.float32, device=self.device)

        # Uploaded once, outside the epoch loop: the graphs do not change while
        # the weights do.
        prepared = []
        for example in usable:
            labels = np.asarray(example[1], dtype=float)
            mask = example[2] if len(example) > 2 else np.ones(len(labels), dtype=bool)
            mask = torch.as_tensor(np.asarray(mask, dtype=bool), device=self.device)
            if not bool(mask.any()):
                continue
            prepared.append((
                self.upload(example[0]),
                torch.as_tensor(labels, dtype=torch.float32, device=self.device),
                mask,
            ))

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        history = []
        for _ in range(epochs):
            opt.zero_grad()
            total, contributing = torch.tensor(0.0, device=self.device), 0
            for uploaded, target, mask in prepared:
                total = total + loss_fn(self._logits(uploaded)[mask], target[mask])
                contributing += 1
            if contributing:
                total.backward()
                opt.step()
            history.append(float(total.detach().cpu()))
        return history


def edge_labels_from_injections(graph, injection_times, tolerance_s: float = 0.5):
    """Label each edge by whether its two triggers share an injection.

    A trigger belongs to the injection whose time falls inside its window,
    widened by `tolerance_s`; an edge is a positive when both belong to the
    same one. This is the ground truth the detector stage can actually be trained on.

    :param graph: a `wdf.analysis.detector_graph.DetectorGraph`.
    :param injection_times: GPS times of the injected signals.
    :type tolerance_s: float
    :param tolerance_s: how far outside its window a trigger may still claim
        an injection, seconds.
    :return: numpy.ndarray -- 1.0 where both triggers share an injection.
    """
    nodes = graph.nodes
    start = nodes["gps"].to_numpy(dtype=float)
    span = nodes["n_coeff"].to_numpy(dtype=float) / np.maximum(
        nodes["fs"].to_numpy(dtype=float), 1e-30)
    times = np.sort(np.asarray(injection_times, dtype=float))

    slot = np.searchsorted(times, start + 0.5 * span)
    owner = np.full(len(nodes), -1)
    for offset in (-1, 0):
        candidate = np.clip(slot + offset, 0, max(len(times) - 1, 0))
        if not len(times):
            break
        inside = ((times[candidate] >= start - tolerance_s)
                  & (times[candidate] <= start + span + tolerance_s))
        owner = np.where(inside & (owner < 0), candidate, owner)

    if not len(graph.edges):
        return np.zeros(0)
    i, j = graph.edges[:, 0], graph.edges[:, 1]
    return ((owner[i] >= 0) & (owner[i] == owner[j])).astype(float)


def usable_device(requested=None):
    """The device to run on, preferring the GPU only when it really works.

    `torch.cuda.is_available()` answers whether a driver and a device were
    found, not whether this build can use them: a runtime built against one
    CUDA version, a driver from another, or a device already full will report
    available and then raise on the first allocation. Since every stage here
    runs on the processor in seconds to minutes, falling back is always better
    than failing.

    :param requested: a device name to use as given, or None to choose.
    :return: str -- the device name.
    """
    if requested is not None:
        return str(requested)
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda") + 1
    except Exception as reason:  # a broken GPU is not a reason to stop
        import warnings

        warnings.warn(f"CUDA reported available but is not usable ({reason}); "
                      "running on the processor", RuntimeWarning)
        return "cpu"
    return "cuda"
