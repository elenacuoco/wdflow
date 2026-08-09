"""Ranking a coincidence by how unlike an accidental one it is.

The network stage has to say which candidates are worth keeping without being
told what a signal looks like. Fitting a classifier on injections would make the
search's own selection depend on the waveform family it was shown, which is what
an un-modelled search exists not to do.

Time slides give the alternative. Shifting one detector against the other by far
more than the light travel time destroys every real coincidence and leaves the
detectors' noise intact, so every candidate built from slid data is accidental
by construction, and there are as many of them as the background needs. A model
fitted to reconstruct only those learns the shape of an accidental coincidence.
What it cannot reconstruct is what does not belong to that population, so the
reconstruction error ranks candidates without any signal ever being shown.

No labels are involved and no injection is read. The same time slides that
measure the false-alarm rate are what the model is fitted on, so the ranking and
its calibration come from one population.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from wdf.analysis.gnn import _IntraMessagePassing, _to_pyg_data


class BackgroundAnomalyScorer(nn.Module):
    """A graph autoencoder fitted on accidental coincidences only.

    Intra-detector message passing gives each event an embedding that carries
    its neighbourhood, as in `GNNCoincidenceScorer`; the two embeddings of a
    candidate pair are then asked to reproduce the pair's own edge features. On
    the population it was fitted to, they can. The score is how badly they fail.
    """

    def __init__(self, node_dim: int, cross_edge_dim: int, hidden: int = 16,
                 seed: int = 0, device: str | None = None):
        """
        :type node_dim: int
        :param node_dim: width of a node's feature vector.
        :type cross_edge_dim: int
        :param cross_edge_dim: number of features on a candidate edge.
        :type hidden: int
        :param hidden: width of the hidden representation.
        :type seed: int
        :param seed: torch seed, for a reproducible initialisation.
        :type device: str or None
        :param device: torch device; CUDA when available if None.
        """
        super().__init__()
        torch.manual_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.intra_mp = _IntraMessagePassing(hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, cross_edge_dim))
        # The scaling is part of the fitted model. Measured per graph instead,
        # the foreground and the background would be read on two different
        # scales, which is the comparison this exists to make.
        self.register_buffer("feature_mean", torch.zeros(node_dim))
        self.register_buffer("feature_scale", torch.ones(node_dim))
        self.register_buffer("edge_mean", torch.zeros(cross_edge_dim))
        self.register_buffer("edge_scale", torch.ones(cross_edge_dim))
        self.to(self.device)

    def _residuals(self, data) -> torch.Tensor:
        """Per-edge reconstruction error of the edge features."""
        x = (data.x.to(self.device) - self.feature_mean) / self.feature_scale
        h = self.encoder(x)
        h = self.intra_mp(h, data.edge_index.to(self.device),
                          data.edge_attr.to(self.device))
        if data.cross_edge_index.numel() == 0:
            return torch.zeros(0, device=self.device)
        i, j = (data.cross_edge_index[0].to(self.device),
                data.cross_edge_index[1].to(self.device))
        target = (data.cross_edge_attr.to(self.device) - self.edge_mean) / self.edge_scale
        # The pair is unordered --- which detector is named first is an accident
        # of the loop that built the graph --- so both orders are reconstructed
        # and the model cannot learn to rely on the order.
        forward = self.decoder(torch.cat([h[i], h[j]], dim=1))
        backward = self.decoder(torch.cat([h[j], h[i]], dim=1))
        return (0.5 * ((forward - target) ** 2 + (backward - target) ** 2)).mean(dim=1)

    def fit(self, graphs, epochs: int = 100, lr: float = 1e-2):
        """Fit on background graphs only.

        :param graphs: iterable of `TriggerGraph` built from time-slid events.
        :type epochs: int
        :param epochs: passes over the data.
        :type lr: float
        :param lr: learning rate.
        :return: list[float] -- the loss at each epoch.
        :raises ValueError: if the graphs carry no candidate edges.
        """
        data = [_to_pyg_data(graph) for graph in graphs
                if graph.cross_edges.shape[0] > 0]
        if not data:
            raise ValueError("no accidental coincidences to fit on; the slides "
                             "produced no candidate edges")

        nodes = torch.cat([d.x for d in data]).to(self.device)
        self.feature_mean.copy_(nodes.mean(dim=0))
        self.feature_scale.copy_(nodes.std(dim=0).clamp_min(1e-6))
        edges = torch.cat([d.cross_edge_attr for d in data]).to(self.device)
        self.edge_mean.copy_(edges.mean(dim=0))
        self.edge_scale.copy_(edges.std(dim=0).clamp_min(1e-6))

        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        history = []
        self.train()
        for _ in range(int(epochs)):
            optimiser.zero_grad()
            loss = torch.cat([self._residuals(d) for d in data]).mean()
            loss.backward()
            optimiser.step()
            history.append(float(loss.detach().cpu()))
        return history

    @torch.no_grad()
    def score(self, graph) -> pd.DataFrame:
        """The candidate table with `anomaly_score` added.

        The score is the reconstruction error, so it grows with how unlike an
        accidental coincidence the candidate is. It is ranked and thresholded
        exactly as any other statistic, against the same background.

        :param graph: a `TriggerGraph`.
        :return: pandas.DataFrame -- one row per candidate edge.
        """
        self.eval()
        table = graph.candidate_table()
        if not len(table):
            return table
        residual = self._residuals(_to_pyg_data(graph)).cpu().numpy()
        return table.assign(anomaly_score=np.asarray(residual, dtype=float))
