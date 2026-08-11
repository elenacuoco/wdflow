"""Level one: the wavelet pixels of one detector as a graph over (t, f, scale).

A node is one surviving wavelet coefficient -- a time-frequency tile, at the
window length it was found at. Edges join tiles that could belong to the same
transient: neighbours in time within the same band, and, when a run is
configured at more than one window length, tiles of different lengths whose
cells cover the same region.

The length is carried as a coordinate rather than resolved away, so that tiles
found at different lengths are related by the same rule that relates tiles found
at one, instead of by a reconciliation between separate clusterings.

Assembling the connected components of this graph gives one detector's events,
which are the nodes of level two, the inter-detector network graph in
`wdf.analysis.network_graph`. The same shape serves either way of deciding
which edges survive: keeping every admissible edge is the deterministic
clustering, and scoring them is the learned one -- both start from the same
graph, which is what makes them comparable.

Only the pairs inside the tolerance are ever formed, by searching a sorted time
axis. A dense adjacency matrix asks the same question in O(n^2) memory, which a
segment's pixel cloud exhausts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wdf.analysis.pairs import neighbour_pairs
from wdf.analysis.robust_events import EPS, _UnionFind

PIXEL_EDGE_FEATURES = [
    "time_gap", "frequency_overlap", "log_scale_ratio",
    "significance_min", "significance_max", "cross_scale",
]


CLUSTER_COLUMNS = [
    "cluster_id", "ifo", "gps", "gpsStart", "gpsCentroid", "tSpread", "gpsPeak",
    "duration", "freqMin", "freqMean", "freqMax", "EnWDF", "sigma", "snrPeak",
    "significance", "energy", "n_pixels", "n_scales", "scale_best", "n_coeff", "fs",
]


@dataclass
class PixelGraphConfig:
    """When two tiles may belong to the same transient.

    :param time_tolerance: largest gap between two tiles' time spans, as a
        fraction of their mean width. Zero joins only tiles that touch or
        overlap; one allows a gap as wide as the tiles themselves.
    :param minimum_significance: tiles below this are not nodes at all. Reading
        it from the calibrated significance rather than from a raw amplitude is
        what lets one threshold serve every window length and every band.
    """

    time_tolerance: float = 1.0
    minimum_significance: float = 0.0


class PixelGraph:
    """Tiles as nodes, possible same-transient relations as edges.

    :param nodes: the pixel cloud, one row per tile.
    :param edges: (n_edges, 2) integer indices into `nodes`.
    :param edge_features: (n_edges, len(PIXEL_EDGE_FEATURES)).
    """

    def __init__(self, nodes: pd.DataFrame, edges: np.ndarray,
                 edge_features: np.ndarray):
        self.nodes = nodes
        self.edges = edges
        self.edge_features = edge_features

    def edge_table(self) -> pd.DataFrame:
        """The edges and their features as a table.

        :return: pandas.DataFrame -- one row per edge.
        """
        table = pd.DataFrame(dict(node_i=self.edges[:, 0], node_j=self.edges[:, 1]))
        for column, name in enumerate(PIXEL_EDGE_FEATURES):
            table[name] = self.edge_features[:, column]
        return table

    def components(self, keep=None) -> np.ndarray:
        """Connected-component label per node, over the surviving edges.

        :param keep: boolean mask over the edges; default, every edge.
        :return: numpy.ndarray -- one label per node.
        """
        union = _UnionFind(len(self.nodes))
        edges = self.edges if keep is None else self.edges[np.asarray(keep, dtype=bool)]
        for i, j in edges:
            union.union(int(i), int(j))
        roots = np.array([union.find(i) for i in range(len(self.nodes))])
        _, labels = np.unique(roots, return_inverse=True)
        return labels


def build_pixel_graph(pixels: pd.DataFrame,
                      significance=None,
                      config: PixelGraphConfig | None = None) -> PixelGraph:
    """The (t, f, scale) graph of one detector's tiles.

    :type pixels: pandas.DataFrame
    :param pixels: a pixel cloud, as `wdf.analysis.scale.pixel_cloud` returns.
    :param significance: each tile's calibrated significance, or None to rank
        the tiles on their energy alone.
    :type config: PixelGraphConfig | None
    :param config: when two tiles may belong to the same transient.
    :return: PixelGraph
    """
    config = PixelGraphConfig() if config is None else config
    nodes = pixels.reset_index(drop=True)

    if significance is None:
        significance = np.log1p(nodes["energy"].to_numpy(dtype=float)) if len(nodes) \
            else np.zeros(0)
    significance = np.asarray(significance, dtype=float)
    significance = np.where(np.isfinite(significance), significance, 0.0)

    keep = significance >= config.minimum_significance
    nodes = nodes[keep].reset_index(drop=True)
    significance = significance[keep]
    if nodes.empty:
        return PixelGraph(nodes, np.zeros((0, 2), dtype=int),
                          np.zeros((0, len(PIXEL_EDGE_FEATURES))))

    t_lo = nodes["t_lo"].to_numpy(dtype=float)
    order = np.argsort(t_lo, kind="mergesort")
    nodes = nodes.iloc[order].reset_index(drop=True)
    significance = significance[order]

    t_lo = nodes["t_lo"].to_numpy(dtype=float)
    t_hi = nodes["t_hi"].to_numpy(dtype=float)
    f_lo = nodes["f_lo"].to_numpy(dtype=float)
    f_hi = nodes["f_hi"].to_numpy(dtype=float)
    scale = nodes["scale"].to_numpy(dtype=float)
    width = t_hi - t_lo

    # The widest tile plus its own tolerance is how far apart two tiles' starts
    # can be and still touch, which bounds the search along the time axis.
    reach = float((width * (1.0 + config.time_tolerance)).max())

    edges, features = [], []
    for left, right in neighbour_pairs(t_lo, reach):
        gap = np.maximum(t_lo[right] - t_hi[left], t_lo[left] - t_hi[right])
        allowed = config.time_tolerance * 0.5 * (width[left] + width[right])
        band = (f_lo[left] <= f_hi[right]) & (f_lo[right] <= f_hi[left])
        join = (gap <= allowed) & band
        if not join.any():
            continue
        i, j = left[join], right[join]
        shared = (np.minimum(f_hi[i], f_hi[j]) - np.maximum(f_lo[i], f_lo[j]))
        narrower = np.maximum(np.minimum(f_hi[i] - f_lo[i], f_hi[j] - f_lo[j]), EPS)
        edges.append(np.column_stack([i, j]))
        features.append(np.column_stack([
            gap[join],
            np.clip(shared / narrower, 0.0, 1.0),
            np.log(scale[i] / scale[j]),
            np.minimum(significance[i], significance[j]),
            np.maximum(significance[i], significance[j]),
            (scale[i] != scale[j]).astype(float),
        ]))

    if not edges:
        return PixelGraph(nodes, np.zeros((0, 2), dtype=int),
                          np.zeros((0, len(PIXEL_EDGE_FEATURES))))
    return PixelGraph(nodes, np.concatenate(edges),
                      np.concatenate(features))


def cluster_events(graph: PixelGraph, significance=None, labels=None) -> pd.DataFrame:
    """One detector's events, from the connected components of its pixel graph.

    This is the step from level one to level two: what comes out is the node
    set of the inter-detector network graph.

    Two quantities are deliberately not the same sum. The cluster's
    significance is the largest of its pixels', which is a maximum over
    correlated searches and so has to be calibrated on the background in its
    own right rather than read from one scale's distribution. Its energy is
    summed over one window length only -- the one carrying the largest
    significance -- because the window lengths all describe the same strain and
    summing across them would count the transient's energy several times.

    :type graph: PixelGraph
    :param graph: the detector's pixel graph.
    :param significance: each node's calibrated significance, or None to rank
        the nodes on their energy alone.
    :param labels: component label per node, or None to take every edge.
    :return: pandas.DataFrame -- one row per event, with `CLUSTER_COLUMNS`.
    """
    nodes = graph.nodes
    if nodes.empty:
        return pd.DataFrame(columns=CLUSTER_COLUMNS)

    labels = graph.components() if labels is None else np.asarray(labels)
    if significance is None:
        significance = np.log1p(nodes["energy"].to_numpy(dtype=float))
    significance = np.asarray(significance, dtype=float)
    significance = np.where(np.isfinite(significance), significance, 0.0)

    frame = nodes.assign(cluster_id=labels, significance=significance)
    rows = []
    for cluster_id, group in frame.groupby("cluster_id", sort=True):
        loudest = group.loc[group["significance"].idxmax()]
        scale_best = int(loudest["scale"])
        best = group[group["scale"] == scale_best]

        energy = float(best["energy"].sum())
        sigma = best["sigma"].to_numpy(dtype=float)
        sigma = sigma[np.isfinite(sigma) & (sigma > 0.0)]
        noise = float(sigma.mean()) if sigma.size else np.nan

        weight = best["energy"].to_numpy(dtype=float)
        centre = 0.5 * (best["t_lo"].to_numpy(dtype=float)
                        + best["t_hi"].to_numpy(dtype=float))
        total = max(weight.sum(), EPS)
        centroid = float((centre * weight).sum() / total)
        spread = float(np.sqrt(max(((centre - centroid) ** 2 * weight).sum() / total, 0.0)))

        f_lo = group["f_lo"].to_numpy(dtype=float)
        f_hi = group["f_hi"].to_numpy(dtype=float)
        band = np.sqrt(np.maximum(best["f_lo"].to_numpy(dtype=float), EPS)
                       * best["f_hi"].to_numpy(dtype=float))
        start = float(group["t_lo"].min())

        rows.append(dict(
            cluster_id=int(cluster_id),
            ifo=loudest.get("ifo", ""),
            gps=start,
            gpsStart=start,
            gpsCentroid=centroid,
            tSpread=spread,
            gpsPeak=float(0.5 * (loudest["t_lo"] + loudest["t_hi"])),
            duration=float(group["t_hi"].max() - start),
            freqMin=float(f_lo.min()),
            freqMean=float((band * weight).sum() / total),
            freqMax=float(f_hi.max()),
            EnWDF=float(np.sqrt(energy) / noise) if noise == noise else np.nan,
            sigma=noise,
            snrPeak=float(np.sqrt(loudest["energy"]) / noise) if noise == noise else np.nan,
            significance=float(loudest["significance"]),
            energy=energy,
            n_pixels=int(len(group)),
            n_scales=int(group["scale"].nunique()),
            scale_best=scale_best,
            n_coeff=scale_best,
            fs=float(loudest["fs"]),
        ))
    return pd.DataFrame(rows, columns=CLUSTER_COLUMNS)
