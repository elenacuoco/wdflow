"""Level one: one detector's triggers, across window lengths, as a graph.

A node is one WDF trigger -- one analysis window's surviving coefficients, at
the window length it was found at. Edges join triggers that could belong to the
same transient: neighbours in time at the same window length, and triggers of
different window lengths covering the same region of the time-frequency plane.
The connected components are the detector's events, which are the nodes of
level two, the inter-detector network graph in `wdf.analysis.network_graph`.

The trigger is the right node for this. The labels that train it exist there
and only there -- an edge is a positive when both triggers belong to the same
injection, which the catalogue states about triggers, not about tiles. The
statistic is defined there too: the Donoho-Johnstone threshold depends on how
many coefficients the window holds, so a window length's background
distribution, and the significance that makes two window lengths comparable,
are properties of a trigger. And the graph is rebuilt once per time slide, so
the node count is paid a hundred times over.

None of the tile-level information is discarded: it becomes the node's own
feature vector. A node carries its wavegram, so what reaches the model is not
"a trigger at 80 Hz with EnWDF 8" but how the trigger looks in the plane.
Wavegram rows are indexed by absolute frequency band rather than by octave
level, because a longer window extends the dyadic ladder downward rather than
subdividing it: the same physical band is then the same row at every window
length, and the feature vector has one width and one meaning across scales.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wdf.analysis.metaparameters import energy_quantile
from wdf.analysis.pairs import neighbour_pairs
from wdf.analysis.robust_events import EPS, _UnionFind
from wdf.analysis.wavelets import coeff_freq_bands, coeff_time_bounds

TRIGGER_EDGE_FEATURES = [
    "dt_s", "time_overlap", "frequency_overlap", "log_scale_ratio",
    "log_energy_ratio", "wavegram_similarity", "significance_min",
    "significance_max", "cross_scale",
]

DETECTOR_EVENT_COLUMNS = [
    "cluster_id", "ifo", "gps", "gpsStart", "gpsCentroid", "tSpread", "gpsPeak",
    "duration", "duration90", "freqMin", "freqMean", "freqMax",
    "freqQ05", "freqQ95", "EnWDF", "sigma", "snrPeak",
    "significance", "n_triggers", "n_scales", "scale_best", "n_coeff", "fs",
]

WAVEGRAM_TIME_BINS = 16


@dataclass
class DetectorGraphConfig:
    """When two triggers may belong to the same transient.

    :param time_tolerance: largest gap between two triggers' time supports, as
        a fraction of their mean window span. Zero joins only triggers that
        touch or overlap; one allows a gap as wide as a window.
    :param minimum_frequency_overlap: least shared fraction of the narrower band.
    :param minimum_significance: triggers below this are not nodes at all.
    :param wavegram_time_bins: time bins per band in a node's wavegram.
    """

    time_tolerance: float = 1.0
    minimum_frequency_overlap: float = 0.0
    minimum_significance: float = 0.0
    wavegram_time_bins: int = WAVEGRAM_TIME_BINS


def band_grid(scales, fs: float) -> np.ndarray:
    """The frequency bands reached by any of these window lengths.

    A longer window extends the dyadic ladder downward rather than subdividing
    it, so the bands of several window lengths share their edges exactly and
    their union is itself a ladder.

    :param scales: the window lengths, in samples.
    :type fs: float
    :param fs: sampling frequency, Hz.
    :return: numpy.ndarray -- (n_bands, 2) of (f_lo, f_hi), ascending.
    """
    edges = set()
    for scale in np.unique(np.asarray(scales, dtype=int)):
        f_lo, f_hi = coeff_freq_bands(int(scale), fs)
        edges.update(zip(np.round(f_lo, 9), np.round(f_hi, 9)))
    return np.array(sorted(edges))


def trigger_wavegrams(triggers: pd.DataFrame, bands: np.ndarray,
                      time_bins: int) -> np.ndarray:
    """Each trigger's coefficients on a band-by-time grid of one fixed shape.

    :type triggers: pandas.DataFrame
    :param triggers: triggers carrying `n_coeff`, `fs` and the coefficients.
    :type bands: numpy.ndarray
    :param bands: (n_bands, 2) band edges, as `band_grid` returns.
    :type time_bins: int
    :param time_bins: time bins per band, spanning each trigger's own window.
    :return: numpy.ndarray -- (n_triggers, n_bands * time_bins).
    """
    grid = np.zeros((len(triggers), len(bands), time_bins))
    if triggers.empty:
        return grid.reshape(len(triggers), -1)

    band_of = {(round(lo, 9), round(hi, 9)): row
               for row, (lo, hi) in enumerate(bands)}
    positions = np.arange(len(triggers))
    for (n_coeff, fs), group in triggers.groupby(["n_coeff", "fs"], sort=False):
        n_coeff, fs = int(n_coeff), float(fs)
        t_lo, t_hi = coeff_time_bounds(n_coeff, fs)
        f_lo, f_hi = coeff_freq_bands(n_coeff, fs)
        span = n_coeff / fs
        row_of = np.array([band_of.get((round(lo, 9), round(hi, 9)), -1)
                           for lo, hi in zip(f_lo, f_hi)])
        column_of = np.clip(
            (0.5 * (t_lo + t_hi) / span * time_bins).astype(int), 0, time_bins - 1)

        where = positions[triggers.index.get_indexer(group.index)]
        for slot, (_, trigger) in zip(where, group.iterrows()):
            index = np.asarray(trigger["wt_index"], dtype=int)
            value = np.abs(np.asarray(trigger["wt_value"], dtype=float))
            keep = row_of[index] >= 0
            np.add.at(grid[slot], (row_of[index[keep]], column_of[index[keep]]),
                      value[keep])
    return grid.reshape(len(triggers), -1)


class DetectorGraph:
    """Triggers as nodes, possible same-transient relations as edges.

    :param nodes: the triggers, one row each.
    :param node_features: (n_nodes, n_features), the wavegram and the scalars.
    :param edges: (n_edges, 2) integer indices into `nodes`.
    :param edge_features: (n_edges, len(TRIGGER_EDGE_FEATURES)).
    """

    def __init__(self, nodes, node_features, edges, edge_features):
        self.nodes = nodes
        self.node_features = node_features
        self.edges = edges
        self.edge_features = edge_features

    def edge_table(self) -> pd.DataFrame:
        """The edges and their features as a table.

        :return: pandas.DataFrame -- one row per edge.
        """
        table = pd.DataFrame(dict(node_i=self.edges[:, 0], node_j=self.edges[:, 1]))
        for column, name in enumerate(TRIGGER_EDGE_FEATURES):
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


def build_detector_graph(triggers: pd.DataFrame,
                         significance=None,
                         config: DetectorGraphConfig | None = None) -> DetectorGraph:
    """One detector's level-one graph, over all its window lengths at once.

    :type triggers: pandas.DataFrame
    :param triggers: the detector's triggers, at one or several window lengths.
    :param significance: each trigger's calibrated significance, or None to
        rank the triggers on `EnWDF` alone -- which is only comparable across
        window lengths once calibrated, so pass it whenever more than one
        window length is present.
    :type config: DetectorGraphConfig | None
    :param config: when two triggers may belong to the same transient.
    :return: DetectorGraph
    """
    config = DetectorGraphConfig() if config is None else config
    nodes = triggers.reset_index(drop=True)
    if nodes.empty:
        return DetectorGraph(nodes, np.zeros((0, 1)), np.zeros((0, 2), dtype=int),
                             np.zeros((0, len(TRIGGER_EDGE_FEATURES))))

    if significance is None:
        significance = np.log1p(nodes["EnWDF"].to_numpy(dtype=float))
    significance = np.asarray(significance, dtype=float)
    significance = np.where(np.isfinite(significance), significance, 0.0)

    keep = significance >= config.minimum_significance
    nodes = nodes[keep].reset_index(drop=True)
    significance = significance[keep]
    if nodes.empty:
        return DetectorGraph(nodes, np.zeros((0, 1)), np.zeros((0, 2), dtype=int),
                             np.zeros((0, len(TRIGGER_EDGE_FEATURES))))

    start = nodes["gps"].to_numpy(dtype=float)
    order = np.argsort(start, kind="mergesort")
    nodes = nodes.iloc[order].reset_index(drop=True)
    significance = significance[order]

    scale = nodes["n_coeff"].to_numpy(dtype=float)
    fs = nodes["fs"].to_numpy(dtype=float)
    start = nodes["gps"].to_numpy(dtype=float)
    span = scale / np.maximum(fs, EPS)
    end = start + span
    f_lo = nodes["freqMin"].to_numpy(dtype=float)
    f_hi = nodes["freqMax"].to_numpy(dtype=float)
    energy = np.maximum(nodes["EnWDF"].to_numpy(dtype=float) ** 2, EPS)

    bands = band_grid(scale, float(fs[0]))
    wavegrams = trigger_wavegrams(nodes, bands, config.wavegram_time_bins)
    norms = np.linalg.norm(wavegrams, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    shapes = wavegrams / norms

    scale_columns = np.unique(scale)
    node_features = np.hstack([
        np.log1p(wavegrams),
        (scale[:, None] == scale_columns[None, :]).astype(float),
        np.column_stack([significance, np.log1p(energy), span, f_hi - f_lo]),
    ]).astype(np.float32)

    reach = float((span * (1.0 + config.time_tolerance)).max())
    edges, features = [], []
    for left, right in neighbour_pairs(start, reach):
        gap = np.maximum(start[right] - end[left], start[left] - end[right])
        allowed = config.time_tolerance * 0.5 * (span[left] + span[right])
        shared = np.minimum(f_hi[left], f_hi[right]) - np.maximum(f_lo[left], f_lo[right])
        narrower = np.maximum(np.minimum(f_hi[left] - f_lo[left],
                                         f_hi[right] - f_lo[right]), EPS)
        overlap = np.clip(shared / narrower, 0.0, 1.0)
        join = (gap <= allowed) & (overlap >= config.minimum_frequency_overlap)
        if not join.any():
            continue
        i, j = left[join], right[join]
        covered = np.minimum(end[i], end[j]) - np.maximum(start[i], start[j])
        edges.append(np.column_stack([i, j]))
        features.append(np.column_stack([
            start[i] - start[j],
            np.clip(covered / np.maximum(np.minimum(span[i], span[j]), EPS), 0.0, 1.0),
            overlap[join],
            np.log(scale[i] / scale[j]),
            np.log(energy[i] / energy[j]),
            np.einsum("ij,ij->i", shapes[i], shapes[j]),
            np.minimum(significance[i], significance[j]),
            np.maximum(significance[i], significance[j]),
            (scale[i] != scale[j]).astype(float),
        ]))

    if not edges:
        return DetectorGraph(nodes, node_features, np.zeros((0, 2), dtype=int),
                             np.zeros((0, len(TRIGGER_EDGE_FEATURES))))
    return DetectorGraph(nodes, node_features, np.concatenate(edges),
                         np.concatenate(features))


def detector_events(graph: DetectorGraph, significance=None,
                    labels=None) -> pd.DataFrame:
    """The detector's events, from the connected components of its graph.

    This is the step from level one to level two: the result is the node set of
    the inter-detector network graph, in the schema that graph reads.

    The event's statistic is that of its loudest member rather than a sum over
    members. Overlapping windows and several window lengths all describe the
    same strain, so summing would count one transient's energy several times;
    a maximum over correlated searches is a look-elsewhere effect instead, and
    is calibrated on the background rather than asserted.

    :type graph: DetectorGraph
    :param graph: the detector's level-one graph.
    :param significance: each node's significance, or None to use `EnWDF`.
    :param labels: component label per node, or None to take every edge.
    :return: pandas.DataFrame -- one row per event, `DETECTOR_EVENT_COLUMNS`.
    """
    nodes = graph.nodes
    if nodes.empty:
        return pd.DataFrame(columns=DETECTOR_EVENT_COLUMNS)

    labels = graph.components() if labels is None else np.asarray(labels)
    if significance is None:
        significance = np.log1p(nodes["EnWDF"].to_numpy(dtype=float))
    significance = np.asarray(significance, dtype=float)
    significance = np.where(np.isfinite(significance), significance, 0.0)

    frame = nodes.assign(cluster_id=labels, significance=significance)
    span = frame["n_coeff"].to_numpy(dtype=float) / np.maximum(
        frame["fs"].to_numpy(dtype=float), EPS)
    frame = frame.assign(window_end=frame["gps"].to_numpy(dtype=float) + span)

    rows = []
    for cluster_id, group in frame.groupby("cluster_id", sort=True):
        loudest = group.loc[group["significance"].idxmax()]
        weight = np.maximum(group["EnWDF"].to_numpy(dtype=float) ** 2, EPS)
        total = weight.sum()

        centroid_of = group["gpsCentroid"].to_numpy(dtype=float) \
            if "gpsCentroid" in group else group["gps"].to_numpy(dtype=float)
        centroid_of = np.where(np.isfinite(centroid_of), centroid_of,
                               group["gps"].to_numpy(dtype=float))
        centroid = float((centroid_of * weight).sum() / total)
        spread_of = group["tSpread"].to_numpy(dtype=float) \
            if "tSpread" in group else np.zeros(len(group))
        spread_of = np.where(np.isfinite(spread_of), spread_of, 0.0)
        spread = float(np.sqrt(max(
            ((spread_of ** 2 + (centroid_of - centroid) ** 2) * weight).sum() / total,
            0.0)))

        start = float(group["gps"].min())
        noise = group["sigma"].to_numpy(dtype=float)
        noise = noise[np.isfinite(noise) & (noise > 0.0)]

        rows.append(dict(
            cluster_id=int(cluster_id),
            ifo=loudest.get("ifo", ""),
            gps=start,
            gpsStart=float(group["gpsStart"].min()) if "gpsStart" in group else start,
            gpsCentroid=centroid,
            tSpread=spread,
            gpsPeak=float(loudest.get("gpsPeak", centroid)),
            duration=float(group["window_end"].max() - start),
            duration90=float(np.diff(energy_quantile(
                group["gpsStart"].to_numpy(dtype=float),
                group["gpsStart"].to_numpy(dtype=float)
                + group["duration90"].to_numpy(dtype=float),
                weight, (0.05, 0.95)))[0]) if "duration90" in group else np.nan,
            freqMin=float(group["freqMin"].min()),
            freqMean=float((group["freqMean"].to_numpy(dtype=float) * weight).sum() / total),
            freqMax=float(group["freqMax"].max()),
            freqQ05=float(np.exp(energy_quantile(
                np.log(np.maximum(group["freqQ05"].to_numpy(dtype=float), EPS)),
                np.log(np.maximum(group["freqQ95"].to_numpy(dtype=float), EPS)),
                weight, (0.05,))[0])) if "freqQ05" in group else np.nan,
            freqQ95=float(np.exp(energy_quantile(
                np.log(np.maximum(group["freqQ05"].to_numpy(dtype=float), EPS)),
                np.log(np.maximum(group["freqQ95"].to_numpy(dtype=float), EPS)),
                weight, (0.95,))[0])) if "freqQ95" in group else np.nan,
            EnWDF=float(loudest["EnWDF"]),
            sigma=float(noise.mean()) if noise.size else np.nan,
            snrPeak=float(group["snrPeak"].max()) if "snrPeak" in group else np.nan,
            significance=float(loudest["significance"]),
            n_triggers=int(len(group)),
            n_scales=int(group["n_coeff"].nunique()),
            scale_best=int(loudest["n_coeff"]),
            n_coeff=int(loudest["n_coeff"]),
            fs=float(loudest["fs"]),
        ))
    return pd.DataFrame(rows, columns=DETECTOR_EVENT_COLUMNS)


class EventWavegram:
    """One level-one event's coefficients on the shared band-by-time grid.

    `ClusterCoefficients` renders a cluster on the octave grid of one window
    length, which a multi-window event does not have: its members come from
    several lengths at once. The rows here are absolute frequency bands, shared
    between the lengths, so one event's members all land on the same grid
    whatever length produced them.

    :param grid: (n_bands, n_time_bins) of summed coefficient magnitude.
    """

    def __init__(self, grid: np.ndarray):
        self.grid = grid

    def wavegram(self, n_time_bins: int | None = None) -> np.ndarray:
        """The event's band-by-time grid.

        :type n_time_bins: int | None
        :param n_time_bins: kept for interface compatibility; the grid is
            already rendered at the width the graph was built with.
        :return: numpy.ndarray -- (n_bands, n_time_bins).
        """
        return self.grid


def event_coefficients(graph: DetectorGraph, labels=None,
                       time_bins: int = WAVEGRAM_TIME_BINS) -> dict:
    """Each level-one event's coefficients, for the network graph's nodes.

    :type graph: DetectorGraph
    :param graph: the detector's level-one graph.
    :param labels: component label per node, or None to take every edge.
    :type time_bins: int
    :param time_bins: time bins per band.
    :return: dict -- ``{cluster_id: EventWavegram}``.
    """
    if graph.nodes.empty:
        return {}
    labels = graph.components() if labels is None else np.asarray(labels)
    bands = band_grid(graph.nodes["n_coeff"].to_numpy(),
                      float(graph.nodes["fs"].iloc[0]))
    grids = trigger_wavegrams(graph.nodes, bands, time_bins)
    grids = grids.reshape(len(graph.nodes), len(bands), time_bins)

    out = {}
    for label in np.unique(labels):
        out[int(label)] = EventWavegram(grids[labels == label].sum(axis=0))
    return out
