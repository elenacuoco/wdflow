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
    "EnWDF_window",
]

WAVEGRAM_TIME_BINS = 64


@dataclass
class DetectorGraphConfig:
    """When two triggers may belong to the same transient.

    :param time_tolerance: largest gap between two triggers' time supports, as
        a fraction of their mean window span. Zero joins only triggers that
        touch or overlap; one allows a gap as wide as a window.
    :param minimum_frequency_overlap: least shared fraction of the two supports.
        The support is the interval the surviving tiles span, which a broadband
        transient makes wide, so the band-set test below carries the frequency
        condition and this defaults to nothing.
    :param band_adjacency: how many octave rows apart two occupied bands may be
        and still count as touching. One says energy in a band connects to
        energy in that band or in either neighbour.
    :param maximum_log_energy_jump: largest jump in coefficient energy between
        two triggers of one transient, as a log ratio. A transient rises and
        falls smoothly; a step of several decades is two things, not one.
    :param minimum_significance: triggers below this are not nodes at all.
    :param wavegram_time_bins: time bins per band in a node's wavegram.
    """

    time_tolerance: float = 1.0
    minimum_frequency_overlap: float = 0.0
    band_adjacency: int = 1
    maximum_log_energy_jump: float = 3.0
    minimum_significance: float = 0.0
    # Whether two windows must have been represented in the same basis to
    # continue one another. Off by default: a transient whose character changes
    # --- a chirp entering a different regime --- can legitimately be won by a
    # different basis, and the band test already asks for continuity of energy.
    same_basis: bool = False
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


def window_strides(triggers: pd.DataFrame) -> dict:
    """How far the search advanced between consecutive windows of each length.

    The stride is `(window - overlap) / sampling`, a property of the run.
    `wdf.analysis.io.triggers_from_files` writes it onto every trigger from the
    configuration beside the file. Triggers that carry no stride fall back to
    their window's own duration, which is the stride of a search with no
    overlap.

    :type triggers: pandas.DataFrame
    :param triggers: triggers carrying `n_coeff`, `fs` and, where the run
        recorded it, `stride`.
    :return: dict -- ``{n_coeff: stride}`` in seconds.
    """
    if triggers.empty:
        return {}
    scale = triggers["n_coeff"].to_numpy(dtype=float)
    fs = np.maximum(triggers["fs"].to_numpy(dtype=float), EPS)
    declared = (triggers["stride"].to_numpy(dtype=float) if "stride" in triggers
                else np.full(len(triggers), np.nan))
    declared = np.where(np.isfinite(declared) & (declared > 0.0),
                        declared, scale / fs)

    lengths, first = np.unique(scale, return_index=True)
    return {float(length): float(declared[row]) for length, row in zip(lengths, first)}


def wavegram_bin_seconds(triggers: pd.DataFrame) -> float:
    """The duration one column of an event's map stands for.

    A column means the same time wherever it is drawn, so that two maps
    compared across the network are not stretched onto each other and the lag
    that best aligns them is a time. The finest the search can resolve is the
    stride of its shortest window, which is what this is.

    :type triggers: pandas.DataFrame
    :param triggers: triggers carrying `gps`, `n_coeff` and `fs`.
    :return: float -- seconds per column.
    """
    strides = window_strides(triggers)
    return min(strides.values()) if strides else 1.0


def trigger_wavegrams(triggers: pd.DataFrame, bands: np.ndarray,
                      time_bins: int) -> np.ndarray:
    """Each trigger's coefficients on a band-by-time grid of one fixed shape.

    :type triggers: pandas.DataFrame
    :param triggers: triggers carrying `n_coeff`, `fs` and the coefficients.
    :type bands: numpy.ndarray
    :param bands: (n_bands, 2) band edges, as `band_grid` returns.
    :type time_bins: int
    :param time_bins: time bins per band, spanning each trigger's own window.
    :return: numpy.ndarray -- (n_triggers, n_bands * time_bins), the coefficient
        magnitudes on the noise scale that produced the trigger.
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
            # On the noise scale, as the statistic is: the raw coefficients are
            # strain, of order 1e-22, and a grid of those is numerically zero
            # once compressed or multiplied by another.
            sigma = float(trigger.get("sigma", 1.0))
            sigma = sigma if np.isfinite(sigma) and sigma > 0.0 else 1.0
            value = np.abs(np.asarray(trigger["wt_value"], dtype=float)) / sigma
            keep = row_of[index] >= 0
            np.add.at(grid[slot], (row_of[index[keep]], column_of[index[keep]]),
                      value[keep])
    return grid.reshape(len(triggers), -1)


def occupied_bands(triggers: pd.DataFrame, bands: np.ndarray) -> np.ndarray:
    """Which octave bands each trigger's surviving coefficients fall in.

    Not the interval they span: the support of a broadband transient covers
    everything beneath it, while the bands it occupies are where its energy
    actually is. The rows are those of the shared ladder, so the same physical
    band is the same bit at every window length.

    :type triggers: pandas.DataFrame
    :param triggers: triggers carrying `n_coeff`, `fs` and the coefficients.
    :type bands: numpy.ndarray
    :param bands: (n_bands, 2) band edges, as `band_grid` returns.
    :return: numpy.ndarray -- one unsigned integer per trigger, bit `r` set
        where the trigger has a coefficient in band `r`.
    """
    mask = np.zeros(len(triggers), dtype=np.uint64)
    if triggers.empty:
        return mask
    if len(bands) > 64:
        raise ValueError(
            f"{len(bands)} bands do not fit one integer; the ladder spans "
            "sampling rate over window length and should not reach this")

    row_of_band = {(round(lo, 9), round(hi, 9)): row
                   for row, (lo, hi) in enumerate(bands)}
    position = np.arange(len(triggers))
    for (n_coeff, fs), group in triggers.groupby(["n_coeff", "fs"], sort=False):
        f_lo, f_hi = coeff_freq_bands(int(n_coeff), float(fs))
        row_of = np.array([row_of_band.get((round(lo, 9), round(hi, 9)), -1)
                           for lo, hi in zip(f_lo, f_hi)])
        bit_of = np.where(row_of >= 0,
                          np.left_shift(np.uint64(1), np.maximum(row_of, 0)
                                        .astype(np.uint64)),
                          np.uint64(0))
        where = position[triggers.index.get_indexer(group.index)]
        for slot, index in zip(where, group["wt_index"]):
            mask[slot] = np.bitwise_or.reduce(
                bit_of[np.asarray(index, dtype=int)], initial=np.uint64(0))
    return mask


def _bands_touch(left: np.ndarray, right: np.ndarray, adjacency: int) -> np.ndarray:
    """Whether two band sets share a row, or sit within `adjacency` of one."""
    spread = right.copy()
    for shift in range(1, int(adjacency) + 1):
        spread |= np.left_shift(right, np.uint64(shift))
        spread |= np.right_shift(right, np.uint64(shift))
    return (left & spread) != np.uint64(0)


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
    # Continuity is between one window's energy and the next window's, not
    # between the windows: the window is the same length whatever it holds, and
    # consecutive windows overlap by construction, so a gap measured on them is
    # negative almost everywhere and joins triggers that share nothing. The
    # surviving tiles are where the energy is.
    onset = nodes["gpsStart"].to_numpy(dtype=float) if "gpsStart" in nodes else start
    extent = nodes["duration"].to_numpy(dtype=float) if "duration" in nodes else span
    onset = np.where(np.isfinite(onset), onset, start)
    extent = np.where(np.isfinite(extent), extent, span)
    end = onset + extent
    f_lo = nodes["freqMin"].to_numpy(dtype=float)
    f_hi = nodes["freqMax"].to_numpy(dtype=float)
    energy = np.maximum(nodes["EnWDF"].to_numpy(dtype=float) ** 2, EPS)
    log_energy = np.log(energy)

    bands = band_grid(scale, float(fs[0]))
    occupied = occupied_bands(nodes, bands)
    basis = (nodes["wave"].to_numpy().astype(str) if "wave" in nodes
             else np.zeros(len(nodes), dtype=int))
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

    # The pairs are searched on the window start, which the nodes are sorted by,
    # so the reach has to cover a tile sitting anywhere inside its window.
    reach = float((2.0 * span.max()) * (1.0 + config.time_tolerance))
    edges, features = [], []
    for left, right in neighbour_pairs(start, reach):
        gap = np.maximum(onset[right] - end[left], onset[left] - end[right])
        allowed = config.time_tolerance * 0.5 * (span[left] + span[right])
        shared = np.minimum(f_hi[left], f_hi[right]) - np.maximum(f_lo[left], f_lo[right])
        narrower = np.maximum(np.minimum(f_hi[left] - f_lo[left],
                                         f_hi[right] - f_lo[right]), EPS)
        overlap = np.clip(shared / narrower, 0.0, 1.0)
        # Continuity, which is what this level asks: close in time, energy in a
        # band finding energy in that band or one beside it in the other
        # trigger, and the coefficient energy continuing. Overlap of the
        # supports would admit a broadband transient against everything it
        # happens to sit on, its energy being nowhere near.
        join = (
            (gap <= allowed)
            & (overlap >= config.minimum_frequency_overlap)
            & _bands_touch(occupied[left], occupied[right], config.band_adjacency)
            & (np.abs(log_energy[left] - log_energy[right])
               <= config.maximum_log_energy_jump)
        )
        if config.same_basis:
            # The basis that represented a window most compactly is a statement
            # about the shape in it. One transient does not change shape from
            # one window to the next, so two windows the competition assigned to
            # different bases are, by the competition's own verdict, two
            # different shapes.
            join &= basis[left] == basis[right]
        if not join.any():
            continue
        i, j = left[join], right[join]
        covered = np.minimum(end[i], end[j]) - np.maximum(onset[i], onset[j])
        edges.append(np.column_stack([i, j]))
        features.append(np.column_stack([
            onset[i] - onset[j],
            np.clip(covered / np.maximum(np.minimum(extent[i], extent[j]), EPS),
                    0.0, 1.0),
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


def event_tiles(nodes, members):
    """Every surviving coefficient of an event, as tiles in absolute time.

    The event's parameters are properties of the coefficients it kept, so they
    are computed from those coefficients placed on the plane --- not from the
    per-window summaries averaged together, which describe each window's own
    view of a transient that spans several.

    :type nodes: pandas.DataFrame
    :param nodes: the detector's triggers.
    :param members: row positions of the event's members.
    :return: tuple -- (t_lo, t_hi, f_lo, f_hi, energy), all numpy arrays over
        the union of the members' surviving coefficients.
    """
    geometry = {}
    t_lo, t_hi, f_lo, f_hi, energy = [], [], [], [], []
    gps = nodes["gps"].to_numpy(dtype=float)
    scale = nodes["n_coeff"].to_numpy(dtype=int)
    rate = nodes["fs"].to_numpy(dtype=float)
    sigma = nodes["sigma"].to_numpy(dtype=float)
    index_of = nodes["wt_index"].to_numpy()
    value_of = nodes["wt_value"].to_numpy()

    for row in members:
        n = int(scale[row])
        if n not in geometry:
            geometry[n] = (coeff_time_bounds(n, float(rate[row])),
                           coeff_freq_bands(n, float(rate[row])))
        (lo, hi), (flo, fhi) = geometry[n]
        index = np.asarray(index_of[row], dtype=int)
        if not index.size:
            continue
        scaled = np.abs(np.asarray(value_of[row], dtype=float))
        noise = sigma[row] if np.isfinite(sigma[row]) and sigma[row] > 0 else 1.0
        t_lo.append(gps[row] + lo[index])
        t_hi.append(gps[row] + hi[index])
        f_lo.append(flo[index])
        f_hi.append(fhi[index])
        energy.append((scaled / noise) ** 2)

    if not energy:
        return (np.zeros(0),) * 5
    return (np.concatenate(t_lo), np.concatenate(t_hi), np.concatenate(f_lo),
            np.concatenate(f_hi), np.concatenate(energy))


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

    def column(name, default=np.nan):
        if name not in nodes:
            return np.full(len(nodes), default, dtype=float)
        values = pd.to_numeric(nodes[name], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(values), values, default)

    gps = column("gps")
    scale = column("n_coeff")
    fs = np.maximum(column("fs", 1.0), EPS)
    # The event's extent is the extent of its energy, from the first surviving
    # tile to the last. Taking the windows' extent instead adds up to one whole
    # window at each end, whatever the transient inside them was.
    onset = column("gpsStart", np.nan)
    onset = np.where(np.isfinite(onset), onset, gps)
    extent = column("duration", np.nan)
    extent = np.where(np.isfinite(extent), extent, scale / fs)
    tile_end = onset + extent

    # Grouped by reduction over a sorted label, not by a Python pass per cluster.
    # A search carrying no per-detector threshold produces tens of thousands of
    # components, and the threshold scan builds the events once per candidate
    # threshold.
    order = np.argsort(labels, kind="stable")
    grouped = labels[order]
    sizes = np.bincount(grouped, minlength=int(labels.max()) + 1)
    n_events = len(sizes)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))

    weight = np.maximum(column("EnWDF", 0.0) ** 2, EPS)
    total = np.bincount(grouped, weights=weight[order], minlength=n_events)

    def weighted(values):
        summed = np.bincount(grouped, weights=values[order] * weight[order],
                             minlength=n_events)
        return np.divide(summed, total, out=np.full(n_events, np.nan),
                         where=total > 0)

    def lowest(values):
        return np.minimum.reduceat(values[order], starts)

    def highest(values):
        return np.maximum.reduceat(values[order], starts)

    # The loudest member of each cluster: ordering by cluster and then by
    # decreasing significance puts it first in every group.
    peak = np.lexsort((-significance, labels))[starts]

    centroid_of = column("gpsCentroid", np.nan)
    centroid_of = np.where(np.isfinite(centroid_of), centroid_of, gps)
    centroid = weighted(centroid_of)
    offset = centroid_of[order] - centroid[grouped]
    spread_of = column("tSpread", 0.0)
    spread = np.sqrt(np.maximum(np.divide(
        np.bincount(grouped, weights=(spread_of[order] ** 2 + offset ** 2) * weight[order],
                    minlength=n_events),
        total, out=np.zeros(n_events), where=total > 0), 0.0))

    noise = column("sigma", np.nan)
    usable = np.isfinite(noise) & (noise > 0.0)
    counted = np.bincount(grouped, weights=usable[order].astype(float), minlength=n_events)
    sigma = np.divide(
        np.bincount(grouped, weights=np.where(usable, noise, 0.0)[order], minlength=n_events),
        counted, out=np.full(n_events, np.nan), where=counted > 0)

    start = lowest(gps)
    first_tile = lowest(onset)
    scales_seen = np.array([len(np.unique(scale[order][a:a + n]))
                            for a, n in zip(starts, sizes)])

    events = pd.DataFrame({
        "cluster_id": np.arange(n_events),
        "ifo": nodes["ifo"].to_numpy()[peak] if "ifo" in nodes else "",
        "gps": start,
        "gpsStart": first_tile,
        "gpsCentroid": centroid,
        "tSpread": spread,
        "gpsPeak": column("gpsPeak", np.nan)[peak] if "gpsPeak" in nodes else centroid,
        "duration": highest(tile_end) - first_tile,
        "freqMin": lowest(column("freqMin", np.inf)),
        "freqMean": weighted(column("freqMean", np.nan)),
        "freqMax": highest(column("freqMax", -np.inf)),
        # The event's statistic is measured over its whole extent, counting
        # each sample once; the loudest single window is kept beside it, since
        # that is what a search without this step would have reported.
        "EnWDF_window": column("EnWDF", 0.0)[peak],
        "EnWDF": stitched_statistic(graph, labels),
        "sigma": sigma,
        "snrPeak": highest(column("snrPeak", -np.inf)),
        "significance": significance[peak],
        "n_triggers": sizes,
        "n_scales": scales_seen,
        "scale_best": scale[peak].astype(int),
        "n_coeff": scale[peak].astype(int),
        "fs": fs[peak],
    })

    # The energy quantiles invert a mixture per cluster, so they are computed
    # only where there is a mixture: a cluster of one member simply keeps its
    # own, which at these event rates is most of them.
    # The quantities that follow the energy are measured on the event's own
    # coefficients, gathered from every member and placed on the plane in
    # absolute time. Averaging the members' summaries instead describes each
    # window's view of a transient that spans several of them, which for a long
    # event is a mixture of fragments and not the event.
    quantile_columns = ("duration90", "freqQ05", "freqQ95")
    for name in quantile_columns:
        events[name] = column(name, np.nan)[peak]

    if {"wt_index", "wt_value"} <= set(nodes.columns):
        for event in np.flatnonzero(sizes > 1):
            members = order[starts[event]:starts[event] + sizes[event]]
            lo, hi, band_lo, band_hi, tile_energy = event_tiles(nodes, members)
            if not tile_energy.size:
                continue
            events.at[event, "duration90"] = float(np.diff(energy_quantile(
                lo, hi, tile_energy, (0.05, 0.95)))[0])
            band = energy_quantile(np.log(np.maximum(band_lo, EPS)),
                                   np.log(np.maximum(band_hi, EPS)),
                                   tile_energy, (0.05, 0.95))
            events.at[event, "freqQ05"] = float(np.exp(band[0]))
            events.at[event, "freqQ95"] = float(np.exp(band[1]))
            # The geometric moment over the same tiles, which is the frequency
            # the event's energy sits at rather than the mean of the windows'.
            centre = np.sqrt(np.maximum(band_lo, EPS) * np.maximum(band_hi, EPS))
            events.at[event, "freqMean"] = float(np.exp(
                (tile_energy @ np.log(centre)) / max(tile_energy.sum(), EPS)))

    return events[DETECTOR_EVENT_COLUMNS]


class EventWavegram:
    """One level-one event's coefficients on the shared band-by-time grid.

    `ClusterCoefficients` renders a cluster on the octave grid of one window
    length, which a multi-window event does not have: its members come from
    several lengths at once. The rows here are absolute frequency bands, shared
    between the lengths, so one event's members all land on the same grid
    whatever length produced them.

    :param grid: (n_bands, n_time_bins) of summed coefficient magnitude.
    :param bin_seconds: duration one column stands for, seconds. Carried with
        the grid so that a lag between two maps can be read as a time.
    :param bands: (n_bands, 2) frequency edges of the rows, Hz.
    :param gps_first: GPS time of the left edge of the first column. With
        `bin_seconds` and `bands` this places the map in time and frequency; a
        map that cannot be placed is not a map, and rederiving the placement at
        the point of use puts two copies of the same arithmetic in the code.
    :param tiles: the event's coefficients as tiles on the plane, as
        `event_tiles` returns them --- the lossless description the grid is a
        projection of. Everything an event is asked for should derive from this
        one object: the map, the parameters, the reconstruction and the
        comparison across detectors, so that they cannot describe different
        things.
    """

    def __init__(self, grid: np.ndarray, bin_seconds: float = 1.0,
                 bands: np.ndarray | None = None, gps_first: float = 0.0,
                 tiles=None):
        self.grid = grid
        self.bin_seconds = float(bin_seconds)
        self.bands = np.zeros((len(grid), 2)) if bands is None else np.asarray(bands)
        self.gps_first = float(gps_first)
        self.tiles = tiles

    def times(self) -> np.ndarray:
        """The left edge of every column, in GPS seconds.

        :return: numpy.ndarray -- one entry per column.
        """
        return self.gps_first + self.bin_seconds * np.arange(self.grid.shape[1])

    def wavegram(self, n_time_bins: int | None = None) -> np.ndarray:
        """The event's band-by-time grid.

        :type n_time_bins: int | None
        :param n_time_bins: kept for interface compatibility; the grid is
            already rendered at the width the graph was built with.
        :return: numpy.ndarray -- (n_bands, n_time_bins).
        """
        return self.grid


def event_coefficients(graph: DetectorGraph, labels=None,
                       time_bins: int = WAVEGRAM_TIME_BINS,
                       bin_seconds: float | None = None) -> dict:
    """Each level-one event's coefficients, assembled into one map.

    The members are laid out in absolute time, not at their position inside
    their own window: an event assembled from windows a second apart describes a
    transient a second long, and summing the members' own grids would fold it
    onto itself.

    The map is on a **common** time base --- a bin is `bin_seconds` wherever it
    is drawn --- and centred on the event's energy centroid. Scaling each map to
    its own event's extent instead would give a bin a different duration in each
    detector, so two maps compared across the network would be stretched onto
    each other, and the lag that best aligns them would be in no unit at all.
    An event longer than `time_bins * bin_seconds` is truncated to the centre of
    its energy; a shorter one leaves the rest of the map empty.

    :type graph: DetectorGraph
    :param graph: the detector's level-one graph.
    :param labels: component label per node, or None to take every edge.
    :type time_bins: int
    :param time_bins: columns of the map.
    :type bin_seconds: float | None
    :param bin_seconds: duration one column stands for, seconds; measured from
        the triggers with `wavegram_bin_seconds` when None.
    :return: dict -- ``{cluster_id: EventWavegram}``.
    """
    nodes = graph.nodes
    if nodes.empty:
        return {}
    labels = graph.components() if labels is None else np.asarray(labels)
    bin_seconds = (wavegram_bin_seconds(nodes) if bin_seconds is None
                   else float(bin_seconds))

    fs = np.maximum(nodes["fs"].to_numpy(dtype=float), EPS)
    scale = nodes["n_coeff"].to_numpy(dtype=int)
    gps = nodes["gps"].to_numpy(dtype=float)
    bands = band_grid(scale, float(fs[0]))
    row_of = {(round(lo, 9), round(hi, 9)): row for row, (lo, hi) in enumerate(bands)}

    geometry = {}
    for length in np.unique(scale):
        t_lo, t_hi = coeff_time_bounds(int(length), float(fs[0]))
        f_lo, f_hi = coeff_freq_bands(int(length), float(fs[0]))
        rows = np.array([row_of.get((round(a, 9), round(b, 9)), -1)
                         for a, b in zip(f_lo, f_hi)])
        geometry[int(length)] = (0.5 * (t_lo + t_hi), rows)

    order = np.argsort(labels, kind="stable")
    grouped = labels[order]
    sizes = np.bincount(grouped, minlength=int(labels.max()) + 1)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))

    index_of = nodes["wt_index"].to_numpy()
    value_of = nodes["wt_value"].to_numpy()
    sigma_of = nodes["sigma"].to_numpy(dtype=float)
    span = scale / fs

    centroid_of = nodes["gpsCentroid"].to_numpy(dtype=float) \
        if "gpsCentroid" in nodes else gps
    weight = np.maximum(nodes["EnWDF"].to_numpy(dtype=float) ** 2, EPS)
    half = 0.5 * time_bins * float(bin_seconds)

    out = {}
    for label, (start, size) in enumerate(zip(starts, sizes)):
        members = order[start:start + size]
        # Centred on where the energy is, so a truncated map keeps the part that
        # carries the signal rather than the part that happens to come first.
        centre = float(np.average(centroid_of[members], weights=weight[members]))
        first = centre - half

        grid = np.zeros((len(bands), time_bins))
        for node in members:
            tile_centre, rows = geometry[int(scale[node])]
            index = np.asarray(index_of[node], dtype=int)
            keep = rows[index] >= 0
            if not keep.any():
                continue
            sigma = sigma_of[node]
            sigma = sigma if np.isfinite(sigma) and sigma > 0.0 else 1.0
            # Absolute time, then the event's own extent.
            when = gps[node] + tile_centre[index[keep]]
            column = np.floor((when - first) / float(bin_seconds)).astype(int)
            inside = (column >= 0) & (column < time_bins)
            if not inside.any():
                continue
            np.add.at(grid, (rows[index[keep]][inside], column[inside]),
                      (np.abs(np.asarray(value_of[node], dtype=float))[keep]
                       / sigma)[inside])
        out[int(label)] = EventWavegram(grid, bin_seconds, bands, first,
                                        tiles=event_tiles(nodes, members))
    return out


def event_waveform(graph, labels=None, cluster_id=None):
    """The event's reconstruction in the time domain.

    The second thing the assembled map is for. Each member window is inverted
    and lays down its own step region, so the pieces tile the covered span
    exactly once and no sample is counted twice. Window lengths are not
    combined: each is a complete description of the same strain, so the one
    carrying most of the event's energy is the one reconstructed, and the others
    are what a search at a single length would have produced instead.

    :type graph: DetectorGraph
    :param graph: the detector's level-one graph.
    :param labels: component label per node, or None to take every edge.
    :type cluster_id: int | None
    :param cluster_id: the event to reconstruct; every event when None.
    :return: dict -- ``{cluster_id: (gps_start, samples, n_coeff)}``.
    """
    from wdf.analysis.reconstruction import stitch

    nodes = graph.nodes
    if nodes.empty:
        return {}
    labels = graph.components() if labels is None else np.asarray(labels)

    strides = window_strides(nodes)
    energy = nodes["EnWDF"].to_numpy(dtype=float) ** 2
    scale = nodes["n_coeff"].to_numpy(dtype=float)
    rate = nodes["fs"].to_numpy(dtype=float)

    out = {}
    for label in ([cluster_id] if cluster_id is not None
                  else np.unique(labels)):
        members = np.flatnonzero(labels == int(label))
        if not members.size:
            continue
        # The length that carries most of this event's energy.
        lengths, inverse = np.unique(scale[members], return_inverse=True)
        carried = np.bincount(inverse, weights=energy[members],
                              minlength=len(lengths))
        best = float(lengths[int(np.argmax(carried))])
        here = members[scale[members] == best]

        fs = float(rate[here[0]])
        overlap = int(round(best - strides[best] * fs))
        out[int(label)] = stitch(nodes.iloc[here], fs, int(best), overlap) + (int(best),)
    return out


def tile_coherence(left, right, tolerance):
    """Coherent energy of two events, on their coefficients and not on a grid.

    Every surviving coefficient is a rectangle on the plane, and two events
    describe the same transient where their rectangles cover the same place.
    The statistic is the geometric mean of the two energies over the tiles that
    meet, summed --- an inner product taken at the resolution the transform
    actually has, with no grid in between and so no resolution chosen by hand.

    A grid answers the same question after rounding both events onto cells whose
    size someone had to pick: too fine and two detectors share nothing, too
    coarse and everything agrees.

    :param left: one event's tiles, as `event_tiles` returns them.
    :param right: the other event's tiles.
    :type tolerance: float
    :param tolerance: how far the two may be displaced in time and still be
        taken to cover the same place, seconds.
    :return: float -- the coherent energy, in units of the noise scale squared.
    """
    t_lo, t_hi, f_lo, f_hi, energy = left
    u_lo, u_hi, g_lo, g_hi, other = right
    if not energy.size or not other.size:
        return 0.0

    # Every pair of tiles, which is affordable because thresholding leaves a
    # handful per window: an event of twenty windows has tens of tiles, not the
    # thousands a dense representation would pair.
    meets_in_time = (np.minimum(t_hi[:, None], u_hi[None, :] + tolerance)
                     >= np.maximum(t_lo[:, None], u_lo[None, :] - tolerance))
    meets_in_band = (np.minimum(f_hi[:, None], g_hi[None, :])
                     >= np.maximum(f_lo[:, None], g_lo[None, :]))
    together = meets_in_time & meets_in_band
    if not together.any():
        return 0.0
    product = np.sqrt(energy[:, None] * other[None, :])
    return float(product[together].sum())


def stitched_statistic(graph: DetectorGraph, labels=None) -> np.ndarray:
    """Each event's statistic over its whole extent, without double counting.

    Consecutive windows of one length step by less than their span, so a sample
    can appear in several of them and summing their coefficient energy counts it
    more than once. Within one window length the step regions tile time exactly
    once, so the energy of an event is accumulated over the step region of each
    of its windows and no further. Window lengths are not combined: each is a
    complete description of the same strain, so the largest is taken rather than
    their sum.

    :type graph: DetectorGraph
    :param graph: the detector's level-one graph.
    :param labels: component label per node, or None to take every edge.
    :return: numpy.ndarray -- one statistic per event.
    """
    nodes = graph.nodes
    if nodes.empty:
        return np.zeros(0)
    labels = graph.components() if labels is None else np.asarray(labels)
    n_events = int(labels.max()) + 1

    fs = np.maximum(nodes["fs"].to_numpy(dtype=float), EPS)
    scale = nodes["n_coeff"].to_numpy(dtype=float)

    # The fraction of a window that is its own step region, which is what it
    # contributes without repeating what a neighbour already carried.
    strides = window_strides(nodes)
    stride = np.array([strides[length] for length in scale])
    share = np.clip(stride / (scale / fs), 0.0, 1.0)

    # A window's step region is what it contributes without repeating what a
    # neighbour carried; a window with no neighbour repeats nothing, so an event
    # is never quieter than its loudest single window.
    best = np.zeros(n_events)
    enwdf = nodes["EnWDF"].to_numpy(dtype=float)
    np.maximum.at(best, labels, enwdf)

    # Each window is already on its own noise scale --- EnWDF is |c|/sigma for
    # the sigma measured in that window --- so the event's statistic is the sum
    # of its members' squares over the step regions. Summing the unnormalised
    # energy and dividing by an average sigma instead is only the same quantity
    # when the noise does not change between windows, which is what a search
    # over a long stretch cannot assume.
    lengths, scale_index = np.unique(scale, return_inverse=True)
    group = labels * len(lengths) + scale_index
    summed = np.bincount(group, weights=share * enwdf ** 2,
                         minlength=n_events * len(lengths))
    per_length = np.sqrt(summed).reshape(n_events, len(lengths))
    return np.maximum(best, per_length.max(axis=1))
