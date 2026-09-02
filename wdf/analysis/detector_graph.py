"""The detector stage: one detector's triggers as a graph, and its events.

A node is one WDF trigger -- one analysis block's surviving coefficients, at the
window length it was found at. Edges join triggers that could belong to the same
transient: close in time, with energy in bands that overlap or touch, and
continuous in coefficient energy. The connected components are the detector's
events, which are the nodes of the network stage, the inter-detector graph in
`wdf.analysis.network_graph`.

The block is a unit of computation and the event is the physical object, so no
quantity an event reports may depend on where the analysis grid happened to
start. A run searches at one window length; the rule below is written over the
length as well, so that a run configured at more than one joins their triggers
by the same test rather than by a second one.

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

import warnings

import numpy as np
import pandas as pd

from wdf.analysis.metaparameters import energy_quantile
from wdf.analysis.ridge import RIDGE_FEATURES, event_ridge_features
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
    "significance", "n_pixels", "n_triggers", "n_scales", "scale_best", "n_coeff", "fs",
    "EnWDF_window",
] + RIDGE_FEATURES

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


def _one_rate(rate) -> float:
    """The single analysis rate a band ladder is built on.

    The ladder maps a coefficient index to a band through the rate, so two
    rates in one table would place the same index in two different bands and
    every lookup below would return the wrong row for one of them.

    :param rate: the sampling rate of every trigger, Hz.
    :return: float -- the rate they share.
    :raises ValueError: if they do not share one.
    """
    rate = np.asarray(rate, dtype=float)
    usable = rate[np.isfinite(rate) & (rate > 0.0)]
    if not len(usable):
        raise ValueError("no trigger declares an analysis rate")
    # Two rates that differ in the last bit are one rate: the ladder is built
    # from a power of two and a rounding of the same number is not a second
    # instrument.
    if not np.allclose(usable, usable[0], rtol=1e-9, atol=0.0):
        raise ValueError(
            "the band ladder needs one analysis rate, and these triggers "
            f"carry {np.unique(usable).tolist()}")
    return float(usable[0])


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
    missing = ~(np.isfinite(declared) & (declared > 0.0))
    if missing.any():
        # Without the run's stride the fallback is the window's own duration,
        # the stride of a search with no overlap, and every overlapped sample
        # is then counted twice in `stitched_statistic`.
        warnings.warn(
            f"{int(missing.sum())} of {len(missing)} triggers carry no stride; "
            "the window duration is used, which assumes no overlap",
            RuntimeWarning, stacklevel=2)
    declared = np.where(missing, scale / fs, declared)

    strides = {}
    for length in np.unique(scale):
        here = declared[scale == length]
        if not np.allclose(here, here[0], rtol=1e-9, atol=0.0):
            raise ValueError(
                f"windows of {int(length)} samples declare more than one "
                f"stride: {np.unique(here).tolist()}")
        strides[float(length)] = float(here[0])
    return strides


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


def _flat_coefficients(triggers: pd.DataFrame, where: np.ndarray):
    """Every trigger's coefficients laid end to end, with their owners.

    The coefficients are a ragged column, and every use of them is a reduction
    over one trigger's share. Concatenating once and carrying the owner of each
    coefficient turns those reductions into single array operations.

    :type triggers: pandas.DataFrame
    :param triggers: triggers carrying `wt_index` and `wt_value`.
    :type where: numpy.ndarray
    :param where: the row each trigger occupies in the caller's output.
    :return: tuple -- `(index, value, owner)`, one entry per coefficient.
    """
    counts = triggers["wt_index"].map(len).to_numpy()
    if not counts.sum():
        return (np.zeros(0, dtype=np.int64), np.zeros(0),
                np.zeros(0, dtype=np.int64))
    index = np.concatenate([np.asarray(i, dtype=np.int64)
                            for i in triggers["wt_index"]])
    value = np.concatenate([np.asarray(v, dtype=float)
                            for v in triggers["wt_value"]])
    return index, value, np.repeat(where, counts)


def _positive_sigma(triggers: pd.DataFrame) -> np.ndarray:
    """Each trigger's noise scale, with anything unusable left as nan.

    :type triggers: pandas.DataFrame
    :param triggers: triggers that may carry `sigma`.
    :return: numpy.ndarray -- one scale per trigger, nan where the
        trigger declared none that could be a noise scale.
    """
    if "sigma" not in triggers:
        return np.full(len(triggers), np.nan)
    sigma = pd.to_numeric(triggers["sigma"], errors="coerce").to_numpy(float)
    # A scale of one on strain of order 1e-22 would put that trigger's energy
    # 44 decades below every other and drop it out of any weighted moment with
    # nothing to say it happened. It is not a scale, so it is not a number.
    return np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, np.nan)


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
    cells = len(bands) * time_bins
    if triggers.empty:
        return np.zeros((0, cells))

    band_of = {(round(lo, 9), round(hi, 9)): row
               for row, (lo, hi) in enumerate(bands)}
    positions = np.arange(len(triggers))
    sigma = _positive_sigma(triggers)

    # One accumulation over every coefficient of every trigger at once. Walking
    # the triggers instead costs a Python iteration and a scattered add each,
    # which at a search's trigger rate is where the grouping spends its time.
    flat_cell, flat_value = [], []
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
        index, value, owner = _flat_coefficients(group, where)
        if not index.size:
            continue
        # A trigger whose noise scale is missing cannot be placed on the noise
        # scale, and a NaN cell would spread through the norm of the whole
        # grid and through every edge feature that compares it. Its
        # coefficients are left out; the row stays, empty, which is what the
        # search knows about it.
        keep = (row_of[index] >= 0) & np.isfinite(sigma[owner]) & (sigma[owner] > 0.0)
        index, value, owner = index[keep], value[keep], owner[keep]
        # On the noise scale, as the statistic is: the raw coefficients are
        # strain, of order 1e-22, and a grid of those is numerically zero once
        # compressed or multiplied by another.
        flat_cell.append(owner * cells + row_of[index] * time_bins
                         + column_of[index])
        flat_value.append(np.abs(value) / sigma[owner])

    if not flat_cell:
        return np.zeros((len(triggers), cells))
    grid = np.bincount(np.concatenate(flat_cell),
                       weights=np.concatenate(flat_value),
                       minlength=len(triggers) * cells)
    return grid.reshape(len(triggers), cells)


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
        index, _, _ = _flat_coefficients(group, where)
        if not index.size:
            continue
        # One reduction over the concatenated coefficients: a trigger's bits are
        # contiguous in it, so the union per trigger is a segmented OR rather
        # than a pass per trigger.
        counts = group["wt_index"].map(len).to_numpy()
        carrying = counts > 0
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))[carrying]
        mask[where[carrying]] = np.bitwise_or.reduceat(bit_of[index], starts)
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
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        n = len(self.nodes)
        edges = self.edges if keep is None else self.edges[np.asarray(keep, dtype=bool)]
        if n == 0:
            return np.zeros(0, dtype=int)
        if not len(edges):
            return np.arange(n)
        # A union-find in Python costs a pass per edge and a find per node, which
        # at a search's trigger rate dominates the grouping. The relation is the
        # same either way: components of the undirected graph the edges define.
        adjacency = coo_matrix(
            (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
            shape=(n, n))
        _, labels = connected_components(adjacency, directed=False)
        return labels


def build_detector_graph(triggers: pd.DataFrame,
                         significance=None,
                         config: DetectorGraphConfig | None = None) -> DetectorGraph:
    """One detector's detector-stage graph, over all its window lengths at once.

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
    # A node the background never calibrated arrives as NaN. It stays a
    # node --- it is still energy the search kept --- but it is ranked
    # last and never reported as a significance of zero, which would be
    # a measurement.
    unmeasured = ~np.isfinite(significance)
    # A node the background never calibrated cannot be judged against a cut, so
    # it survives only where no cut is asked for. What it reports stays NaN;
    # what the feature matrices carry is zero, since a NaN cell spreads through
    # every norm and every comparison built on it.
    keep = np.where(unmeasured, config.minimum_significance <= 0.0,
                    significance >= config.minimum_significance)
    nodes = nodes[keep].reset_index(drop=True)
    significance, unmeasured = significance[keep], unmeasured[keep]
    if nodes.empty:
        return DetectorGraph(nodes, np.zeros((0, 1)), np.zeros((0, 2), dtype=int),
                             np.zeros((0, len(TRIGGER_EDGE_FEATURES))))

    start = nodes["gps"].to_numpy(dtype=float)
    order = np.argsort(start, kind="mergesort")
    nodes = nodes.iloc[order].reset_index(drop=True)
    significance, unmeasured = significance[order], unmeasured[order]
    featured = np.where(unmeasured, 0.0, significance)

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

    bands = band_grid(scale, _one_rate(fs))
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
        np.column_stack([featured, np.log1p(energy), span, f_hi - f_lo]),
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
            np.minimum(featured[i], featured[j]),
            np.maximum(featured[i], featured[j]),
            (scale[i] != scale[j]).astype(float),
        ]))

    if not edges:
        return DetectorGraph(nodes, node_features, np.zeros((0, 2), dtype=int),
                             np.zeros((0, len(TRIGGER_EDGE_FEATURES))))
    return DetectorGraph(nodes, node_features, np.concatenate(edges),
                         np.concatenate(features))


def tile_context(nodes):
    """The column arrays and geometry cache `event_tiles` reads.

    Extracted once for a whole detector. The columns belong to the table and
    not to one event, so rebuilding them per event turns a pass over the
    coefficients into a pass over the table for every event there is.

    :type nodes: pandas.DataFrame
    :param nodes: the detector's triggers.
    :return: dict -- to be passed to `event_tiles` as `context`. The geometry
        cache is filled lazily, per window length, and shared across events.
    """
    return {
        "gps": nodes["gps"].to_numpy(dtype=float),
        "scale": nodes["n_coeff"].to_numpy(dtype=int),
        "rate": nodes["fs"].to_numpy(dtype=float),
        "sigma": nodes["sigma"].to_numpy(dtype=float),
        "index_of": nodes["wt_index"].to_numpy(),
        "value_of": nodes["wt_value"].to_numpy(),
        "geometry": {},
    }


def event_tiles(nodes, members, context=None):
    """Every surviving coefficient of an event, as tiles in absolute time.

    The event's parameters are properties of the coefficients it kept, so they
    are computed from those coefficients placed on the plane --- not from the
    per-window summaries averaged together, which describe each window's own
    view of a transient that spans several.

    :type nodes: pandas.DataFrame
    :param nodes: the detector's triggers.
    :param members: row positions of the event's members.
    :param context: what `tile_context` returns, built once for the detector.
        Built here when None, which is correct but reads the whole table on
        every call: a caller asking for every event's tiles must build it once
        and pass it, or the cost grows with events times triggers rather than
        with the coefficients there are.
    :return: tuple -- (t_lo, t_hi, f_lo, f_hi, energy, amplitude), all numpy
        arrays over the union of the members' surviving coefficients. `energy`
        is the squared amplitude on the noise scale; `amplitude` keeps the
        coefficient's sign, which a coherent product across detectors needs.
    """
    context = tile_context(nodes) if context is None else context
    geometry = context["geometry"]
    gps, scale, rate = context["gps"], context["scale"], context["rate"]
    sigma = context["sigma"]
    index_of, value_of = context["index_of"], context["value_of"]
    t_lo, t_hi, f_lo, f_hi, energy, amplitude = [], [], [], [], [], []

    for row in members:
        n = int(scale[row])
        if n not in geometry:
            geometry[n] = (coeff_time_bounds(n, float(rate[row])),
                           coeff_freq_bands(n, float(rate[row])))
        (lo, hi), (flo, fhi) = geometry[n]
        index = np.asarray(index_of[row], dtype=int)
        if not index.size:
            continue
        signed = np.asarray(value_of[row], dtype=float)
        scaled = np.abs(signed)
        noise = sigma[row]
        if not (np.isfinite(noise) and noise > 0.0):
            # No scale, no measurement: this window's coefficients are left
            # out rather than divided by one, which on strain would place
            # them forty decades below every other tile.
            continue
        t_lo.append(gps[row] + lo[index])
        t_hi.append(gps[row] + hi[index])
        f_lo.append(flo[index])
        f_hi.append(fhi[index])
        energy.append((scaled / noise) ** 2)
        # The coefficient's sign, kept because a coherent statistic across two
        # detectors is a product of amplitudes and not of magnitudes: the
        # product of magnitudes is positive whatever the data and accumulates
        # with the number of tile pairs, while the signed product has mean zero
        # under the null.
        amplitude.append(signed / noise)

    if not energy:
        return (np.zeros(0),) * 6
    return (np.concatenate(t_lo), np.concatenate(t_hi), np.concatenate(f_lo),
            np.concatenate(f_hi), np.concatenate(energy),
            np.concatenate(amplitude))


def detector_events(graph: DetectorGraph, significance=None,
                    labels=None) -> pd.DataFrame:
    """The detector's events, from the connected components of its graph.

    This is the step from the detector stage to the network stage: the result is the node set of
    the inter-detector network graph, in the schema that graph reads.

    The event's statistic is measured on the reconstruction stitched across its
    members, which counts each sample once. Summing the members instead would
    count one transient's energy several times, since consecutive windows
    overlap, and taking the loudest member would discard the accumulation the
    grouping exists to recover. The loudest member is kept beside it as
    `EnWDF_window`, since that is what a search without this step would report.

    :type graph: DetectorGraph
    :param graph: the detector's detector-stage graph.
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
    # A node the background never calibrated arrives as NaN. It stays a
    # node --- it is still energy the search kept --- but it is ranked
    # last and never reported as a significance of zero, which would be
    # a measurement.
    unmeasured = ~np.isfinite(significance)
    ranked = np.where(unmeasured, -np.inf, significance)

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

    if "wt_index" in nodes:
        kept = nodes["wt_index"].map(len).to_numpy(dtype=float)
        tiles_per_event = np.bincount(grouped, weights=kept[order],
                                      minlength=n_events)
    else:
        tiles_per_event = np.full(n_events, np.nan)

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
    peak = np.lexsort((-ranked, labels))[starts]

    centroid_of = column("gpsCentroid", np.nan)
    centroid_of = np.where(np.isfinite(centroid_of), centroid_of, gps)
    centroid = weighted(centroid_of)
    offset = centroid_of[order] - centroid[grouped]
    spread_of = column("tSpread", 0.0)
    spread = np.sqrt(np.maximum(np.divide(
        np.bincount(grouped, weights=(spread_of[order] ** 2 + offset ** 2) * weight[order],
                    minlength=n_events),
        total, out=np.zeros(n_events), where=total > 0), 0.0))

    # One noise scale per event, the median over the windows it spans. The
    # mean would be pulled by a single loud window's scale, and the statistic
    # this column explains --- the norm of the reconstruction, which
    # `ClusterCoefficients.noise_scale` divides by --- uses the median.
    noise = column("sigma", np.nan)
    usable = np.isfinite(noise) & (noise > 0.0)
    sigma = (pd.Series(np.where(usable, noise, np.nan)).groupby(labels).median()
             .reindex(range(n_events)).to_numpy())

    start = lowest(gps)
    first_tile = lowest(onset)
    # Distinct window lengths per event, counted by reduction: a pass per event
    # is a Python loop over the whole event list.
    pairs = np.unique(np.column_stack([grouped, scale[order]]), axis=0)
    scales_seen = np.bincount(pairs[:, 0].astype(np.int64), minlength=n_events)

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
        # How many tiles the event owns. This is the size its statistic sums
        # over, so it is the scale of that statistic under the null, and unlike
        # the number of member windows it does not move when the analysis grid
        # does. A tile a second window also kept is one tile: the count is of
        # distinct positions in the plane, as the reconstruction is.
        "n_pixels": tiles_per_event,
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

    # The track the event's tiles lie on, where they lie on one. It describes a
    # morphology and decides nothing: admissibility is geometry and physics, and
    # a shape preference there would make the injections a description of the
    # search rather than a check on it.
    for name in RIDGE_FEATURES:
        events[name] = np.nan

    if {"wt_index", "wt_value"} <= set(nodes.columns):
        # The context holds the whole table's tile geometry, so it is built
        # once here: rebuilding it inside the loop costs events times triggers
        # rather than the coefficients there are.
        context = tile_context(nodes)
        for event in np.flatnonzero(sizes > 1):
            members = order[starts[event]:starts[event] + sizes[event]]
            lo, hi, band_lo, band_hi, tile_energy, _ = event_tiles(
                nodes, members, context=context)
            if not tile_energy.size:
                continue
            # Consecutive windows overlap, so one position in the plane can be
            # kept twice. The event owns it once.
            events.at[event, "n_pixels"] = float(len(np.unique(
                np.column_stack([lo, hi, band_lo, band_hi]), axis=0)))
            for name, value in event_ridge_features(
                    lo, hi, band_lo, band_hi, tile_energy).items():
                events.at[event, name] = value
            events.at[event, "duration90"] = float(np.diff(energy_quantile(
                lo, hi, tile_energy, (0.05, 0.95)))[0])
            # The coarsest tile starts at zero frequency, which has no
            # logarithm and no geometric centre: it is represented by half its
            # upper edge, as `tile_frequency` represents it everywhere else.
            # Substituting a tiny positive number instead would put that tile
            # at 1e-154 Hz and drag the whole moment with it.
            floor = np.where(band_lo > 0.0, band_lo, 0.5 * band_hi)
            band = energy_quantile(np.log(np.maximum(floor, EPS)),
                                   np.log(np.maximum(band_hi, EPS)),
                                   tile_energy, (0.05, 0.95))
            events.at[event, "freqQ05"] = float(np.exp(band[0]))
            events.at[event, "freqQ95"] = float(np.exp(band[1]))
            # The geometric moment over the same tiles, which is the frequency
            # the event's energy sits at rather than the mean of the windows'.
            centre = np.where(band_lo > 0.0, np.sqrt(band_lo * band_hi),
                              0.5 * band_hi)
            events.at[event, "freqMean"] = float(np.exp(
                (tile_energy @ np.log(centre)) / max(tile_energy.sum(), EPS)))

    return events[DETECTOR_EVENT_COLUMNS]


class EventWavegram:
    """One event's coefficients on the shared band-by-time grid.

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
    """Each event's coefficients, assembled into one map.

    The members are laid out in absolute time, not at their position inside
    their own window: an event assembled from windows a second apart describes a
    transient a second long, and summing the members' own grids would fold it
    onto itself.

    The map is on a **common** time base --- a bin is `bin_seconds` wherever it
    is drawn --- and anchored on the centre of the tile carrying the event's
    largest coefficient. Scaling each map to
    its own event's extent instead would give a bin a different duration in each
    detector, so two maps compared across the network would be stretched onto
    each other, and the lag that best aligns them would be in no unit at all.
    An event longer than `time_bins * bin_seconds` is truncated around that
    anchor; a shorter one leaves the rest of the map empty. The anchor is what
    makes two maps comparable: it is an instant both detectors measure on the
    same transient, whereas a centroid is a property of what each of them
    recovered.

    :type graph: DetectorGraph
    :param graph: the detector's detector-stage graph.
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
    bands = band_grid(scale, _one_rate(fs))
    row_of = {(round(lo, 9), round(hi, 9)): row for row, (lo, hi) in enumerate(bands)}

    geometry = {}
    for length in np.unique(scale):
        rate = _one_rate(fs)
        t_lo, t_hi = coeff_time_bounds(int(length), rate)
        f_lo, f_hi = coeff_freq_bands(int(length), rate)
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

    # Anchored on the centre of the tile carrying each event's largest
    # coefficient, which is an instant both detectors share. The energy centroid
    # is not: it follows how much of a transient survived threshold in that
    # detector, so an event recovered as one block and its counterpart recovered
    # as five have centroids far apart, and two maps anchored on them are
    # offset by a large part of their own width --- their agreement then
    # measures the difference in extent and not the difference in morphology.
    anchor_of = nodes["gpsPeak"].to_numpy(dtype=float) \
        if "gpsPeak" in nodes else gps
    weight = np.maximum(nodes["EnWDF"].to_numpy(dtype=float) ** 2, EPS)
    half = 0.5 * time_bins * float(bin_seconds)

    # Every event's map is one scatter-add, not one per event and one per
    # member inside it. The maps differ only in where each coefficient lands,
    # so the coefficients of the whole detector are placed once, in flat
    # arrays, and accumulated into a single (event, band, column) volume. The
    # per-event loop and `np.add.at` it replaces cost more than the grouping
    # that produced the events.
    n_labels = int(labels.max()) + 1
    n_bands = len(bands)

    # The loudest member of each event, taken as the first maximum in node
    # order so that ties resolve as a stable pass over the members would.
    key = np.lexsort((np.arange(len(labels)), -weight, labels))
    in_key = labels[key]
    heads = np.flatnonzero(np.r_[True, in_key[1:] != in_key[:-1]])
    loudest = np.zeros(n_labels, dtype=int)
    loudest[in_key[heads]] = key[heads]
    first_of = anchor_of[loudest] - half

    # The ragged per-trigger coefficient lists, concatenated once.
    counts = np.fromiter((len(np.atleast_1d(index_of[n])) for n in range(len(labels))),
                         dtype=int, count=len(labels))
    if counts.sum():
        flat_node = np.repeat(np.arange(len(labels)), counts)
        flat_index = np.concatenate(
            [np.atleast_1d(np.asarray(index_of[n], dtype=int)) for n in range(len(labels))])
        flat_value = np.abs(np.concatenate(
            [np.atleast_1d(np.asarray(value_of[n], dtype=float)) for n in range(len(labels))]))

        # Band row and tile centre depend only on the window length, of which
        # there are a handful, so each is resolved for all its coefficients at
        # once rather than trigger by trigger.
        flat_row = np.full(flat_node.shape, -1, dtype=int)
        flat_centre = np.zeros(flat_node.shape, dtype=float)
        node_scale = scale[flat_node]
        for length in np.unique(scale):
            tile_centre, rows = geometry[int(length)]
            here = node_scale == int(length)
            flat_row[here] = rows[flat_index[here]]
            flat_centre[here] = tile_centre[flat_index[here]]

        sigma = sigma_of[flat_node]
        sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, 1.0)
        event = labels[flat_node]
        when = gps[flat_node] + flat_centre
        column = np.floor((when - first_of[event]) / float(bin_seconds)).astype(int)

        keep = (flat_row >= 0) & (column >= 0) & (column < time_bins)
        linear = ((event[keep] * n_bands + flat_row[keep]) * time_bins
                  + column[keep])
        grids = np.bincount(linear, weights=flat_value[keep] / sigma[keep],
                            minlength=n_labels * n_bands * time_bins)
        grids = grids.reshape(n_labels, n_bands, time_bins)
    else:
        grids = np.zeros((n_labels, n_bands, time_bins))

    # Once for the detector, not once per event.
    tiles = tile_context(nodes)

    out = {}
    for label, (start, size) in enumerate(zip(starts, sizes)):
        members = order[start:start + size]
        out[int(label)] = EventWavegram(grids[label], bin_seconds, bands,
                                        float(first_of[label]),
                                        tiles=event_tiles(nodes, members,
                                                          context=tiles))
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
    :param graph: the detector's detector-stage graph.
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

    # The members of every event, found once by sorting the labels rather than
    # by scanning them once per event, which is quadratic in the event count.
    order = np.argsort(labels, kind="stable")
    sizes = np.bincount(labels, minlength=int(labels.max()) + 1 if len(labels) else 1)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))

    out = {}
    for label in ([cluster_id] if cluster_id is not None
                  else np.flatnonzero(sizes > 0)):
        label = int(label)
        if label >= len(sizes) or not sizes[label]:
            continue
        members = order[starts[label]:starts[label] + sizes[label]]
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
    The statistic is the product of the two amplitudes on their noise scales,
    summed over the tiles that meet --- an inner product taken at the resolution
    the transform actually has, with no grid in between and so no resolution
    chosen by hand.

    The product carries the coefficients' signs. A product of magnitudes is
    positive whatever the data, so it has a mean of about `2 ln N + 2` for every
    pair of tiles that happen to meet and grows with how many pairs there are:
    two long events overlapping by accident then score higher than two short
    ones describing one transient. The signed product has mean zero under the
    null, so what accumulates over many tiles is agreement and not extent.

    It is signed, and both signs are physical: the two detectors' responses to
    one source can have opposite polarity, so a coherent pair may sum to a large
    negative number. What ranks a pair is the magnitude.

    A grid answers the same question after rounding both events onto cells whose
    size someone had to pick: too fine and two detectors share nothing, too
    coarse and everything agrees.

    :param left: one event's tiles, as `event_tiles` returns them.
    :param right: the other event's tiles.
    :type tolerance: float
    :param tolerance: how far the two may be displaced in time and still be
        taken to cover the same place, seconds.
    :return: float -- the coherent energy, in units of the noise scale squared,
        signed by the relative polarity of the two detectors' coefficients.
    """
    t_lo, t_hi, f_lo, f_hi, energy, amplitude = left
    u_lo, u_hi, g_lo, g_hi, other, other_amplitude = right
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
    product = amplitude[:, None] * other_amplitude[None, :]
    return float(product[together].sum())


def flatten_clouds(clouds):
    """Every event's tiles in one set of flat arrays, with offsets.

    :param clouds: one tile tuple per event, as `event_tiles` returns them,
        or None for an event with no tiles.
    :return: dict -- the six tile arrays concatenated, with `starts` and
        `counts` placing each event's stretch inside them.
    """
    counts = np.array([0 if c is None else len(c[4]) for c in clouds],
                      dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    kept = [c for c in clouds if c is not None and len(c[4])]
    if not kept:
        empty = np.zeros(0)
        return dict(t_lo=empty, t_hi=empty, f_lo=empty, f_hi=empty,
                    energy=empty, amplitude=empty, starts=starts, counts=counts)
    field = lambda k: np.concatenate(
        [np.asarray(c[k], dtype=float) for c in clouds
         if c is not None and len(c[4])])
    return dict(t_lo=field(0), t_hi=field(1), f_lo=field(2), f_hi=field(3),
                energy=field(4), amplitude=field(5), starts=starts,
                counts=counts)


# Tile pairs laid out per run of event pairs: enough to keep the reduction
# vectorised, small enough that the working arrays stay bounded whatever the
# events' sizes.
TILE_PAIR_BLOCK = 20_000_000


def tile_coherence_many(flat, i_sel, j_sel, tolerance) -> np.ndarray:
    """`tile_coherence` for many pairs at once, by one reduction.

    The per-pair function forms every pair of tiles inside one call; across a
    background of slides that call is made once per admitted pair, and a Python
    loop over tens of thousands of pairs per slide is where the accidental
    estimate spends its time. Here the cross product of every pair's tiles is
    laid out as one flat index computation and summed with `bincount`, so the
    cost is one pass over the tile pairs however many event pairs there are.

    :param flat: the flattened tiles, as `flatten_clouds` returns them.
    :param i_sel: left event of each pair, as indices into the flattening.
    :param j_sel: right event of each pair.
    :type tolerance: float
    :param tolerance: how far the two may be displaced in time and still be
        taken to cover the same place, seconds.
    :return: numpy.ndarray -- the coherent energy of each pair.
    """
    i_sel = np.asarray(i_sel, dtype=np.int64)
    j_sel = np.asarray(j_sel, dtype=np.int64)
    nl, nr = flat["counts"][i_sel], flat["counts"][j_sel]
    per_pair = (nl * nr).astype(np.int64)
    out = np.zeros(len(i_sel))
    if not int(per_pair.sum()):
        return out

    # The flat layout is exact but its working arrays are the size of the
    # cross product summed over every pair at once: one event assembled from
    # thousands of windows meets hundreds of partners, and the product of the
    # two is billions of tile pairs in a single allocation. The pairs are
    # therefore taken in runs bounded by the tile pairs they lay out, and a
    # single pair too large for a run on its own is split along its left
    # tiles; only the per-pair sums survive either way, so the numbers are
    # the ones the one-shot layout produces.
    bound = int(TILE_PAIR_BLOCK)
    edges = np.cumsum(per_pair)
    first = 0
    while first < len(i_sel):
        base = edges[first] - per_pair[first]
        last = int(np.searchsorted(edges, base + bound, side="left"))
        last = max(last, first + 1)

        if per_pair[first] > bound and last == first + 1:
            li, ri = flat["starts"][i_sel[first]], flat["starts"][j_sel[first]]
            n_left, n_right = int(nl[first]), int(nr[first])
            step = max(bound // max(n_right, 1), 1)
            total = 0.0
            for lo in range(0, n_left, step):
                left = np.repeat(li + np.arange(lo, min(lo + step, n_left)),
                                 n_right)
                right = np.tile(ri + np.arange(n_right),
                                min(lo + step, n_left) - lo)
                total += _tile_pair_energy(flat, left, right, tolerance).sum()
            out[first] = total
            first = last
            continue

        sel = slice(first, last)
        counts = per_pair[sel]
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pair_id = np.repeat(np.arange(last - first), counts)
        within = np.arange(int(counts.sum())) - offsets[pair_id]
        right_count = nr[sel][pair_id]
        left = flat["starts"][i_sel[sel]][pair_id] + within // right_count
        right = flat["starts"][j_sel[sel]][pair_id] + within % right_count
        product = _tile_pair_energy(flat, left, right, tolerance)
        out[first:last] = np.bincount(pair_id, weights=product,
                                      minlength=last - first)
        first = last
    return out


def _tile_pair_energy(flat, left, right, tolerance):
    """The coherent energy of each tile pair, zero where they do not meet.

    :param flat: the flattened tiles, as `flatten_clouds` returns them.
    :param left: flat index of each pair's left tile.
    :param right: flat index of each pair's right tile.
    :type tolerance: float
    :param tolerance: displacement in time still counted as the same place.
    :return: numpy.ndarray -- one signed coherent energy per tile pair.
    """
    meets = ((np.minimum(flat["t_hi"][left], flat["t_hi"][right] + tolerance)
              >= np.maximum(flat["t_lo"][left], flat["t_lo"][right] - tolerance))
             & (np.minimum(flat["f_hi"][left], flat["f_hi"][right])
                >= np.maximum(flat["f_lo"][left], flat["f_lo"][right])))
    product = np.zeros(len(left))
    product[meets] = (flat["amplitude"][left[meets]]
                      * flat["amplitude"][right[meets]])
    return product


def stitched_statistic(graph: DetectorGraph, labels=None) -> np.ndarray:
    """A fast estimate of each event's statistic over its whole extent.

    The statistic of an event is the norm of its stitched reconstruction on the
    noise scale, which counts each sample once however many windows cover it;
    :meth:`wdf.analysis.cluster_coefficients.ClusterCoefficients.enwdf`
    computes it, at the cost of inverting every window of every event. What is
    computed here instead is an estimate of that norm which needs no inverse
    transform: each window's coefficient energy is scaled by the fraction of
    the window its own step region occupies --- a geometric fraction, applied
    to the whole window and not to the tiles that fall inside the step --- and
    the scaled energies are summed in quadrature.

    The two agree when a window's energy is spread evenly over its samples, and
    they differ when it is not. This is an approximation, and an event that is
    ranked, thresholded or reported should carry the reconstruction's own norm
    rather than this.

    Window lengths are not combined: each is a complete description of the same
    strain, so the largest is taken rather than their sum. The result is never
    below the loudest single window, so grouping cannot make an event quieter
    than the search already found it.

    :type graph: DetectorGraph
    :param graph: the detector's detector-stage graph.
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
    lengths, inverse = np.unique(scale, return_inverse=True)
    stride = np.array([strides[length] for length in lengths])[inverse]
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
