"""The detector stage: the wavelet pixels of one detector as a graph over (t, f, scale).

A node is one surviving wavelet coefficient -- a time-frequency tile, at the
window length it was found at. Edges join tiles that could belong to the same
transient: neighbours in time within the same band, and, when a run is
configured at more than one window length, tiles of different lengths whose
cells cover the same region.

The length is carried as a coordinate rather than resolved away, so that tiles
found at different lengths are related by the same rule that relates tiles found
at one, instead of by a reconciliation between separate clusterings.

Assembling the connected components of this graph gives one detector's events,
which are the nodes of the network stage, the inter-detector graph in
`wdf.analysis.network_graph`. The same shape serves either way of deciding
which edges survive: keeping every admissible edge is the deterministic
clustering, and scoring them is the learned one -- both start from the same
graph, which is what makes them comparable.

Which pairs are admissible is a statement about the analysis and never about
the source. The two allowances are the stride the search advanced by and the
step of the dyadic ladder it tiled frequency with; both are ours, both are
symmetric in time and in direction, and neither prefers a rising frequency, a
slope or a monotone sweep. A burst that jumps to a high band and comes back is
admitted exactly as a chirp is. Whether the tiles so admitted *are* a track is
said afterwards, by the descriptors in `wdf.analysis.ridge`, which is where
morphology belongs: a prior about shape in the admission would make the
injections a description of the search instead of a check on it.

Only the pairs inside the tolerance are ever formed, by searching a sorted time
axis. A dense adjacency matrix asks the same question in O(n^2) memory, which a
segment's pixel cloud exhausts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wdf.analysis.pairs import neighbour_pairs
from wdf.analysis.ridge import (RIDGE_FEATURES, event_ridge,
                                event_ridge_features, ridge_members,
                                ridge_track)
from wdf.analysis.robust_events import EPS, _UnionFind

PIXEL_EDGE_FEATURES = [
    "time_gap", "frequency_overlap", "log_scale_ratio",
    "significance_min", "significance_max", "cross_scale",
]


CLUSTER_COLUMNS = [
    "cluster_id", "ifo", "gps", "gpsStart", "gpsCentroid", "tSpread", "gpsPeak",
    "duration", "duration90", "freqMin", "freqMean", "freqMax", "freqQ05",
    "freqQ95", "EnWDF", "EnWDF_window", "sigma", "snrPeak", "significance",
    "energy", "n_pixels", "n_triggers", "n_scales", "scale_best", "n_coeff",
    "fs", "member_indices",
] + RIDGE_FEATURES


@dataclass
class PixelGraphConfig:
    """When two tiles may belong to the same transient.

    :param time_tolerance: largest gap between two tiles' time spans, as a
        fraction of their mean width. Zero joins only tiles that touch or
        overlap; one allows a gap as wide as the tiles themselves.
    :param stride_tolerance: largest gap between two tiles' time spans, in
        strides of the search. The stride is `(window - overlap) / sampling`,
        how far the analysis advanced between consecutive windows, and it is
        the distance at which one transient is detected again: at window 512,
        overlap 32 and 2048 Hz it is 0.234 s, while the tiles themselves are
        0.002 to 0.031 s wide. A tolerance expressed in tile widths therefore
        cannot reach from one detection of a transient to the next whatever
        value it is given, which is why this second allowance exists. It is
        read from the cloud rather than passed in: `pixel_cloud` carries the
        `stride` the run declared onto every tile, and `cloud_strides` reads
        it back. Where the cloud declares none --- a trigger file read without
        its configuration --- the stride is unknown and this allowance
        contributes nothing, leaving `time_tolerance` alone to decide, which
        is the behaviour there was before it. The default of zero reproduces
        exactly that width-based rule.
    :param band_tolerance: how many empty steps of the dyadic ladder may
        separate two tiles' bands, in either direction. A step is one band of
        the ladder the cloud's own edges define, the finest the search
        resolved. Zero admits only bands that overlap or touch, which is the
        rule this generalises; one admits bands with a single empty band
        between them, and so on. The allowance is symmetric --- upward and
        downward are the same number of steps --- so no sweep direction,
        slope or monotonicity is assumed: whether a group of tiles is a track
        is said afterwards, by `ridge_slope`, `ridge_scatter` and
        `ridge_monotonicity`. The default of zero leaves the present rule
        untouched.
    :param minimum_significance: tiles below this are not nodes at all. Reading
        it from the calibrated significance rather than from a raw amplitude is
        what lets one threshold serve every window length and every band.
    """

    time_tolerance: float = 1.0
    stride_tolerance: float = 0.0
    band_tolerance: int = 0
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
        edges = self.edges if keep is None else self.edges[np.asarray(keep, dtype=bool)]
        n = len(self.nodes)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        if not len(edges):
            return np.arange(n, dtype=np.int64)
        # A pixel cloud is one to two orders of magnitude larger than the
        # trigger list it came from, so the components are found by a sparse
        # graph traversal rather than by a Python pass over every edge.
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        adjacency = coo_matrix(
            (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
            shape=(n, n))
        _, labels = connected_components(adjacency, directed=False)
        return labels.astype(np.int64)


def cloud_strides(pixels: pd.DataFrame) -> dict:
    """How far the search advanced between windows, per window length.

    The stride is a property of the run, `(window - overlap) / sampling`, which
    the configuration beside the trigger file declares and which
    `wdf.analysis.scale.pixel_cloud` carries onto every tile. It is read from
    there and never inferred from how far apart the tiles landed: that would
    measure the transients instead of the search, and a stretch where only
    every third window fired would report three times the truth. A cloud that
    declares no stride has none here, and the allowance that rests on it then
    grants nothing.

    :type pixels: pandas.DataFrame
    :param pixels: a pixel cloud, as `wdf.analysis.scale.pixel_cloud` returns.
    :return: dict -- ``{scale: stride}`` in seconds, holding only the window
        lengths whose stride the cloud declares.
    """
    if pixels.empty or "stride" not in pixels:
        return {}
    scale = pixels["scale"].to_numpy(dtype=float)
    declared = pixels["stride"].to_numpy(dtype=float)
    usable = np.isfinite(declared) & (declared > 0.0)
    strides = {}
    for length in np.unique(scale[usable]):
        here = declared[usable & (scale == length)]
        if not np.allclose(here, here[0], rtol=1e-9, atol=0.0):
            raise ValueError(
                f"windows of {int(length)} samples declare more than one "
                f"stride: {np.unique(here).tolist()}")
        strides[float(length)] = float(here[0])
    return strides


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
    from wdf.analysis.scale import unique_tiles

    config = PixelGraphConfig() if config is None else config
    # Two windows share samples, so the same region of the plane can be
    # thresholded in both and arrive twice. It is one region and it becomes one
    # node: otherwise the event's energy counts it as many times as there were
    # windows over it.
    kept_rows = unique_tiles(pixels)
    # A significance the caller measured on the cloud has one entry per row of
    # the cloud, so it is indexed by the rows that survived rather than taken
    # whole: after the first duplicate the two would be a row apart.
    survivors = pixels.index.get_indexer(kept_rows.index)
    nodes = kept_rows.reset_index(drop=True)

    if significance is None:
        significance = np.log1p(nodes["energy"].to_numpy(dtype=float)) if len(nodes) \
            else np.zeros(0)
    significance = np.asarray(significance, dtype=float)
    if len(significance) == len(pixels) and len(pixels) != len(nodes):
        significance = significance[survivors]
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
        return PixelGraph(nodes, np.zeros((0, 2), dtype=int),
                          np.zeros((0, len(PIXEL_EDGE_FEATURES))))

    t_lo = nodes["t_lo"].to_numpy(dtype=float)
    order = np.argsort(t_lo, kind="mergesort")
    nodes = nodes.iloc[order].reset_index(drop=True)
    significance, unmeasured = significance[order], unmeasured[order]
    featured = np.where(unmeasured, 0.0, significance)

    t_lo = nodes["t_lo"].to_numpy(dtype=float)
    t_hi = nodes["t_hi"].to_numpy(dtype=float)
    f_lo = nodes["f_lo"].to_numpy(dtype=float)
    f_hi = nodes["f_hi"].to_numpy(dtype=float)
    scale = nodes["scale"].to_numpy(dtype=float)
    width = t_hi - t_lo

    # The admission rests on the geometry of the analysis and on nothing else:
    # the stride the search advanced by, and the ladder it tiled frequency
    # with. Both are ours. Neither says anything about the source, so a burst
    # that jumps an octave and comes back is admitted exactly as a sweep is.
    strides = cloud_strides(nodes) if config.stride_tolerance > 0.0 else {}
    stride = np.array([strides.get(float(value), 0.0) for value in scale])
    # The ladder the cloud itself resolved: every band edge present, so a step
    # is one band and a gap is a whole number of them.
    ladder = np.unique(np.concatenate([f_lo, f_hi]))

    # The widest tile plus its own tolerance is how far apart two tiles' starts
    # can be and still touch, which bounds the search along the time axis.
    reach = float((width * (1.0 + config.time_tolerance)).max()
                  + config.stride_tolerance * (stride.max() if stride.size else 0.0))

    edges, features = [], []
    for left, right in neighbour_pairs(t_lo, reach):
        gap = np.maximum(t_lo[right] - t_hi[left], t_lo[left] - t_hi[right])
        allowed = np.maximum(
            config.time_tolerance * 0.5 * (width[left] + width[right]),
            config.stride_tolerance * np.maximum(stride[left], stride[right]))
        # Empty steps of the ladder between the two bands: zero where they
        # overlap or touch, which is the rule a tolerance of zero keeps.
        below = np.minimum(f_hi[left], f_hi[right])
        above = np.maximum(f_lo[left], f_lo[right])
        steps = (np.searchsorted(ladder, above) - np.searchsorted(ladder, below))
        band = steps <= int(config.band_tolerance)
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
            np.minimum(featured[i], featured[j]),
            np.maximum(featured[i], featured[j]),
            (scale[i] != scale[j]).astype(float),
        ]))

    if not edges:
        return PixelGraph(nodes, np.zeros((0, 2), dtype=int),
                          np.zeros((0, len(PIXEL_EDGE_FEATURES))))
    return PixelGraph(nodes, np.concatenate(edges),
                      np.concatenate(features))


def cluster_events(graph: PixelGraph, significance=None, labels=None) -> pd.DataFrame:
    """One detector's events, from the connected components of its pixel graph.

    This is the step from the detector stage to the network stage: what comes out is the node
    set of the inter-detector network graph. An event is the wavegram itself,
    the connected set of tiles, and every quantity below is a moment over those
    tiles and over nothing else.

    Two groups of them, deliberately. The energy, the statistic, the centroid,
    the spread and the mean frequency are taken over the tiles of one window
    length, the one carrying the loudest tile, because the lengths all describe
    the same strain. The support --- first and last time, lowest and highest
    band --- and the counts are taken over every tile the event holds, since
    that is what the event covers.

    Each tile is normalised by the noise scale of the window that produced it,
    so the event's statistic is::

        rho = sqrt( sum_k |c_k|^2 / sigma_k^2 ),

    which is the norm of the waveform those tiles invert to, in units of the
    noise. A cluster spanning windows of different noise is then summed on each
    window's own scale rather than on an average of them.

    Energy is summed over one window length only --- the one carrying the
    loudest tile --- because the lengths all describe the same strain and
    summing across them would count the transient several times. The
    significance is the largest of the cluster's tiles, a maximum over
    correlated searches, and has to be calibrated on the background in its own
    right.

    :type graph: PixelGraph
    :param graph: the detector's pixel graph, whose tiles carry `energy` and
        the `sigma` of the window each came from.
    :param significance: each node's calibrated significance, or None to rank
        the nodes on their energy alone. A tile the background never calibrated
        arrives as NaN and stays NaN; it is not read as zero.
    :param labels: component label per node, or None to take every edge.
    :return: pandas.DataFrame -- one row per event, with `CLUSTER_COLUMNS`.
    """
    from wdf.analysis.metaparameters import energy_quantile
    from wdf.analysis.scale import normalised_energy
    from wdf.analysis.wavelets import tile_frequency

    nodes = graph.nodes
    if nodes.empty:
        return pd.DataFrame(columns=CLUSTER_COLUMNS)

    labels = graph.components() if labels is None else np.asarray(labels)
    labels = np.asarray(labels, dtype=np.int64)
    n_events = int(labels.max()) + 1

    if significance is None:
        significance = np.log1p(nodes["energy"].to_numpy(dtype=float))
    significance = np.asarray(significance, dtype=float)

    # Each tile on the noise scale of its own window. A tile whose scale was
    # not recorded carries no measurable energy and is left out of the sums
    # rather than counted as zero.
    weight = normalised_energy(nodes)
    usable = np.isfinite(weight) & (weight > 0.0)
    weight = np.where(usable, weight, 0.0)

    t_lo = nodes["t_lo"].to_numpy(dtype=float)
    t_hi = nodes["t_hi"].to_numpy(dtype=float)
    f_lo = nodes["f_lo"].to_numpy(dtype=float)
    f_hi = nodes["f_hi"].to_numpy(dtype=float)
    scale = nodes["scale"].to_numpy(dtype=float)
    centre = 0.5 * (t_lo + t_hi)
    width = np.maximum(t_hi - t_lo, 0.0)
    # The lower edge of the coarsest tile is zero, which has no logarithm; that
    # tile is represented by half its upper edge, as `tile_frequency` does.
    band_lo = np.where(f_lo > 0.0, f_lo, 0.5 * f_hi)
    band_centre = np.where(f_lo > 0.0, np.sqrt(np.maximum(f_lo, EPS) * f_hi),
                           0.5 * f_hi)

    # The loudest tile of each event, and the window length it was found at:
    # the energy is summed on that length alone.
    order = np.lexsort((-weight, labels))
    sizes = np.bincount(labels, minlength=n_events)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    peak = order[starts]
    scale_best = scale[peak]
    on_best = scale == scale_best[labels]
    counted = weight * on_best

    total = np.bincount(labels, weights=counted, minlength=n_events)
    n_pixels = np.bincount(labels, weights=on_best.astype(float),
                           minlength=n_events).astype(int)
    safe = np.maximum(total, EPS)

    centroid = np.bincount(labels, weights=counted * centre,
                           minlength=n_events) / safe
    # The tiles' own widths belong in the spread: a tile holds its energy over
    # its extent and not at its centre, and a uniform extent of width w carries
    # a variance of w^2/12 about it.
    about = (centre - centroid[labels]) ** 2 + width ** 2 / 12.0
    spread = np.sqrt(np.maximum(
        np.bincount(labels, weights=counted * about, minlength=n_events) / safe, 0.0))
    log_frequency = np.bincount(labels, weights=counted * np.log(np.maximum(band_centre, EPS)),
                                minlength=n_events) / safe

    lowest_time = np.full(n_events, np.inf)
    np.minimum.at(lowest_time, labels, t_lo)
    highest_time = np.full(n_events, -np.inf)
    np.maximum.at(highest_time, labels, t_hi)
    lowest_band = np.full(n_events, np.inf)
    np.minimum.at(lowest_band, labels, f_lo)
    highest_band = np.full(n_events, -np.inf)
    np.maximum.at(highest_band, labels, f_hi)
    loudest = np.zeros(n_events)
    np.maximum.at(loudest, labels, weight)
    best_significance = np.full(n_events, -np.inf)
    finite_significance = np.isfinite(significance)
    np.maximum.at(best_significance, labels[finite_significance],
                  significance[finite_significance])
    best_significance = np.where(np.isfinite(best_significance),
                                 best_significance, np.nan)

    # What one window alone would have reported: the loudest single trigger's
    # share of this event, which is the quantity the grouping is judged against.
    trigger = nodes["trigger_index"].to_numpy()
    _, per_trigger = np.unique(np.column_stack([labels, trigger]), axis=0,
                               return_inverse=True)
    per_trigger = np.asarray(per_trigger).reshape(-1)
    by_trigger = np.bincount(per_trigger, weights=counted)
    event_of_trigger = np.zeros(len(by_trigger), dtype=np.int64)
    event_of_trigger[per_trigger] = labels
    window_best = np.zeros(n_events)
    np.maximum.at(window_best, event_of_trigger, by_trigger)
    n_triggers = np.bincount(event_of_trigger, minlength=n_events)

    # One noise scale per event, the median over the windows it spans, which is
    # what `ClusterCoefficients` reports for the same cluster.
    sigma = nodes["sigma"].to_numpy(dtype=float)
    noise = (pd.Series(np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, np.nan))
             .groupby(labels).median().reindex(range(n_events)).to_numpy())

    # The quantiles invert a mixture over the tiles' own extents, so they are
    # computed only where there is a mixture; a single tile is its own support.
    # The quantiles are a property of the tiles that carry the energy, so an
    # event with fewer than two of them has none. `detector_events` leaves the
    # same columns unset for a single-member event; reporting the full extent
    # instead would be a different quantity under the same name.
    duration90 = np.full(n_events, np.nan)
    band_q05 = np.full(n_events, np.nan)
    band_q95 = np.full(n_events, np.nan)
    multiple = np.flatnonzero(sizes > 1)
    if len(multiple):
        for event in multiple:
            rows = order[starts[event]:starts[event] + sizes[event]]
            rows = rows[counted[rows] > 0.0]
            if len(rows) < 2:
                continue
            here = counted[rows]
            low, high = energy_quantile(t_lo[rows], t_hi[rows], here, (0.05, 0.95))
            duration90[event] = high - low
            low, high = energy_quantile(np.log(np.maximum(band_lo[rows], EPS)),
                                        np.log(np.maximum(f_hi[rows], EPS)),
                                        here, (0.05, 0.95))
            band_q05[event], band_q95[event] = np.exp(low), np.exp(high)

    # Which triggers the event was assembled from. The reconstruction reads the
    # coefficients back from the trigger frame, and a trigger's tiles can fall
    # in more than one event, so the mapping is event to triggers and not the
    # other way round.
    members = np.split(trigger[order], np.cumsum(sizes)[:-1])
    member_indices = [np.unique(group) for group in members]

    # The ridge descriptors, from the event's own tiles. `detector_events`
    # computes the same five from the same function, so the two paths report
    # one definition.
    ridge = {name: np.full(n_events, np.nan) for name in RIDGE_FEATURES}
    for event in np.flatnonzero(sizes > 1):
        rows = order[starts[event]:starts[event] + sizes[event]]
        rows = rows[counted[rows] > 0.0]
        if len(rows) < 2:
            continue
        for name, value in event_ridge_features(
                t_lo[rows], t_hi[rows], f_lo[rows], f_hi[rows],
                counted[rows]).items():
            ridge[name][event] = value

    measured = total > 0.0
    events = pd.DataFrame({
        "cluster_id": np.arange(n_events),
        "ifo": nodes["ifo"].to_numpy()[peak] if "ifo" in nodes else "",
        "gps": lowest_time,
        "gpsStart": lowest_time,
        "gpsCentroid": np.where(measured, centroid, centre[peak]),
        "tSpread": np.where(measured, spread, width[peak] / np.sqrt(12.0)),
        "gpsPeak": centre[peak],
        "duration": highest_time - lowest_time,
        "duration90": duration90,
        "freqMin": lowest_band,
        "freqMean": np.where(measured, np.exp(log_frequency), band_centre[peak]),
        "freqMax": highest_band,
        "freqQ05": band_q05,
        "freqQ95": band_q95,
        "EnWDF": np.where(measured, np.sqrt(total), np.nan),
        "EnWDF_window": np.where(measured, np.sqrt(window_best), np.nan),
        "sigma": noise,
        "snrPeak": np.where(measured, np.sqrt(loudest), np.nan),
        "significance": best_significance,
        "energy": np.where(measured, total, np.nan),
        "n_pixels": n_pixels,
        "n_triggers": n_triggers,
        "n_scales": pd.Series(scale).groupby(labels).nunique()
                      .reindex(range(n_events)).to_numpy(),
        "scale_best": scale_best.astype(int),
        "n_coeff": scale_best.astype(int),
        "fs": nodes["fs"].to_numpy(dtype=float)[peak],
        **ridge,
    })
    events["member_indices"] = member_indices
    return events[CLUSTER_COLUMNS]


def follow_ridges(graph: PixelGraph, labels=None, n_bins: int = 32) -> np.ndarray:
    """Each event cut down to the tiles its own ridge passes through.

    The admission joins tiles on the geometry of the analysis alone, so an
    event can hold, beside the path a transient traces across the plane, tiles
    that only sit near it. The ridge says which of them follow the path:
    `wdf.analysis.ridge.event_ridge` keeps the loudest tile of each time bin,
    `ridge_track` fills the bins between them by a straight line in log
    frequency, and `ridge_members` keeps every tile whose own band holds the
    track at the centre of the tile's time support. The band is the corridor,
    so no width is chosen, and the fill reads the same forwards and backwards
    in time, so a falling path is followed exactly as a rising one and a
    burst that jumps up and comes back is followed as it goes.

    The tiles on the path are the event. What the path leaves behind is not
    discarded: the rest of the cluster's tiles are grouped again by the
    graph's own edges among themselves, and every group is an event of its
    own. The stage therefore takes no tile out of the detector's events; it
    decides which tiles are summed together.

    A path is one frequency per instant. Tiles of a cluster lying at the same
    instant in a band the track does not cross are, for this stage, another
    event --- which is what it says about a transient spread over several
    bands at once, whatever its origin.

    :type graph: PixelGraph
    :param graph: the detector's pixel graph.
    :param labels: component label per node, or None to take every edge.
    :type n_bins: int
    :param n_bins: time bins an event's extent is divided into for its ridge.
    :return: numpy.ndarray -- one label per node, contiguous from zero, the
        path of an event and each group it leaves behind labelled apart.
    """
    from wdf.analysis.scale import normalised_energy

    nodes = graph.nodes
    if nodes.empty:
        return np.zeros(0, dtype=np.int64)
    labels = np.asarray(graph.components() if labels is None else labels,
                        dtype=np.int64)
    t_lo = nodes["t_lo"].to_numpy(dtype=float)
    t_hi = nodes["t_hi"].to_numpy(dtype=float)
    f_lo = nodes["f_lo"].to_numpy(dtype=float)
    f_hi = nodes["f_hi"].to_numpy(dtype=float)
    energy = normalised_energy(nodes)
    energy = np.where(np.isfinite(energy), energy, 0.0)

    on_path = np.ones(len(nodes), dtype=bool)
    order = np.argsort(labels, kind="stable")
    sizes = np.bincount(labels)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    for event in np.flatnonzero(sizes > 1):
        rows = order[starts[event]:starts[event] + sizes[event]]
        time, log_frequency, _ = event_ridge(t_lo[rows], t_hi[rows], f_lo[rows],
                                             f_hi[rows], energy[rows], n_bins)
        edges = np.linspace(float(t_lo[rows].min()), float(t_hi[rows].max()),
                            int(n_bins) + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        track, _ = ridge_track(time, log_frequency, bin_centres=centres)
        on_path[rows] = ridge_members(t_lo[rows], t_hi[rows], f_lo[rows],
                                      f_hi[rows], track, centres)

    # What the path leaves behind is regrouped by the edges joining two tiles
    # that are both off it; a tile on the path takes its event's label.
    edges = graph.edges
    if len(edges):
        off = ~on_path[edges[:, 0]] & ~on_path[edges[:, 1]]
        rest = graph.components(keep=off)
    else:
        rest = np.arange(len(nodes), dtype=np.int64)
    key = np.where(on_path, labels, int(labels.max()) + 1 + rest)
    _, relabelled = np.unique(key, return_inverse=True)
    return np.asarray(relabelled, dtype=np.int64).reshape(-1)


def tile_labels(pixels: pd.DataFrame, graph: PixelGraph, labels=None) -> np.ndarray:
    """The event every tile of a cloud belongs to, the repeated ones included.

    The graph holds each region of the plane once: where two overlapping
    windows both kept a tile, `wdf.analysis.scale.unique_tiles` keeps the
    larger estimate as the node, so that the event's energy counts the region
    once. Both windows nevertheless measured it, and a reconstruction stitched
    across them needs both estimates: the overlap-add averages the windows over
    the samples they share, and a window stripped of the tile its neighbour
    kept would halve the region there. Every tile of the cloud is therefore
    given the label of the node describing the same region --- same detector,
    same window length, same edges --- which is the rule `unique_tiles` joins
    them by.

    A tile no node describes, one below the graph's `minimum_significance`,
    belongs to no event and is labelled -1.

    :type pixels: pandas.DataFrame
    :param pixels: the cloud the graph was built from, as
        `wdf.analysis.scale.pixel_cloud` returns it.
    :type graph: PixelGraph
    :param graph: its pixel graph.
    :param labels: component label per node, or None to take every edge.
    :return: numpy.ndarray -- one label per row of `pixels`, positionally.
    :raises ValueError: if two nodes describe the same region, which a graph
        built by `build_pixel_graph` never holds.
    """
    if pixels.empty:
        return np.zeros(0, dtype=np.int64)
    nodes = graph.nodes
    if nodes.empty:
        return np.full(len(pixels), -1, dtype=np.int64)
    labels = np.asarray(graph.components() if labels is None else labels,
                        dtype=np.int64)
    key = [name for name in ("ifo",) if name in pixels and name in nodes]
    key += ["scale", "t_lo", "f_lo"]
    owners = nodes[key].assign(_label=labels)
    if owners.duplicated(subset=key).any():
        raise ValueError("two nodes of the graph describe the same region of "
                         "the plane")
    joined = pixels[key].reset_index(drop=True).merge(owners, on=key, how="left")
    return joined["_label"].fillna(-1).to_numpy(dtype=np.int64)


def cluster_wavegrams(graph: PixelGraph, labels=None, time_bins: int = 64,
                      bin_seconds: float | None = None) -> dict:
    """Each event's wavegram, rendered from the tiles the event owns.

    The map is the event itself on a band by time grid: one row per band of the
    shared ladder, one column per `bin_seconds`, and a cell carrying the
    coefficient magnitude on its own window's noise scale. Nothing outside the
    cluster is drawn --- a tile the neighbouring window kept but that the
    grouping did not join is not part of this event and does not appear.

    The columns are anchored on the centre of the event's loudest tile, an
    instant both detectors measure on the same transient, rather than on a
    centroid, which follows how much of the transient each of them recovered.
    A column stands for the same time wherever it is drawn, so two maps
    compared across the network are not stretched onto each other.

    Each map also carries the event's tiles themselves, in the six arrays
    `wdf.analysis.detector_graph.event_tiles` returns: support in time and
    band, energy on the noise scale, and the signed amplitude on the noise
    scale. That is what the network stage renders and compares, so an event
    assembled here is a node of `wdf.analysis.network_graph.TriggerGraphBuilder`
    exactly as an event of the trigger graph is. The sign is the coefficient's
    own and is read from the cloud's `value`; a cloud that did not record it
    carries no signed amplitude, and the amplitude is then not a number rather
    than a magnitude standing in for it, since a coherent product of
    magnitudes is positive whatever the data.

    :type graph: PixelGraph
    :param graph: the detector's pixel graph.
    :param labels: component label per node, or None to take every edge.
    :type time_bins: int
    :param time_bins: columns of the map.
    :type bin_seconds: float | None
    :param bin_seconds: seconds a column stands for; the narrowest tile of the
        cloud when None, which is the finest the search resolved.
    :return: dict -- ``{cluster_id: EventWavegram}``.
    """
    from wdf.analysis.detector_graph import EventWavegram, band_grid
    from wdf.analysis.scale import normalised_energy

    nodes = graph.nodes
    if nodes.empty:
        return {}
    labels = np.asarray(graph.components() if labels is None else labels,
                        dtype=np.int64)
    n_events = int(labels.max()) + 1

    t_lo = nodes["t_lo"].to_numpy(dtype=float)
    t_hi = nodes["t_hi"].to_numpy(dtype=float)
    f_lo = nodes["f_lo"].to_numpy(dtype=float)
    f_hi = nodes["f_hi"].to_numpy(dtype=float)
    scale = nodes["scale"].to_numpy(dtype=float)
    rate = nodes["fs"].to_numpy(dtype=float)
    centre = 0.5 * (t_lo + t_hi)
    amplitude = np.sqrt(np.maximum(normalised_energy(nodes), 0.0))
    amplitude = np.where(np.isfinite(amplitude), amplitude, 0.0)
    sigma = nodes["sigma"].to_numpy(dtype=float)
    signed = np.full(len(nodes), np.nan)
    if "value" in nodes:
        signed = np.divide(nodes["value"].to_numpy(dtype=float), sigma,
                           out=signed,
                           where=np.isfinite(sigma) & (sigma > 0.0))

    bands = band_grid(scale, float(np.median(rate[np.isfinite(rate)])))
    row_of = {(round(lo, 9), round(hi, 9)): row
              for row, (lo, hi) in enumerate(bands)}
    rows = np.array([row_of.get((round(a, 9), round(b, 9)), -1)
                     for a, b in zip(f_lo, f_hi)])

    width = np.maximum(t_hi - t_lo, 0.0)
    if bin_seconds is None:
        positive = width[width > 0.0]
        bin_seconds = float(positive.min()) if positive.size else 1.0
    bin_seconds = float(bin_seconds)

    # The loudest tile of each event, and the window it opens on it.
    order = np.lexsort((np.arange(len(labels)), -amplitude, labels))
    heads = np.flatnonzero(np.r_[True, labels[order][1:] != labels[order][:-1]])
    loudest = np.zeros(n_events, dtype=int)
    loudest[labels[order][heads]] = order[heads]
    first_of = centre[loudest] - 0.5 * time_bins * bin_seconds

    column = np.floor((centre - first_of[labels]) / bin_seconds).astype(int)
    keep = (rows >= 0) & (column >= 0) & (column < time_bins) & (amplitude > 0.0)
    n_bands = len(bands)
    grids = np.bincount(
        ((labels[keep] * n_bands + rows[keep]) * time_bins + column[keep]),
        weights=amplitude[keep],
        minlength=n_events * n_bands * time_bins
    ).reshape(n_events, n_bands, time_bins)

    out = {}
    sizes = np.bincount(labels, minlength=n_events)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    by_label = np.argsort(labels, kind="stable")
    for event in range(n_events):
        members = by_label[starts[event]:starts[event] + sizes[event]]
        out[event] = EventWavegram(
            grids[event], bin_seconds=bin_seconds, bands=bands,
            gps_first=float(first_of[event]),
            tiles=(t_lo[members], t_hi[members], f_lo[members], f_hi[members],
                   amplitude[members] ** 2, signed[members]))
    return out
