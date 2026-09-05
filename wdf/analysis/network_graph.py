"""The network graph: single-detector events as nodes, physically possible
coincidences as cross-detector edges.

Plain numpy and pandas. The graph is the structure a network statistic is
computed on, whether that statistic is learned or not, so it must not require
the learning machinery to exist: `wdf.analysis.baseline` fits a deterministic
ranking on exactly this graph without torch, and `wdf.analysis.gnn` fits a
learned one on the same graph with it. Comparing the two at a fixed false alarm
rate is only meaningful because both start from the same candidate set.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from wdf.analysis.detector_graph import (WAVEGRAM_TIME_BINS, flatten_clouds,
                                         tile_coherence_many)
from wdf.analysis.pairs import neighbour_pairs, paired_dot
from wdf.analysis.wavegram_match import (BIN_PER_TILE,
                                         compare_on_pair_grids, render)
from wdf.analysis.robust_events import (
    EPS,
    _numeric,
    _overlap_fraction,
    CoincidenceConfig,
    IndexedCoincidenceFinder,
    _coefficient_energy,
)


# A node is described by the signed wavelet coefficients of its cluster on a
# compact octave-by-time grid, not by scalar summaries. That local grid is a
# fixed-size node feature; it does not carry the detector-to-detector arrival
# delay. The physical coincidence is measured separately by rendering each
# event's tiles on its own instant and searching the map lags that reach the
# displacements the light travel time admits, the anchor difference being
# carried alongside so that the displacement found is an absolute one.
# The grid's width is the detector stage's, imported rather than restated: two
# constants for the shape of one object drift apart.

# What a candidate edge carries: the arrival-time difference, the agreement
# between the two wavegrams, the shared fraction of band and of time support,
# and the log ratio of the two energies. The ratio is a feature and not a
# penalty: the antenna responses make the same signal reach two detectors with
# amplitudes differing by a factor of a few, so an unequal pair is physical.
EDGE_FEATURES = ["dt_s", "wavegram_similarity", "frequency_overlap",
                 "time_overlap", "log_energy_ratio",
                 "wavegram_overlap", "energy_band_overlap", "dt_over_tolerance",
                 "network_correlation", "coherent_statistic", "tile_coherence"]

# The coherent energy alone is large wherever two events are loud together,
# which a strong instrumental transient in one detector paired with a strong
# noise event in the other satisfies. The correlation normalises it by the
# energy present,
#
#     cc = 2 |<w1, w2>| / (<w1, w1> + <w2, w2>),
#
# so it reaches one only where the two grids agree in shape *and* in amplitude,
# and a pair that is merely loud does not. The ranking statistic is the coherent
# amplitude weighted by it, after coherent WaveBurst's construction
# \cite{klimenko2016cwb}: loud alone is not enough, and consistent alone is not
# enough.

# A cosine between two unit-normalised grids is a statement about direction and
# not about evidence: two events each occupying one cell of the plane reach
# exactly one whenever that cell is the same, which independent noise does by
# chance. The overlap is the inner product of the grids before normalisation,
# so it is large only where the two are loud together and aligned --- the
# coherent energy of the pair, in the units the wavegram carries.

# Independently anchoring two compact node grids removes their arrival-time
# difference. Such grids may describe each event to a learned model, but they
# are not the detector-coincidence test. That test uses the tiles' absolute GPS
# supports and one global shift bounded by the light travel time and timing
# spread; `dt_s` records the resulting convention `t_left - t_right`.
N_EDGE_FEATURES = len(EDGE_FEATURES)


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    """Compress coefficient magnitude without discarding its sign."""
    values = np.asarray(values)
    return np.sign(values) * np.log1p(np.abs(values))


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
        cross_edges, cross_edge_features, ifos, cross_edge_profiles=None,
        cross_edge_lags=None, cross_edge_match=None, cross_edge_match_dt=None,
        cross_edge_measured=None,
    ):
        self.nodes = nodes
        self.node_features = node_features
        self.intra_edges = intra_edges
        self.intra_edge_features = intra_edge_features
        self.cross_edges = cross_edges
        self.cross_edge_features = cross_edge_features
        self.cross_edge_profiles = (np.zeros((len(cross_edges), 0, 0), dtype=np.float32)
                                    if cross_edge_profiles is None else cross_edge_profiles)
        self.cross_edge_lags = (np.zeros(0, dtype=float) if cross_edge_lags is None
                                else cross_edge_lags)
        # The comparison is taken on each pair's own grid, over a search as
        # wide as that pair's transient, so its result cannot be recovered from
        # the profile carried here --- that is the tolerance's own lags, the
        # part every pair can be read on. The agreement, the displacement it
        # was found at, and whether the pair was compared at all are therefore
        # carried beside it rather than reduced again.
        n_edges = len(cross_edges)
        self.cross_edge_match = (np.zeros(n_edges) if cross_edge_match is None
                                 else cross_edge_match)
        self.cross_edge_match_dt = (np.full(n_edges, np.nan)
                                    if cross_edge_match_dt is None
                                    else cross_edge_match_dt)
        self.cross_edge_measured = (np.zeros(n_edges, dtype=bool)
                                    if cross_edge_measured is None
                                    else cross_edge_measured)
        self.ifos = ifos

    def candidate_table(self) -> pd.DataFrame:
        """Cross-detector candidate edges as a table.

        Shares its schema with `IndexedCoincidenceFinder.find`'s output, so the
        classical and the learned statistic can be ranked and compared through
        the same machinery. Every edge feature is carried through, which is what
        a deterministic baseline is fitted on.

        :return: pandas.DataFrame -- one row per candidate edge.
        """
        i, j = self.cross_edges[:, 0], self.cross_edges[:, 1]
        # The clock the two events' maps are anchored on, and the one the
        # wavegram match reports its displacement in. It is not the clock
        # `dt_s` is measured on --- that is each event's own instant, which
        # `INSTANT_COLUMNS` prefers read on the reconstruction --- so the two
        # are not to be added to each other. The energy centroid is a property
        # of how much of the transient each detector recovered, so two
        # detectors seeing one source at different amplitudes place their
        # centroids differently, which is why it anchors nothing.
        gps = self.nodes["gpsPeak"].to_numpy(dtype=float)
        # An event is ranked by the block that selects it and not by the
        # energy that measures it: a sum over an event's tiles carries the
        # threshold's floor once per tile, so it grows with the event's
        # size in noise as well as in signal.
        column = ("EnWDF_window" if "EnWDF_window" in self.nodes
                  else "EnWDF")
        enwdf = self.nodes[column].to_numpy(dtype=float)
        ifo = self.nodes["ifo"].to_numpy()
        table = pd.DataFrame(dict(
            candidate_id=np.arange(len(i)),
            gps_candidate=(gps[i] + gps[j]) / 2,
            ifos_involved=np.char.add(np.char.add(ifo[i].astype(str), ","),
                                      ifo[j].astype(str)),
            network_enwdf=np.hypot(enwdf[i], enwdf[j]),
            network_min_enwdf=np.minimum(enwdf[i], enwdf[j]),
            n_ifos=2,
            node_i=i,
            node_j=j,
        ))
        for column, name in enumerate(EDGE_FEATURES):
            table[name] = self.cross_edge_features[:, column].astype(float)
        # The comparison already answered all three questions the profile is
        # carried for, on each pair's own grid: how alike the two renderings
        # are, at which displacement, and whether the pair was compared at all.
        # Reducing the carried profile again would answer a fourth --- the
        # best agreement within the tolerance --- and give it these names.
        table["network_wavegram_match"] = self.cross_edge_match
        # The pair's arrival-time difference as the match measures it, out of
        # the same pass as the agreement, so it is not a second estimate to be
        # reconciled with it: it is where the agreement is. It is not measured
        # on the same clock as `dt_s`: the maps are anchored on `gpsPeak`, and
        # the whole bins of the difference between the two anchors are applied
        # as a shift, so what is reported is the absolute displacement the two
        # renderings agree at on that clock. A pair compared at no displacement
        # has none to report, and says so rather than naming the first point of
        # an axis.
        table["network_wavegram_match_dt"] = self.cross_edge_match_dt
        table["network_wavegram_matched"] = self.cross_edge_measured
        if (self.cross_edge_profiles.ndim == 3
                and self.cross_edge_profiles.shape[0] == len(table)
                and self.cross_edge_profiles.shape[2] > 0):
            within = self.cross_edge_profiles.sum(axis=1)
            table["network_correlation_lag_s"] = self.cross_edge_lags[
                np.argmax(np.abs(within), axis=1)]
        else:
            table["network_correlation_lag_s"] = np.nan

        # How loud the pair is, weighted by how alike the two grids are. The
        # weight carries no amplitude of its own -- the cosine is invariant to
        # scale -- so the loudness is not counted twice.
        table["network_shape_weighted"] = (
            table["network_enwdf"] * table["wavegram_similarity"])

        # Admissibility is on the events' extents, so a pair may now be admitted
        # seconds apart when both events are long. The arrival times of one
        # signal still differ by at most the light travel time, and dt is
        # measured on every pair -- so a candidate that used none of the
        # tolerance it was allowed keeps its full statistic, and one that used
        # all of it is discounted. This puts dt where GstLAL puts it, in the
        # ranking rather than in the gate.
        table["network_min_enwdf_timed"] = (
            table["network_min_enwdf"] / (1.0 + table["dt_over_tolerance"]))

        # The morphological statistic, on the coefficients themselves. Where
        # network_min_enwdf asks only that both detectors were loud, this asks
        # that they were loud in the same places on the plane --- which is what
        # the whole representation was built to be able to ask.
        # Both polarities are physical --- two detectors can respond to one
        # source with opposite sign --- so what ranks a pair is how much
        # coherent energy it carries, not which way it points.
        table["network_morphology"] = table["tile_coherence"].abs()
        return table


class TriggerGraphBuilder:
    """The network graph: single-detector events as nodes, physically possible
    coincidences as cross-detector edges.

    The edges are not the graph's own invention. They are the candidate pairs
    the classical finder admits -- the two events' stretches of time meeting
    once one may shift by the light travel time plus their own timing spreads,
    and their bands overlapping -- so that the learned and the classical
    statistic rank the same candidate set and can be compared at a fixed false
    alarm rate. The difference of the two events' own instants is measured on
    every such pair and ranks it; it does not decide which pairs exist. What the
    graph decides is which of those survivors are coherent, not which pairs are
    geometrically possible: that is known physics, and it is imposed rather than
    learned.

    A node carries the cluster's wavegram, not only its scalar summary, so the
    morphology that the coefficients measured reaches the model. The detector's
    identity is carried explicitly, because the antenna responses make unequal
    amplitudes between detectors physical rather than suspicious.
    """

    def __init__(
        self,
        intra_ifo_window_s: float = 5.0,
        coincidence: "CoincidenceConfig | None" = None,
        ifos: list[str] | None = None,
        wavegram_time_bins: int = WAVEGRAM_TIME_BINS,
        match_wavegrams: bool = True,
    ):
        """
        :type intra_ifo_window_s: float
        :param intra_ifo_window_s: how far apart two events of the same detector
            may be and still be joined, as local-density context.
        :type coincidence: wdf.analysis.robust_events.CoincidenceConfig | None
        :param coincidence: the physical admissibility rule the cross-detector
            edges obey. Default: the classical finder's own configuration.
        :type ifos: list[str] | None
        :param ifos: detector order; default, the order of the events given.
        :type wavegram_time_bins: int
        :param wavegram_time_bins: time bins per octave in a node's wavegram.
        """
        self.intra_ifo_window_s = intra_ifo_window_s
        self.coincidence = CoincidenceConfig() if coincidence is None else coincidence
        # Whether the two renderings of a coincident pair are compared. The
        # comparison is a morphological baseline and enters no ranking, and it
        # costs a correlation per candidate, so a study that does not read it
        # need not pay for it. With it off, `network_wavegram_match`,
        # `network_correlation` and `coherent_statistic` are not measured, and
        # `network_wavegram_matched` says so rather than a zero standing in for
        # a measurement.
        self.match_wavegrams = bool(match_wavegrams)
        self.ifos = ifos
        self.wavegram_time_bins = wavegram_time_bins

    @staticmethod
    def event_order(clustered, ifos):
        """Return the stable event order used by prepared node arrays."""
        return [(ifo, int(label))
                for ifo in ifos
                for label in clustered[ifo]["cluster_id"].astype(int)]

    def _stack(self, clustered, coefficients, ifos):
        """Every event's map, in the detector order given.

        :return: tuple -- (frames, grids, grid_shape, bin_seconds).
        """
        frames, grids, clouds = [], [], []
        grid_shape = (0, self.wavegram_time_bins)
        # Like the width, the duration of a column comes from the grids that
        # arrived: the detector stage measures it from the data the search was run on.
        bin_seconds = 1.0
        for ifo in ifos:
            per_cluster = coefficients[ifo]
            events = clustered[ifo].reset_index(drop=True)
            missing = set(events["cluster_id"].astype(int)) - set(per_cluster)
            if missing:
                raise KeyError(
                    f"no coefficients for cluster {min(missing)} of {ifo}; the graph is "
                    "built from the wavelet coefficients, so every event needs them"
                )
            frames.append(events.assign(ifo=ifo))
            for label in events["cluster_id"].astype(int):
                rendered = per_cluster[label]
                grid = np.asarray(rendered.wavegram(self.wavegram_time_bins))
                grid_shape = grid.shape
                bin_seconds = getattr(rendered, "bin_seconds", bin_seconds)
                grids.append(grid.ravel())
                clouds.append(getattr(rendered, "tiles", None))
        return frames, grids, grid_shape, bin_seconds, clouds

    def build(self, clustered: dict[str, pd.DataFrame],
              coefficients: dict[str, dict],
              comparison: dict[str, dict] | None = None) -> TriggerGraph:
        """Assemble the graph from each detector's events and their coefficients.

        Two renderings of the same events answer two different questions and
        want opposite resolutions. The map that assembles an event needs wide
        columns and a long span, so that windows seconds apart describe one
        transient. The map two detectors are compared on needs columns finer
        than the light travel time, or a real delay moves no cell and no
        comparison --- computed here or learned downstream --- can see it.

        :type clustered: dict[str, pandas.DataFrame]
        :param clustered: ``{ifo: event catalogue}``, each row carrying at
            least `cluster_id` and `gpsPeak`.
        :type coefficients: dict[str, dict]
        :param coefficients: ``{ifo: {cluster_id: EventWavegram}}``, the
            assembly map. Every event must have an entry.
        :type comparison: dict[str, dict] | None
        :param comparison: the same events rendered for the cross-detector
            comparison; the assembly map when None.
        :return: TriggerGraph
        :raises KeyError: if an event has no coefficients.
        """
        return self.build_from_prepared(
            clustered, self.prepare(clustered, coefficients, comparison))

    def prepare(self, clustered: dict[str, pd.DataFrame],
                coefficients: dict[str, dict],
                comparison: dict[str, dict] | None = None) -> dict:
        """Everything about the nodes that does not depend on when they happened.

        A time slide moves a detector's events, so it changes which pairs are
        admissible; it changes nothing about what those events are. Their maps,
        their feature vectors, their unit-norm shapes, their energy and their
        bands are properties of the coefficients, so they are computed once here
        and reused by every `build_from_prepared` that follows. That is what
        keeps a background estimate proportional to the number of slides instead
        of to the number of slides times the number of events.

        :type clustered: dict[str, pandas.DataFrame]
        :param clustered: ``{ifo: event catalogue}``, carrying `cluster_id`.
        :type coefficients: dict[str, dict]
        :param coefficients: ``{ifo: {cluster_id: EventWavegram}}``, the
            assembly map. Every event must have an entry.
        :type comparison: dict[str, dict] | None
        :param comparison: the same events rendered for the cross-detector
            comparison; the assembly map when None.
        :return: dict -- the node-side arrays, to be given to
            `build_from_prepared` together with the events they describe.
        :raises KeyError: if an event has no coefficients.
        """
        ifos = self.ifos or list(clustered.keys())
        frames, grids, grid_shape, bin_seconds, clouds = self._stack(
            clustered, coefficients, ifos)
        if comparison is None:
            fine, fine_shape, fine_bin = grids, grid_shape, bin_seconds
            comparison_clouds = clouds
        else:
            _, fine, fine_shape, fine_bin, comparison_clouds = self._stack(
                clustered, comparison, ifos)
        nodes_df = pd.concat(frames, ignore_index=True)
        first_rendered = next((comparison[ifo][int(label)]
                       for ifo in ifos
                       for label in clustered[ifo]["cluster_id"]), None) \
            if comparison is not None else next((coefficients[ifo][int(label)]
                              for ifo in ifos
                              for label in clustered[ifo]["cluster_id"]), None)

        # Coefficient/sigma spans decades in both directions. Signed log1p
        # compresses the magnitude, preserves polarity and leaves empty cells
        # exactly zero.
        wavegrams = (_signed_log1p(np.vstack(grids))
                     if grids else np.zeros((0, 1)))
        onehot = (pd.get_dummies(nodes_df["ifo"])
                  .reindex(columns=ifos, fill_value=0).to_numpy(dtype=float))
        X = np.hstack([wavegrams, onehot])
        X = X.astype(np.float32)

        # Unit-norm compact grids give a local shape summary rather than an
        # amplitude comparison: the same signal reaches two detectors with
        # amplitudes set by their antenna responses. They do not replace the
        # matcher, which works on the tiles and an admitted displacement.
        raw = np.vstack(fine) if fine else np.zeros((0, 1))
        compressed = _signed_log1p(raw)
        norms = np.linalg.norm(compressed, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        shapes = compressed / norms
        # The width comes from the grids that arrived, not from this builder's
        # own constant: a multi-window event is rendered on the shared band grid
        # by its own level, which need not have chosen the same number of bins.
        energy = _coefficient_energy(nodes_df)
        # The support is what decides admissibility, and it is wide by
        # construction, so its overlap is near one for almost every admitted
        # pair and carries little as a ranking quantity. The band the energy
        # occupies varies from pair to pair, and does.
        band_lo = _numeric(nodes_df, ("freqQ05", "freqMin"), default=np.nan)
        band_hi = _numeric(nodes_df, ("freqQ95", "freqMax"), default=np.nan)
        spread = _numeric(nodes_df, ("tSpread",), default=0.0)
        spread = np.where(np.isfinite(spread), spread, 0.0)
        if first_rendered is not None and hasattr(first_rendered, "bands"):
            profile_bands = np.asarray(first_rendered.bands, dtype=float)
        else:
            pairs = [(lo, hi) for cloud in comparison_clouds if cloud is not None
                     for lo, hi in zip(cloud[2], cloud[3])]
            profile_bands = (np.unique(np.asarray(pairs, dtype=float), axis=0)
                             if pairs else np.zeros((1, 2), dtype=float))

        # The bin the two detectors are compared on is fixed by what the
        # comparison has to resolve, not by the column width of whichever
        # rendering happened to arrive. Two conditions bound it from opposite
        # sides and both are physical. A bin coarser than a tile adds the
        # signed coefficients of one detector to each other before either
        # detector is compared with the other, so an oscillating transient
        # cancels against itself and the highest bands, whose tiles are
        # shortest, lose most --- a frequency prior the search does not carry.
        # A bin coarser than the light travel time cannot represent the delay
        # the coincidence exists to measure, so the lag axis collapses onto
        # zero whatever the tolerance allows. The shortest tile the ladder
        # holds is one over the upper edge of its highest band; where that is
        # longer than the travel time, the travel time is what binds.
        upper = float(np.max(profile_bands[:, 1])) if len(profile_bands) else 0.0
        travel = self.coincidence.travel_time(tuple(ifos))
        bounds = [b for b in (BIN_PER_TILE / upper if upper > 0.0 else 0.0,
                              travel) if b > 0.0]
        profile_bin = min(bounds) if bounds else fine_bin
        profile_refs = nodes_df["gpsPeak"].to_numpy(dtype=float)
        # Each event on a window of its own, centred on its own instant and no
        # wider than the event reaches. What two events are compared on is then
        # a property of that pair and of nothing else. Rendering every event on
        # the widest event of the run would make a transient lasting minutes
        # fix the grid of every millisecond-long one, and what may turn up in a
        # run is not known in advance: at a bin of a millisecond that array
        # does not exist.
        profile_extent = np.zeros(len(comparison_clouds))
        profile_half = np.zeros(len(comparison_clouds), dtype=np.int64)
        profile_windows = []
        empty = tuple(np.zeros(0) for _ in range(6))
        # The windows are the only expensive thing here: one rendering per
        # event, on the finest bin the ladder holds. A builder that will not
        # compare them has no reason to build them, so a study that does not
        # read the comparison does not pay for it in time or in memory. The
        # extents are kept either way --- they cost a minimum and a maximum
        # over tiles already in hand, and they describe the events.
        blank = render(empty, profile_bands, 0.0, profile_bin, profile_bin)
        for at, (cloud, reference) in enumerate(zip(comparison_clouds, profile_refs)):
            if cloud is None or not len(cloud[5]):
                profile_windows.append(blank)
                continue
            shifted = [np.asarray(field, dtype=float).copy() for field in cloud]
            shifted[0] -= reference
            shifted[1] -= reference
            # The event's own maximum extension in time, from the tiles it is
            # made of. It is how far the two detectors' renderings of one
            # transient can be displaced and still describe it, which is not
            # the same question as how far apart the arrival times may be.
            profile_extent[at] = shifted[1].max() - shifted[0].min()
            # Symmetric about the instant, so that the middle bin holds it
            # whatever side of it the event's tiles fall on and one number
            # places the window.
            if not self.match_wavegrams:
                profile_windows.append(blank)
                continue
            half = int(np.ceil(max(abs(float(shifted[0].min())),
                                   abs(float(shifted[1].max()))) / profile_bin))
            profile_half[at] = half
            profile_windows.append(render(
                tuple(shifted), profile_bands, -half * profile_bin,
                (half + 1) * profile_bin, profile_bin))

        return dict(
            ifos=ifos, node_features=X, shapes=shapes, raw=raw, clouds=clouds,
            cloud_flat=flatten_clouds(clouds),
            profile_clouds=comparison_clouds,
            profile_bands=profile_bands,
            profile_windows=profile_windows,
            profile_half=profile_half,
            profile_bin=profile_bin,
            profile_extent=profile_extent,
            profile_refs=profile_refs,
            energy=energy, band_lo=band_lo, band_hi=band_hi, spread=spread,
            fine_bin=fine_bin, raw_maps=raw.reshape(len(raw), *fine_shape),
            gps_peak=nodes_df["gpsPeak"].to_numpy(dtype=float),
            # The order the arrays are in, so that a later call cannot silently
            # index them with a different event set.
            order=[(ifo, int(label))
                   for ifo in ifos
                   for label in clustered[ifo]["cluster_id"].astype(int)])

    def build_from_prepared(self, clustered: dict[str, pd.DataFrame],
                            prepared: dict, with_neighbours: bool = True) -> TriggerGraph:
        """The graph, from events whose node-side quantities are already known.

        Only the times are read from `clustered`: everything else comes from
        `prepared`. The two must therefore describe the same events in the same
        order, which is checked rather than assumed --- indexing the prepared
        arrays with a different event set would attach one event's morphology to
        another's time and produce a plausible, wrong graph.

        :type clustered: dict[str, pandas.DataFrame]
        :param clustered: ``{ifo: event catalogue}``, at the times to use.
        :type prepared: dict
        :param prepared: what `prepare` returned for these events.
        :return: TriggerGraph
        :raises ValueError: if the events are not the ones that were prepared.
        """
        ifos = prepared["ifos"]
        order = [(ifo, int(label))
                 for ifo in ifos
                 for label in clustered[ifo]["cluster_id"].astype(int)]
        if order != prepared["order"]:
            raise ValueError(
                "the events given are not the ones prepared: the node-side "
                "arrays are indexed by position, so they cannot be reused for "
                "a different event set or a different order")

        X = prepared["node_features"]
        shapes, raw, clouds = prepared["shapes"], prepared["raw"], prepared["clouds"]
        profile_windows = prepared["profile_windows"]
        profile_half = prepared["profile_half"]
        profile_bands = prepared["profile_bands"]

        profile_bands = prepared["profile_bands"]
        energy = prepared["energy"]
        band_lo, band_hi = prepared["band_lo"], prepared["band_hi"]
        spread = prepared["spread"]
        fine_bin = prepared["fine_bin"]
        # The comparison bin, which is the profile's alone: `fine_bin` is the
        # width of the compact node grid and says nothing about what the two
        # detectors can be told apart by.
        profile_bin = prepared["profile_bin"]
        # Prepared alongside the clouds; an older prepared dict without it is
        # flattened here once rather than refused.
        cloud_flat = prepared.get("cloud_flat")
        if cloud_flat is None:
            cloud_flat = flatten_clouds(clouds)

        nodes_df = pd.concat([clustered[ifo].reset_index(drop=True).assign(ifo=ifo)
                              for ifo in ifos], ignore_index=True)
        displacement = (nodes_df["gpsPeak"].to_numpy(dtype=float)
                - prepared["gps_peak"])
        idx_by_ifo = {ifo: nodes_df.index[nodes_df["ifo"] == ifo].to_numpy() for ifo in ifos}
        gps = nodes_df["gpsPeak"].to_numpy(dtype=float)

        # Only the pairs inside the window are ever formed, by searching a
        # sorted time axis. A pairwise-difference matrix asks the same question
        # in O(n^2) memory, which a search carrying no per-detector threshold
        # exhausts: 10^5 events in one detector is 10^10 matrix entries.
        sorted_by_ifo = {
            ifo: idxs[np.argsort(gps[idxs], kind="mergesort")]
            for ifo, idxs in idx_by_ifo.items()
        }

        intra_edges, intra_feats = [], []
        for idxs in sorted_by_ifo.values() if with_neighbours else []:
            for left, right in neighbour_pairs(gps[idxs], self.intra_ifo_window_s):
                i_sel, j_sel = idxs[left], idxs[right]
                dt_sel = gps[i_sel] - gps[j_sel]
                intra_edges.append(np.column_stack([i_sel, j_sel]))
                intra_edges.append(np.column_stack([j_sel, i_sel]))
                intra_feats.append(dt_sel[:, None])
                intra_feats.append(-dt_sel[:, None])

        # The physically possible pairs, from the classical finder's own rule:
        # the graph ranks the survivors of known physics rather than rediscovering
        # that two events a second apart cannot be one gravitational wave.
        finder = IndexedCoincidenceFinder(self.coincidence)
        cross_edges, cross_feats, cross_profiles = [], [], []
        cross_match, cross_dt, cross_measured = [], [], []
        profile_lags = np.zeros(0, dtype=float)
        for ifo_a, ifo_b in combinations(ifos, 2):
            idx_a, idx_b = idx_by_ifo[ifo_a], idx_by_ifo[ifo_b]
            admissible = finder.candidate_edges(
                nodes_df.iloc[idx_a].reset_index(drop=True),
                nodes_df.iloc[idx_b].reset_index(drop=True))
            if not admissible:
                continue
            local = np.array([(i, j, dt, f_overlap, t_overlap)
                              for i, j, _, dt, f_overlap, t_overlap in admissible])
            i_sel = idx_a[local[:, 0].astype(int)]
            j_sel = idx_b[local[:, 1].astype(int)]
            # These fixed-size grids are compact node views anchored within
            # each event. Their dot products are retained as local shape
            # summaries, not presented as the physical time-of-flight match.
            # The latter is computed on the tile supports, at one lag for the
            # whole plane, with the two anchors' difference carried alongside
            # so that the displacement it reports is an absolute one.
            # The morphology at the resolution the transform has, with no grid
            # in between: the two events' own coefficients, paired where their
            # tiles cover the same place.
            travel = self.coincidence.travel_time((ifo_a, ifo_b))
            coherence = tile_coherence_many(
                cloud_flat, i_sel, j_sel, travel, displacement=displacement)
            tolerance = self.coincidence.timing_tolerance(
                spread[i_sel], spread[j_sel], (ifo_a, ifo_b))
            # The wavegram match is a comparison of two morphologies at a
            # displacement, and it is asked only of pairs that are already
            # coincident: the events' own instants within the tolerance the
            # geometry and their own timing spreads allow. A pair whose
            # instants are further apart than that is admitted --- a transient
            # longer than one analysis window is assembled as several events
            # and the two detectors need not keep the same one --- but no
            # displacement the tolerance permits brings its two instants
            # together, so what a search over those displacements would find is
            # the agreement between the tail of one event and the head of the
            # other, at the price of a trials factor, and not what the
            # statistic means. It is reported as no agreement, and the pair is
            # ranked on the statistics that do not require a displacement.
            coincident = np.abs(local[:, 2]) <= tolerance
            matched = np.flatnonzero(coincident)
            # The displacements searched are the tolerance's. Each map is laid
            # on its own event's instant and the whole bins of the difference
            # between the two instants are applied as a shift of the map, at no
            # cost whatever that difference is, so the alignment where the same
            # part of one transient meets itself --- which is at an
            # arrival-time difference no larger than the light travel time --- is
            # already reachable. Searching further would only find agreements
            # at displacements the geometry forbids, and would cost the square
            # of the transient's own length.
            carry = int(np.ceil(
                (self.coincidence.maximum_tolerance((ifo_a, ifo_b))
                 + 0.5 * profile_bin) / profile_bin))
            profile_lags = np.arange(-carry, carry + 1, dtype=float) * profile_bin
            profiles = np.zeros((len(i_sel), len(profile_bands), len(profile_lags)))
            edge_match = np.zeros(len(i_sel))
            edge_dt = np.full(len(i_sel), np.nan)
            edge_measured = np.zeros(len(i_sel), dtype=bool)
            if self.match_wavegrams:
                found, found_match, found_dt, found_measured = compare_on_pair_grids(
                    profile_windows, profile_half, profile_bin,
                    i_sel[matched], j_sel[matched], tolerance[matched],
                    gps[i_sel[matched]] - gps[j_sel[matched]], profile_lags)
                profiles[matched] = found
                edge_match[matched] = found_match
                edge_dt[matched] = found_dt
                edge_measured[matched] = found_measured

            # Gathered a block at a time, and on a GPU when there is one. The
            # plain form --- `shapes[i_sel]` inside an einsum --- builds one
            # copy of a map per pair before reducing it: at a few million pairs
            # that is tens of gigabytes held to produce one number each, which
            # is what makes this stage the memory wall of the run.
            signed_similarity = paired_dot(shapes, shapes, i_sel, j_sel)
            similarity = np.abs(signed_similarity)
            signed_coherent = paired_dot(raw, raw, i_sel, j_sel)
            coherent = np.abs(signed_coherent)
            overlap = np.sqrt(coherent)
            present = (paired_dot(raw, raw, i_sel, i_sel)
                       + paired_dot(raw, raw, j_sel, j_sel))
            # Taken once, where the comparison took it, and carried: a
            # second maximum over the part of the profile that was kept would
            # be a different quantity wearing the same name.
            correlation = np.clip(edge_match, 0.0, 1.0)
            cross_edges.append(np.column_stack([i_sel, j_sel]))
            cross_profiles.append(profiles)
            cross_match.append(edge_match)
            cross_dt.append(edge_dt)
            cross_measured.append(edge_measured)
            cross_feats.append(np.column_stack([
                local[:, 2],                                   # dt
                similarity,                                    # polarity-free local shape
                local[:, 3],                                   # frequency overlap
                local[:, 4],                                   # time overlap
                np.log(np.maximum(energy[i_sel], EPS)
                       / np.maximum(energy[j_sel], EPS)),      # log energy ratio
                overlap,                                       # coherent energy
                _overlap_fraction(band_lo[i_sel], band_hi[i_sel],
                                  band_lo[j_sel], band_hi[j_sel]),
                # How far apart they are relative to what this pair may claim:
                # a long event is allowed a wide tolerance, and should not be
                # ranked as if it had used none of it.
                np.abs(local[:, 2]) / np.maximum(
                    self.coincidence.timing_tolerance(spread[i_sel], spread[j_sel],
                                                      (ifo_a, ifo_b)),
                    EPS),
                correlation,
                overlap * correlation,
                # The coherent energy is signed, both polarities being
                # physical, and what the pair is worth is how much of it
                # there is: the feature is the coherent amplitude, the
                # root of the magnitude. A root of the signed value is
                # not a number wherever the two detectors' responses
                # oppose, which is half the sky.
                np.sqrt(np.abs(coherence)),
            ]))

        intra_edges = (np.concatenate(intra_edges) if intra_edges
                       else np.zeros((0, 2), dtype=np.int64))
        intra_feats = (np.concatenate(intra_feats) if intra_feats
                       else np.zeros((0, 1), dtype=np.float32))
        cross_edges = (np.concatenate(cross_edges) if cross_edges
                       else np.zeros((0, 2), dtype=np.int64))
        cross_feats = (np.concatenate(cross_feats) if cross_feats
                       else np.zeros((0, N_EDGE_FEATURES), dtype=np.float32))
        n_bands = profile_windows[0].shape[0] if profile_windows else 1
        cross_profiles = (np.concatenate(cross_profiles) if cross_profiles
                          else np.zeros((0, n_bands, len(profile_lags)), dtype=np.float32))
        cross_match = (np.concatenate(cross_match) if cross_match
                       else np.zeros(0, dtype=float))
        cross_dt = (np.concatenate(cross_dt) if cross_dt
                    else np.zeros(0, dtype=float))
        cross_measured = (np.concatenate(cross_measured) if cross_measured
                          else np.zeros(0, dtype=bool))

        return TriggerGraph(
            nodes=nodes_df,
            node_features=X,
            intra_edges=intra_edges.astype(np.int64).reshape(-1, 2),
            intra_edge_features=intra_feats.astype(np.float32).reshape(-1, 1),
            cross_edges=cross_edges.astype(np.int64).reshape(-1, 2),
            cross_edge_features=cross_feats.astype(np.float32).reshape(-1, N_EDGE_FEATURES),
            cross_edge_profiles=cross_profiles.astype(np.float32),
            cross_edge_lags=profile_lags, cross_edge_match=cross_match,
            cross_edge_match_dt=cross_dt, cross_edge_measured=cross_measured,
            ifos=ifos,
        )




def edge_labels_from_injections(graph, injection_times, window_s: float = 0.5):
    """Label each candidate edge by whether both its events are one injection.

    A pair is a positive when the two detectors' events each cover the same
    injection, not when the pair's mean time happens to land near one. The
    weaker rule admits a noise event that merely sits close to a real signal in
    the other detector, and a model trained on it learns temporal proximity to
    an injection rather than coherence between two views of one signal --- which
    is the thing the network stage exists to measure.

    :param graph: a `TriggerGraph`.
    :param injection_times: GPS times of the injected signals.
    :type window_s: float
    :param window_s: how far outside its own extent an event still covers an
        injection, seconds.
    :return: numpy.ndarray -- 1.0 where both events cover the same injection.
    """
    from wdf.analysis.injections import candidate_spans

    if not len(graph.cross_edges):
        return np.zeros(0)

    times = np.sort(np.asarray(injection_times, dtype=float))
    owner = np.full(len(graph.nodes), -1)
    if len(times):
        start, end = candidate_spans(graph.nodes, candidate_time="gpsPeak")
        # The nearest injection is the only one an event's own extent can cover.
        slot = np.clip(np.searchsorted(times, start), 0, len(times) - 1)
        for candidate in (slot, np.maximum(slot - 1, 0)):
            covers = ((times[candidate] >= start - window_s)
                      & (times[candidate] <= end + window_s))
            owner = np.where(covers & (owner < 0), candidate, owner)

    i, j = graph.cross_edges[:, 0], graph.cross_edges[:, 1]
    return ((owner[i] >= 0) & (owner[i] == owner[j])).astype(float)


class WavegramCoincidenceFinder:
    """A finder whose candidates carry what the wavegrams say about them.

    `IndexedCoincidenceFinder` admits pairs and describes them with times,
    bands and energies; the agreement between two events' wavegrams is added by
    the graph, which needs their coefficient maps. Anything that ranks a
    candidate on that agreement therefore has to see the graph's table, and so
    does the background it is calibrated against --- a statistic measured on
    the foreground and absent from the background cannot be given a rate.

    This wraps the two so that one object produces the same table for both. It
    satisfies the interface `TimeSlideFAR` expects, so the accidental
    population is built the same way as the zero-lag one rather than by a
    second route that would have to be kept in step.

    The compact node-side quantities are prepared once and reused for every
    slide because a slide changes no coefficient. An absolute-time wavegram
    comparison is different: it must apply the slide displacement to the tile
    times before searching the admitted global lag. Original absolute GPS tile
    coordinates must never be reused unchanged as if they had been slid.

    :param finder: the finder deciding which pairs are admissible.
    :param builder: the builder rendering the admitted pairs as a graph.
    :param coefficients: `{ifo: {cluster_id: EventWavegram}}`, the assembly map.
    :param comparison: the same events rendered for the cross-detector
        comparison; the assembly map when None.
    :param events: the events the maps belong to, used to prepare the node-side
        arrays once. None prepares them on the first call instead.
    :param scorer: a ranking that reads the graph, or None for the graph's own
        columns alone. It is applied here, where the graph still exists: the
        table this returns keeps no reference to it, so a caller cannot attach
        a learned ranking afterwards. A background built without the scorer the
        foreground was read with carries no rate for that ranking.
    """

    def __init__(self, finder, builder, coefficients, comparison=None,
                 events=None, prepared=None, scorer=None):
        self.finder = finder
        self.scorer = scorer
        self.builder = builder
        self.coefficients = coefficients
        self.comparison = comparison
        if prepared is not None and events is not None:
            raise ValueError("pass either prepared or events, not both")
        self._prepared = (prepared if prepared is not None else
                          None if events is None else
                          builder.prepare(events, coefficients, comparison))
        self._prepared_order = None if prepared is None else prepared["order"]

    def find(self, events_by_ifo) -> pd.DataFrame:
        """The admitted pairs, described by the graph.

        :type events_by_ifo: dict
        :param events_by_ifo: `{ifo: event catalogue}`.
        :return: pandas.DataFrame -- `TriggerGraph.candidate_table`'s schema,
            widened by whatever the scorer adds when there is one; empty when
            no pair is admitted.
        """
        ifos = list(events_by_ifo)
        if any(events_by_ifo[ifo].empty for ifo in ifos):
            return pd.DataFrame()
        if self._prepared is None:
            self._prepared = self.builder.prepare(
                events_by_ifo,
                {ifo: self.coefficients[ifo] for ifo in ifos},
                comparison=None if self.comparison is None
                else {ifo: self.comparison[ifo] for ifo in ifos})
        elif self.builder.event_order(events_by_ifo, ifos) != self._prepared["order"]:
            raise ValueError("prepare a finder for the set of events being scored")
        graph = self.builder.build_from_prepared(events_by_ifo, self._prepared)
        if not len(graph.cross_edges):
            return pd.DataFrame()
        # Scored where the graph is still in hand. A ranking absent from part
        # of a background is a rate for a population that was never measured,
        # so the scorer applies to every pair set this forms or to none.
        return (graph.candidate_table() if self.scorer is None
                else self.scorer.score(graph))
 