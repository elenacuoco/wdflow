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
from wdf.analysis.robust_events import (
    EPS,
    _numeric,
    _overlap_fraction,
    CoincidenceConfig,
    IndexedCoincidenceFinder,
    _coefficient_energy,
)


# A node is described by the wavelet coefficients of its cluster, rendered on a
# fixed octave-by-time grid, not by scalar summaries. Scoring a coincidence on
# peak time, peak frequency and the statistic is what the classical finder
# already does; the coefficients carry the transient's time-frequency pattern,
# which is the information a learned combiner can use and a time window cannot.
# The grid's width is the detector stage's, imported rather than restated: two constants
# for the shape of one object drift apart.

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
#     cc = 2 <w1, w2> / (<w1, w1> + <w2, w2>),
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

# A real signal reaches the two detectors up to the light travel time apart, so
# two maps anchored each at its own event's time are offset by that much. No
# alignment is searched for here and none is precomputed as a feature: on a map
# whose columns are finer than the light travel time the offset is visible in
# the maps themselves, and the pair carries its own dt, so the correspondence
# between the two morphologies in time is left for the network to learn.
N_EDGE_FEATURES = len(EDGE_FEATURES)


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
        """Cross-detector candidate edges as a table.

        Shares its schema with `IndexedCoincidenceFinder.find`'s output, so the
        classical and the learned statistic can be ranked and compared through
        the same machinery. Every edge feature is carried through, which is what
        a deterministic baseline is fitted on.

        :return: pandas.DataFrame -- one row per candidate edge.
        """
        i, j = self.cross_edges[:, 0], self.cross_edges[:, 1]
        # The same clock the pair was admitted on, and the one `dt_s` is
        # measured with. The energy centroid is a property of how much of the
        # transient each detector recovered, so two detectors seeing one source
        # at different amplitudes place their centroids differently and that
        # difference lands in the arrival-time difference, which is the whole of
        # the coincidence and of the sky position.
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

        # How loud the pair is, weighted by how alike the two grids are. The
        # weight carries no amplitude of its own -- the cosine is invariant to
        # scale -- so the loudness is not counted twice, and that invariance is
        # what keeps the antenna responses from penalising a real signal seen
        # unequally in the two detectors, which normalising by amplitude does.
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
    the classical finder admits -- within the light travel time plus the events'
    own timing spreads, overlapping in band, overlapping in time once the
    travel time is allowed for -- so that the learned and the classical
    statistic rank the same candidate set and can be compared at a fixed false
    alarm rate. What the graph decides is which of those survivors are coherent,
    not which pairs are geometrically possible: that is known physics, and it is
    imposed rather than learned.

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
        self.ifos = ifos
        self.wavegram_time_bins = wavegram_time_bins

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
        else:
            _, fine, fine_shape, fine_bin, _ = self._stack(
                clustered, comparison, ifos)
        nodes_df = pd.concat(frames, ignore_index=True)

        # |coefficient|/sigma spans decades, so compress before standardising;
        # log1p leaves the empty cells of the grid at exactly zero.
        wavegrams = np.log1p(np.vstack(grids)) if grids else np.zeros((0, 1))
        onehot = pd.get_dummies(nodes_df["ifo"]).reindex(columns=ifos, fill_value=0).to_numpy(dtype=float)
        X = np.hstack([wavegrams, onehot])
        X = X.astype(np.float32)

        # Unit-norm grids, so the similarity between two of them is a shape
        # comparison and not an amplitude one: the same signal reaches two
        # detectors with amplitudes set by their antenna responses.
        # The comparison is made on the fine rendering, where the light travel
        # time is several columns wide.
        raw = np.vstack(fine) if fine else np.zeros((0, 1))
        compressed = np.log1p(raw)
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
        return dict(
            ifos=ifos, node_features=X, shapes=shapes, raw=raw, clouds=clouds,
            cloud_flat=flatten_clouds(clouds),
            energy=energy, band_lo=band_lo, band_hi=band_hi, spread=spread,
            fine_bin=fine_bin,
            # The order the arrays are in, so that a later call cannot silently
            # index them with a different event set.
            order=[(ifo, int(label))
                   for ifo in ifos
                   for label in clustered[ifo]["cluster_id"].astype(int)])

    def build_from_prepared(self, clustered: dict[str, pd.DataFrame],
                            prepared: dict) -> TriggerGraph:
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
        energy = prepared["energy"]
        band_lo, band_hi = prepared["band_lo"], prepared["band_hi"]
        spread = prepared["spread"]
        fine_bin = prepared["fine_bin"]
        # Prepared alongside the clouds; an older prepared dict without it is
        # flattened here once rather than refused.
        cloud_flat = prepared.get("cloud_flat")
        if cloud_flat is None:
            cloud_flat = flatten_clouds(clouds)

        nodes_df = pd.concat([clustered[ifo].reset_index(drop=True).assign(ifo=ifo)
                              for ifo in ifos], ignore_index=True)
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
        for idxs in sorted_by_ifo.values():
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
        cross_edges, cross_feats = [], []
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
            # No alignment is applied and none is precomputed. The maps are
            # rendered finely enough that a real delay moves several columns,
            # and the pair carries its own dt, so how the two morphologies
            # correspond in time is left for the network to learn rather than
            # answered here.
            # The morphology at the resolution the transform has, with no grid
            # in between: the two events' own coefficients, paired where their
            # tiles cover the same place.
            travel = self.coincidence.travel_time((ifo_a, ifo_b))
            coherence = tile_coherence_many(cloud_flat, i_sel, j_sel, travel)

            # Gathered a block at a time, and on a GPU when there is one. The
            # plain form --- `shapes[i_sel]` inside an einsum --- builds one
            # copy of a map per pair before reducing it: at a few million pairs
            # that is tens of gigabytes held to produce one number each, which
            # is what makes this stage the memory wall of the run.
            similarity = paired_dot(shapes, shapes, i_sel, j_sel)
            coherent = np.maximum(paired_dot(raw, raw, i_sel, j_sel), 0.0)
            overlap = np.sqrt(coherent)
            present = (paired_dot(raw, raw, i_sel, i_sel)
                       + paired_dot(raw, raw, j_sel, j_sel))
            correlation = np.clip(2.0 * coherent / np.maximum(present, EPS),
                                  0.0, 1.0)
            cross_edges.append(np.column_stack([i_sel, j_sel]))
            cross_feats.append(np.column_stack([
                local[:, 2],                                   # dt
                similarity,                                    # agreement at zero lag
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

        intra_edges = np.concatenate(intra_edges) if intra_edges else np.zeros((0, 2), dtype=np.int64)
        intra_feats = np.concatenate(intra_feats) if intra_feats else np.zeros((0, 1), dtype=np.float32)
        cross_edges = np.concatenate(cross_edges) if cross_edges else np.zeros((0, 2), dtype=np.int64)
        cross_feats = np.concatenate(cross_feats) if cross_feats else np.zeros((0, N_EDGE_FEATURES), dtype=np.float32)

        return TriggerGraph(
            nodes=nodes_df,
            node_features=X,
            intra_edges=intra_edges.astype(np.int64).reshape(-1, 2),
            intra_edge_features=intra_feats.astype(np.float32).reshape(-1, 1),
            cross_edges=cross_edges.astype(np.int64).reshape(-1, 2),
            cross_edge_features=cross_feats.astype(np.float32).reshape(-1, N_EDGE_FEATURES),
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

    The node-side quantities are prepared once and reused for every slide. A
    slide moves an event in time and does not change its coefficients, and the
    map two detectors are compared on is centred on each event's own energy
    rather than on an absolute time, so nothing prepared depends on the shift.
    Rebuilding them per slide would make the cost of a background estimate grow
    as the number of slides times the number of events.

    :param finder: the finder deciding which pairs are admissible.
    :param builder: the builder rendering the admitted pairs as a graph.
    :param coefficients: `{ifo: {cluster_id: EventWavegram}}`, the assembly map.
    :param comparison: the same events rendered for the cross-detector
        comparison; the assembly map when None.
    :param events: the events the maps belong to, used to prepare the node-side
        arrays once. None prepares them on the first call instead.
    """

    def __init__(self, finder, builder, coefficients, comparison=None,
                 events=None):
        self.finder = finder
        self.builder = builder
        self.coefficients = coefficients
        self.comparison = comparison
        self._prepared = (None if events is None
                          else builder.prepare(events, coefficients, comparison))

    def find(self, events_by_ifo) -> pd.DataFrame:
        """The admitted pairs, described by the graph.

        :type events_by_ifo: dict
        :param events_by_ifo: `{ifo: event catalogue}`.
        :return: pandas.DataFrame -- `TriggerGraph.candidate_table`'s schema;
            empty when no pair is admitted.
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
        graph = self.builder.build_from_prepared(events_by_ifo, self._prepared)
        if not len(graph.cross_edges):
            return pd.DataFrame()
        return graph.candidate_table()
