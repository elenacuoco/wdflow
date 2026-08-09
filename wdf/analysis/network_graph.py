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

from wdf.analysis.detector_graph import WAVEGRAM_TIME_BINS
from wdf.analysis.pairs import neighbour_pairs
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
# The grid's width is level one's, imported rather than restated: two constants
# for the shape of one object drift apart.

# What a candidate edge carries: the arrival-time difference, the agreement
# between the two wavegrams, the shared fraction of band and of time support,
# and the log ratio of the two energies. The ratio is a feature and not a
# penalty: the antenna responses make the same signal reach two detectors with
# amplitudes differing by a factor of a few, so an unequal pair is physical.
EDGE_FEATURES = ["dt_s", "wavegram_similarity", "frequency_overlap",
                 "time_overlap", "log_energy_ratio",
                 "wavegram_overlap", "energy_band_overlap", "dt_over_tolerance",
                 "network_correlation", "coherent_statistic"]

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
        gps = self.nodes["gpsCentroid"].to_numpy(dtype=float) \
            if "gpsCentroid" in self.nodes else self.nodes["gpsPeak"].to_numpy(dtype=float)
        enwdf = self.nodes["EnWDF"].to_numpy(dtype=float)
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
        frames, grids = [], []
        grid_shape = (0, self.wavegram_time_bins)
        # Like the width, the duration of a column comes from the grids that
        # arrived: level one measures it from the data the search was run on.
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
        return frames, grids, grid_shape, bin_seconds

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
        ifos = self.ifos or list(clustered.keys())
        frames, grids, grid_shape, bin_seconds = self._stack(
            clustered, coefficients, ifos)
        if comparison is None:
            fine, fine_shape, fine_bin = grids, grid_shape, bin_seconds
        else:
            _, fine, fine_shape, fine_bin = self._stack(clustered, comparison, ifos)
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
        panels = shapes.reshape(len(shapes), *fine_shape) if shapes.size else \
            shapes.reshape(len(shapes), 0, self.wavegram_time_bins)
        energy_panels = raw.reshape(len(raw), *fine_shape) if raw.size else \
            raw.reshape(len(raw), 0, self.wavegram_time_bins)

        energy = _coefficient_energy(nodes_df)
        # The support is what decides admissibility, and it is wide by
        # construction: on three hours of the simulated set its overlap has a
        # median of exactly one, so as a ranking quantity it carries nothing.
        # The band the energy occupies varies from pair to pair.
        band_lo = _numeric(nodes_df, ("freqQ05", "freqMin"), default=np.nan)
        band_hi = _numeric(nodes_df, ("freqQ95", "freqMax"), default=np.nan)
        spread = _numeric(nodes_df, ("tSpread",), default=0.0)
        spread = np.where(np.isfinite(spread), spread, 0.0)
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
            similarity = np.einsum("ij,ij->i", shapes[i_sel], shapes[j_sel])
            coherent = np.maximum(np.einsum("ij,ij->i", raw[i_sel], raw[j_sel]), 0.0)
            overlap = np.sqrt(coherent)
            present = (np.einsum("ij,ij->i", raw[i_sel], raw[i_sel])
                       + np.einsum("ij,ij->i", raw[j_sel], raw[j_sel]))
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
        start, end = candidate_spans(
            graph.nodes,
            candidate_time="gpsCentroid" if "gpsCentroid" in graph.nodes else "gpsPeak")
        # The nearest injection is the only one an event's own extent can cover.
        slot = np.clip(np.searchsorted(times, start), 0, len(times) - 1)
        for candidate in (slot, np.maximum(slot - 1, 0)):
            covers = ((times[candidate] >= start - window_s)
                      & (times[candidate] <= end + window_s))
            owner = np.where(covers & (owner < 0), candidate, owner)

    i, j = graph.cross_edges[:, 0], graph.cross_edges[:, 1]
    return ((owner[i] >= 0) & (owner[i] == owner[j])).astype(float)
