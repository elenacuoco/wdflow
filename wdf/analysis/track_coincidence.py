"""Stages 3 and 4 of the track path: the classical finder's admitted pairs of
track events, described by their tracks and their waveforms.

`TrackCoincidenceFinder` is the track path's counterpart of
`wdf.analysis.network_graph.WavegramCoincidenceFinder`. The pairs are the
ones `IndexedCoincidenceFinder.candidate_edge_array` admits, the rule every
coincidence of the package starts from; each is described by
`TRACK_EDGE_FEATURES`:

- **track compatibility** (`track_compatibility`): the two tracks' spans
  overlapping once one may shift by the light travel time plus
  `TrackTimingJitter`, and their f(t) agreeing there within
  `TrackFrequencyTolerance` octaves;
- **waveform coherence** (`waveform_coherence`): the complex
  cross-correlation of the two reconstructed waveforms' analytic signals,
  normalised by their energies, over the lags the light travel time plus the
  jitter allows: the lag of its largest modulus, that modulus (the coherence,
  0 to 1) and its phase.

The table carries the classical finder's columns (`dt_s`,
`frequency_overlap`, `time_overlap`, `coincidence_cost`, `network_enwdf`,
`network_min_enwdf`) beside them, so the classical rankings apply. A time
slide (`TimeSlideFAR`) moves an event's times; the finder reads each event's
displacement from its `gpsPeak` against the unslid events it was built with
and moves the event's track and waveform by it.

With a scorer (`GNNCoincidenceScorer`), the pairs are those of a
`TriggerGraphBuilder` graph of the track events, whose node maps are drawn
from the waveforms (`track_event_coefficients`), and the track features are
appended to the graph's edge features: the model must have been fitted with
``cross_edge_dim = N_EDGE_FEATURES + len(TRACK_EDGE_FEATURES)``.
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

#: The features a track pair adds to the classical ones.
TRACK_EDGE_FEATURES = ["track_time_overlap", "track_f_distance", "track_f_overlap",
                       "track_compatible", "track_lag_s", "track_coherence", "track_phase"]


@dataclass
class TrackCoincidenceSettings:
    """What makes two tracks compatible.

    :param jitter_s: timing jitter added to the light travel time, s
        (`TrackTimingJitter`).
    :param frequency_tolerance: largest ``|d log2 f|`` between the two tracks'
        f(t), octaves (`TrackFrequencyTolerance`).
    :param light_travel_time_s: overrides the pair's light travel time, s
        (`TrackLightTravelTime`); None reads it from the detectors.
    """

    jitter_s: float = 0.0035
    frequency_tolerance: float = 0.5
    light_travel_time_s: float = None

    KEYS = dict(jitter_s="TrackTimingJitter", frequency_tolerance="TrackFrequencyTolerance",
                light_travel_time_s="TrackLightTravelTime")

    @classmethod
    def from_parameters(cls, par):
        """The settings a run asks for, defaults filled in and written back."""
        out = cls()
        for name, key in cls.KEYS.items():
            value = getattr(par, key, None)
            if value is not None:
                setattr(out, name, float(value))
            setattr(par, key, getattr(out, name))
        return out

    def window(self, a, b):
        """Largest lag between detectors `a` and `b`, s."""
        from wdf.analysis.detectors import light_travel_time

        travel = (self.light_travel_time_s if self.light_travel_time_s is not None
                  else light_travel_time(a, b))
        return float(travel) + float(self.jitter_s)


def waveform_coherence(first, second, fs, max_lag_s):
    """The complex cross-correlation of two waveforms over a bounded lag.

    ``G(tau) = sum_t A(t) conj(B(t - tau)) / sqrt(sum |A|^2 sum |B|^2)``, A
    and B the analytic signals of the two series placed on one absolute grid.
    The lag is that of the largest ``|G|`` within ``|tau| <= max_lag_s``,
    refined by a parabola through it and its neighbours; positive when the
    first series arrives after the second, as `dt_s`.

    :type first: tuple
    :param first: (GPS of the first sample, samples).
    :type second: tuple
    :param second: the same, for the other detector.
    :type fs: float
    :param fs: sampling rate of both, Hz.
    :type max_lag_s: float
    :param max_lag_s: half-width of the lag search, s.
    :return: tuple -- (lag s, coherence ``|G|``, phase rad); NaN for an empty
        or silent series.
    """
    from scipy.signal import fftconvolve, hilbert

    (t1, a), (t2, b) = first, second
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    nan = (float("nan"),) * 3
    if not a.size or not b.size:
        return nan
    ea, eb = float(np.sum(a ** 2)), float(np.sum(b ** 2))
    if ea <= 0 or eb <= 0:
        return nan
    A, B = hilbert(a), hilbert(b)
    norm = np.sqrt(np.sum(np.abs(A) ** 2) * np.sum(np.abs(B) ** 2))
    # full correlation: index m <-> sample shift (m - (b.size - 1)) of A against B
    corr = fftconvolve(A, np.conj(B[::-1]), mode="full") / norm
    shift = np.arange(corr.size) - (b.size - 1)
    lag = (t1 - t2) + shift / fs
    inside = np.abs(lag) <= max_lag_s + 0.5 / fs
    if not inside.any():
        return nan
    idx = np.flatnonzero(inside)
    k = idx[int(np.argmax(np.abs(corr[idx])))]
    value = corr[k]
    refined = lag[k]
    if 0 < k < corr.size - 1:
        y0, y1, y2 = np.abs(corr[k - 1]), np.abs(corr[k]), np.abs(corr[k + 1])
        den = y0 - 2 * y1 + y2
        if den < 0:
            refined = lag[k] + 0.5 * (y0 - y2) / den / fs
    return float(refined), float(np.abs(value)), float(np.angle(value))


def track_compatibility(left, right, displacement_left, displacement_right, window_s,
                        tolerance):
    """How two tracks agree in time and in f(t).

    :param left: an event row with `track_t_start`, `track_t_end`,
        `track_times`, `track_f`.
    :param right: the other.
    :param displacement_left: seconds the left event was moved by (slides).
    :param displacement_right: the same for the right one.
    :type window_s: float
    :param window_s: light travel time plus jitter, s.
    :type tolerance: float
    :param tolerance: largest ``|d log2 f|``, octaves.
    :return: dict -- `track_time_overlap` (the shared span over the shorter
        track), `track_f_distance` (median ``|d log2 f|`` over it, octaves; the
        smallest over the shifts -window, 0, +window), `track_f_overlap` (the
        fraction of the shared slots within the tolerance) and
        `track_compatible`.
    """
    s1 = float(left["track_t_start"]) + displacement_left
    e1 = float(left["track_t_end"]) + displacement_left
    s2 = float(right["track_t_start"]) + displacement_right
    e2 = float(right["track_t_end"]) + displacement_right
    shared = min(e1, e2 + window_s) - max(s1, s2 - window_s)
    shortest = max(min(e1 - s1, e2 - s2), 1e-12)
    out = dict(track_time_overlap=float(np.clip(shared / shortest, 0.0, 1.0)),
               track_f_distance=float("nan"), track_f_overlap=0.0, track_compatible=0.0)
    if shared <= 0:
        return out
    t1 = np.asarray(left["track_times"], dtype=float) + displacement_left
    f1 = np.log2(np.asarray(left["track_f"], dtype=float))
    t2 = np.asarray(right["track_times"], dtype=float) + displacement_right
    f2 = np.log2(np.asarray(right["track_f"], dtype=float))
    best = None
    for shift in (-window_s, 0.0, window_s):
        here = (t1 >= t2[0] + shift) & (t1 <= t2[-1] + shift)
        if not here.any():
            continue
        d = np.abs(f1[here] - np.interp(t1[here] - shift, t2, f2))
        if best is None or np.median(d) < np.median(best):
            best = d
    if best is None:
        return out
    out["track_f_distance"] = float(np.median(best))
    out["track_f_overlap"] = float(np.mean(best <= tolerance))
    out["track_compatible"] = float(out["track_f_distance"] <= tolerance)
    return out


def track_event_coefficients(events, waveforms, fs, window, basis="LocalCos64",
                             bin_seconds=None):
    """Node maps of track events, drawn from their waveforms, for the graph.

    Each waveform is cut into windows of `window` samples, stride ``window /
    2``, taken to `basis` (NumPy local cosine); the coefficients of each
    window's central half become one trigger-like row (`wt_index`,
    `wt_value`, ``sigma`` 1: the waveform is in noise units), and
    `event_coefficients` assembles the rows of one event.

    :return: dict -- ``{cluster_id: EventWavegram}``.
    """
    from wdf.analysis.coefficients import from_dense
    from wdf.analysis.detector_graph import DetectorGraph, event_coefficients
    from wdf.analysis.local_cosine import local_cosine
    from wdf.analysis.wavelets import _local_cosine_segment, coeff_time_bounds

    m = _local_cosine_segment(basis)
    stride = window // 2
    t_lo, t_hi = coeff_time_bounds(window, fs, basis)
    centre = 0.5 * (t_lo + t_hi) * fs
    central = (centre >= window // 4) & (centre < window - window // 4)
    rows, labels = [], []
    ids = events["cluster_id"].astype(int).tolist()
    for label, cid in enumerate(ids):
        t0, y = waveforms.get(cid, (np.nan, np.zeros(0)))
        y = np.asarray(y, dtype=float)
        pad = np.concatenate([np.zeros(window // 4), y, np.zeros(window)])
        made = 0
        for a in range(0, max(pad.size - window + 1, 1), stride):
            seg = pad[a:a + window]
            if seg.size < window:
                break
            c = np.where(central, local_cosine(seg[None, :], m)[0], 0.0)
            index, value = from_dense(c)
            if not len(index) and made:
                continue
            gps = t0 + (a - window // 4) / fs
            peak = gps + float(centre[int(np.argmax(np.abs(c)))]) / fs
            rows.append(dict(gps=gps, gpsPeak=peak, n_coeff=window, fs=fs, sigma=1.0, wave=basis,
                             EnWDF=float(np.sqrt(np.sum(c ** 2))), wt_index=index, wt_value=value))
            labels.append(label)
            made += 1
    if not rows:
        return {}
    nodes = pd.DataFrame(rows)
    graph = DetectorGraph(nodes, np.zeros((len(nodes), 1)), np.zeros((0, 2), dtype=int),
                          np.zeros((0, 9)))
    maps = event_coefficients(graph, labels=np.asarray(labels), bin_seconds=bin_seconds)
    return {ids[label]: m_ for label, m_ in maps.items()}


class TrackCoincidenceFinder:
    """Admitted pairs of track events, with their track and waveform features.

    It satisfies the interface `TimeSlideFAR` expects (`find`, and the
    coincidence configuration as `config`).

    :param finder: the classical finder (`IndexedCoincidenceFinder`); its
        admission rule decides which pairs exist.
    :param events: ``{ifo: track event table}`` (`track_events`), unslid;
        each event's slide displacement is read against these.
    :param waveforms: ``{ifo: {cluster_id: (t0, waveform)}}``.
    :param fs: sampling rate of the waveforms, Hz.
    :type settings: TrackCoincidenceSettings or None
    :param settings: jitter, frequency tolerance.
    :param scorer: a learned ranking (`GNNCoincidenceScorer`) or None.
    :param builder: the `TriggerGraphBuilder` the scorer's graph is built
        with (scorer only).
    :param coefficients: ``{ifo: {cluster_id: EventWavegram}}`` of the track
        events (`track_event_coefficients`), scorer only.
    :param comparison: the comparison maps, scorer only (None: the assembly
        maps).
    """

    def __init__(self, finder, events, waveforms, fs, settings=None, scorer=None, builder=None,
                 coefficients=None, comparison=None):
        self.finder = finder
        self.config = finder.config
        self.settings = TrackCoincidenceSettings() if settings is None else settings
        self.fs = float(fs)
        self.waveforms = waveforms
        self.scorer, self.builder = scorer, builder
        self.coefficients, self.comparison = coefficients, comparison
        if scorer is not None and (builder is None or coefficients is None):
            raise ValueError("a scorer needs the graph builder and the track events' maps")
        self.reference = {ifo: dict(zip(ev["cluster_id"].astype(int),
                                        pd.to_numeric(ev["gpsPeak"]).to_numpy(float)))
                          for ifo, ev in events.items()}
        self._prepared = None

    def _features(self, ifo_a, row_a, ifo_b, row_b):
        ca, cb = int(row_a["cluster_id"]), int(row_b["cluster_id"])
        da = float(row_a["gpsPeak"]) - self.reference[ifo_a][ca]
        db = float(row_b["gpsPeak"]) - self.reference[ifo_b][cb]
        da, db = (0.0 if not np.isfinite(da) else da), (0.0 if not np.isfinite(db) else db)
        window = self.settings.window(ifo_a, ifo_b)
        out = track_compatibility(row_a, row_b, da, db, window, self.settings.frequency_tolerance)
        wa = self.waveforms.get(ifo_a, {}).get(ca)
        wb = self.waveforms.get(ifo_b, {}).get(cb)
        if wa is None or wb is None:
            lag = coherence = phase = float("nan")
        else:
            lag, coherence, phase = waveform_coherence((wa[0] + da, wa[1]), (wb[0] + db, wb[1]),
                                                       self.fs, window)
        out.update(track_lag_s=lag, track_coherence=coherence, track_phase=phase)
        return out

    def _classic(self, events_by_ifo):
        from wdf.analysis.robust_events import INSTANT_COLUMNS, _numeric

        blocks = []
        for a, b in combinations(list(events_by_ifo), 2):
            left = events_by_ifo[a].reset_index(drop=True).assign(ifo=a)
            right = events_by_ifo[b].reset_index(drop=True).assign(ifo=b)
            if left.empty or right.empty:
                continue
            rows = self.finder.candidate_edge_array(left, right)
            if not len(rows):
                continue
            i, j = rows[:, 0].astype(int), rows[:, 1].astype(int)
            col = "EnWDF_window" if "EnWDF_window" in left else "EnWDF"
            sa = pd.to_numeric(left[col]).to_numpy(float)[i]
            sb = pd.to_numeric(right[col]).to_numpy(float)[j]
            ta, tb = _numeric(left, INSTANT_COLUMNS)[i], _numeric(right, INSTANT_COLUMNS)[j]
            table = pd.DataFrame(dict(
                gps_candidate=0.5 * (ta + tb), ifos_involved=f"{a},{b}", n_ifos=2,
                ifo_i=a, ifo_j=b, cluster_i=left["cluster_id"].to_numpy(int)[i],
                cluster_j=right["cluster_id"].to_numpy(int)[j],
                network_enwdf=np.hypot(sa, sb), network_min_enwdf=np.minimum(sa, sb),
                dt_s=rows[:, 3], frequency_overlap=rows[:, 4], time_overlap=rows[:, 5],
                coincidence_cost=rows[:, 2]))
            feats = [self._features(a, left.iloc[p], b, right.iloc[q]) for p, q in zip(i, j)]
            blocks.append(pd.concat([table, pd.DataFrame(feats, columns=TRACK_EDGE_FEATURES)],
                                    axis=1))
        if not blocks:
            return pd.DataFrame()
        out = pd.concat(blocks, ignore_index=True)
        out.insert(0, "candidate_id", np.arange(len(out)))
        return out

    def _scored(self, events_by_ifo):
        from wdf.analysis.network_graph import N_EDGE_FEATURES, TriggerGraph

        ifos = list(events_by_ifo)
        if self._prepared is None:
            self._prepared = self.builder.prepare(
                events_by_ifo, {i: self.coefficients[i] for i in ifos},
                None if self.comparison is None else {i: self.comparison[i] for i in ifos})
        graph = self.builder.build_from_prepared(events_by_ifo, self._prepared)
        if not len(graph.cross_edges):
            return pd.DataFrame()
        nodes = graph.nodes
        feats = []
        for p, q in graph.cross_edges:
            ra, rb = nodes.iloc[int(p)], nodes.iloc[int(q)]
            feats.append(self._features(ra["ifo"], ra, rb["ifo"], rb))
        extra = pd.DataFrame(feats, columns=TRACK_EDGE_FEATURES)
        width = N_EDGE_FEATURES + len(TRACK_EDGE_FEATURES)
        model = int(self.scorer.edge_head[0].in_features - 2 * self.scorer.encoder[0].out_features
                    - self.scorer.profile_dim)
        if model != width:
            raise ValueError(f"the GNN reads {model} edge features; the track path gives "
                             f"{width} (the graph's {N_EDGE_FEATURES} and "
                             f"{len(TRACK_EDGE_FEATURES)} track features): fit a model with "
                             f"cross_edge_dim={width}")
        wide = TriggerGraph(
            graph.nodes, graph.node_features, graph.intra_edges, graph.intra_edge_features,
            graph.cross_edges,
            np.hstack([graph.cross_edge_features,
                       np.nan_to_num(extra.to_numpy(float)).astype(np.float32)]),
            graph.ifos, graph.cross_edge_profiles, graph.cross_edge_lags, graph.cross_edge_match,
            graph.cross_edge_match_dt, graph.cross_edge_measured)
        table = self.scorer.score(wide)
        for side in ("i", "j"):
            table[f"ifo_{side}"] = nodes["ifo"].to_numpy()[table[f"node_{side}"].astype(int)]
            table[f"cluster_{side}"] = nodes["cluster_id"].to_numpy()[table[f"node_{side}"].astype(int)]
        return pd.concat([table.reset_index(drop=True), extra], axis=1)

    def find(self, events_by_ifo):
        """The admitted pairs with their track features.

        :type events_by_ifo: dict
        :param events_by_ifo: ``{ifo: track event table}``, possibly slid.
        :return: pandas.DataFrame -- one row per admitted pair; empty when
            none is.
        """
        if any(ev.empty for ev in events_by_ifo.values()):
            return pd.DataFrame()
        return self._classic(events_by_ifo) if self.scorer is None else self._scored(events_by_ifo)
