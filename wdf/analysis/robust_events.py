 
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


EPS = np.finfo(float).tiny


def _first_existing(frame: pd.DataFrame, names: Iterable[str], default=None):
    for name in names:
        if name in frame.columns:
            return frame[name]
    if default is None:
        raise KeyError(f"None of the columns {tuple(names)} is available")
    return pd.Series(default, index=frame.index)


def _numeric(frame: pd.DataFrame, names: Iterable[str], default=np.nan) -> np.ndarray:
    return pd.to_numeric(
        _first_existing(frame, names, default=default),
        errors="coerce",
    ).to_numpy(dtype=float)


def clean_triggers_robust(
    triggers: pd.DataFrame,
    segment_bounds: tuple[float, float] | None = None,
    edge_margin_s: float = 0.0,
) -> pd.DataFrame:
    """Remove only non-finite, non-physical and segment-edge triggers.

    No fixed EnWDF/SNR ceiling is applied: loud events are retained.
    """
    if triggers is None or len(triggers) == 0:
        return pd.DataFrame() if triggers is None else triggers.copy()

    out = triggers.copy()
    time = _numeric(out, ("gpsPeak", "gpsMax", "gps"))
    rank = _numeric(out, ("EnWDF", "mSNR"))
    sigma = _numeric(out, ("sigmaWin", "sigma", "mSigma"), default=1.0)

    keep = (
        np.isfinite(time)
        & np.isfinite(rank)
        & np.isfinite(sigma)
        & (rank >= 0.0)
        & (sigma > 0.0)
    )

    for low_name, high_name in (
        ("freqMin", "freqMax"),
        ("gpsStart", "gpsEnd"),
    ):
        if low_name in out and high_name in out:
            low = pd.to_numeric(out[low_name], errors="coerce").to_numpy(float)
            high = pd.to_numeric(out[high_name], errors="coerce").to_numpy(float)
            keep &= np.isfinite(low) & np.isfinite(high) & (high >= low)

    if segment_bounds is not None:
        start, end = map(float, segment_bounds)
        keep &= (
            (time >= start + float(edge_margin_s))
            & (time <= end - float(edge_margin_s))
        )

    out = out.loc[keep].copy()
    out["trigger_index"] = out.index.to_numpy()
    return out.reset_index(drop=True)


def stride_seconds(parameters) -> float:
    """Real spacing between consecutive WDF windows after resampling."""
    window = float(parameters.window)
    overlap = float(parameters.overlap)
    sampling = getattr(parameters, "resampling", None)
    if sampling is None:
        sampling = float(parameters.sampling) / float(parameters.ResamplingFactor)
    sampling = float(sampling)
    if sampling <= 0 or window <= overlap:
        raise ValueError("Invalid WDF window/overlap/resampling parameters")
    return (window - overlap) / sampling


def _frequency_interval(frame: pd.DataFrame):
    fmin = _numeric(frame, ("freqMin", "freqPeak", "freqMean"), default=0.0)
    fmax = _numeric(frame, ("freqMax", "freqPeak", "freqMean"), default=0.0)
    fmean = _numeric(frame, ("freqMean", "freqPeak"), default=0.5 * (fmin + fmax))
    fmin, fmax = np.minimum(fmin, fmax), np.maximum(fmin, fmax)
    return fmin, fmean, fmax


def _coefficient_energy(frame: pd.DataFrame) -> np.ndarray:
    wt = sorted(
        (
            c for c in frame.columns
            if c.startswith("wt") and c[2:].isdigit()
        ),
        key=lambda c: int(c[2:]),
    )
    if wt:
        values = frame[wt].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, 0.0)
        return np.sum(values * values, axis=1)
    rank = _numeric(frame, ("EnWDF", "mSNR"), default=0.0)
    return rank * rank


def _overlap_fraction(a0, a1, b0, b1):
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    width = max(min(a1 - a0, b1 - b0), EPS)
    return overlap / width


class _UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return int(x)

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class ClusterConfig:
    max_missing_windows: int = 1
    minimum_frequency_overlap: float = 0.0
    maximum_log_energy_jump: float = 3.0
    edge_margin_s: float = 0.0


def cluster_detector_triggers(
    triggers: pd.DataFrame,
    parameters,
    config: ClusterConfig | None = None,
    segment_bounds: tuple[float, float] | None = None,
):
    """Graph clustering of one detector's WDF triggers.

    Edges require temporal adjacency tied to the real WDF stride, compatible
    frequency intervals and no discontinuous coefficient-energy jump.
    """
    config = ClusterConfig() if config is None else config
    cleaned = clean_triggers_robust(
        triggers,
        segment_bounds=segment_bounds,
        edge_margin_s=config.edge_margin_s,
    )

    if cleaned.empty:
        return cleaned.assign(cluster_id=pd.Series(dtype=int)), pd.DataFrame()

    time = _numeric(cleaned, ("gpsPeak", "gpsMax", "gps"))
    order = np.argsort(time, kind="mergesort")
    cleaned = cleaned.iloc[order].reset_index(drop=True)
    time = time[order]

    fmin, fmean, fmax = _frequency_interval(cleaned)
    energy = np.maximum(_coefficient_energy(cleaned), EPS)
    rank = _numeric(cleaned, ("EnWDF", "mSNR"), default=0.0)

    stride = stride_seconds(parameters)
    time_eps = stride * (1 + int(config.max_missing_windows))
    uf = _UnionFind(len(cleaned))

    for i in range(len(cleaned)):
        stop = np.searchsorted(time, time[i] + time_eps, side="right")
        for j in range(i + 1, stop):
            freq_ok = (
                _overlap_fraction(fmin[i], fmax[i], fmin[j], fmax[j])
                >= config.minimum_frequency_overlap
            )
            energy_ok = (
                abs(np.log(energy[i] / energy[j]))
                <= config.maximum_log_energy_jump
            )
            if freq_ok and energy_ok:
                uf.union(i, j)

    roots = np.array([uf.find(i) for i in range(len(cleaned))])
    _, cluster_ids = np.unique(roots, return_inverse=True)
    cleaned["cluster_id"] = cluster_ids

    rows = []
    for cluster_id, group in cleaned.groupby("cluster_id", sort=True):
        idx = group.index.to_numpy()
        group_energy = energy[idx]
        group_rank = rank[idx]
        weights = np.maximum(group_energy, EPS)

        gps_start = float(
            pd.to_numeric(
                _first_existing(group, ("gpsStart", "gps"), default=time[idx]),
                errors="coerce",
            ).min()
        )
        if "gpsEnd" in group:
            gps_end = float(pd.to_numeric(group["gpsEnd"], errors="coerce").max())
        elif "duration" in group:
            gps_end = float(
                np.nanmax(
                    _numeric(group, ("gpsStart", "gps"), default=time[idx])
                    + _numeric(group, ("duration",), default=0.0)
                )
            )
        else:
            gps_end = float(np.max(time[idx]) + float(parameters.window) / float(parameters.resampling))

        peak_local = int(np.argmax(group_rank))
        peak_row = group.iloc[peak_local]

        # Coincidence time: use the actual time of the loudest WDF member.
        # The energy centroid is retained only as a diagnostic because it can
        # move by hundreds of milliseconds in long or contaminated clusters.
        peak_time = float(time[idx][peak_local])
        gps_energy_centroid = float(
            np.average(time[idx], weights=weights)
        )

        member_indices = tuple(int(v) for v in group["trigger_index"].to_numpy())
        n_triggers = len(group)

        # Primary cluster ranking: do not add EnWDF values in quadrature across
        # overlapping WDF windows, because that double-counts common samples.
        enwdf_peak = float(np.max(group_rank))
        enwdf_quadrature_sum = float(
            np.sqrt(np.sum(group_rank * group_rank))
        )

        member_snr_mean = _numeric(
            group,
            ("snrMean",),
            default=0.0,
        )
        member_snr_peak = _numeric(
            group,
            ("snrPeak",),
            default=0.0,
        )
        member_sigma = _numeric(
            group,
            ("sigmaWin", "sigma", "mSigma"),
            default=np.nan,
        )

        snr_mean_cluster = float(
            np.average(member_snr_mean, weights=weights)
        )
        snr_peak_cluster = float(
            np.nanmax(member_snr_peak)
        )

        finite_sigma = np.isfinite(member_sigma) & (member_sigma > 0.0)
        sigma_cluster = (
            float(np.average(member_sigma[finite_sigma], weights=weights[finite_sigma]))
            if np.any(finite_sigma)
            else np.nan
        )

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "ifo": peak_row.get("ifo", getattr(parameters, "itf", "")),
                "gps": gps_start,
                "gpsStart": gps_start,
                "gpsEnd": gps_end,
                "gpsMax": peak_time,
                "gpsPeak": peak_time,
                "gpsEnergyCentroid": gps_energy_centroid,
                "duration": max(0.0, gps_end - gps_start),
                "gps_span_s": max(0.0, gps_end - gps_start),
                "freqMin": float(np.min(fmin[idx])),
                "freqMean": float(np.average(fmean[idx], weights=weights)),
                "freqMax": float(np.max(fmax[idx])),
                "freqPeak": float(peak_row.get("freqPeak", fmean[idx][peak_local])),
                "EnWDF": enwdf_peak,
                "cluster_sum_enwdf": enwdf_quadrature_sum,
                "snrMean": snr_mean_cluster,
                "snrPeak": snr_peak_cluster,
                "sigmaWin": sigma_cluster,
                "coefficient_energy": float(np.sum(group_energy)),
                "n_triggers": int(n_triggers),
                "singleton": bool(n_triggers == 1),
                "member_indices": member_indices,
                "wave": peak_row.get("wave", peak_row.get("mWave", "")),
            }
        )

    return cleaned, pd.DataFrame(rows)


def select_events_for_coincidence(
    events: pd.DataFrame,
    minimum_enwdf: float | None = None,
) -> pd.DataFrame:
    """Every event goes to coincidence, whatever its number of windows.

    Clustering exists to assemble a transient that spans several windows. A
    transient short enough to fall inside one window is a single-window event,
    and it already passed the search threshold -- its multiplicity is a
    statement about its duration, not about whether it is real. Deciding what
    is a physical signal is the cross-detector coincidence's job, and the
    time-slide background measures exactly how often single-window events
    coincide by accident.

    :type events: pandas.DataFrame
    :param events: one detector's event catalogue.
    :type minimum_enwdf: float or None
    :param minimum_enwdf: optional floor on the statistic, for the case where
        the coincidence stage has to be run on a reduced catalogue for cost
        reasons. None keeps everything.
    :return: pandas.DataFrame -- the events to send to coincidence.
    """
    if events.empty:
        return events.copy()
    if minimum_enwdf is None:
        return events.reset_index(drop=True)
    rank = _numeric(events, ("EnWDF",))
    return events.loc[rank >= float(minimum_enwdf)].reset_index(drop=True)


@dataclass
class CoincidenceConfig:
    light_travel_time_s: float = 0.01001
    timing_jitter_s: float = 0.05
    minimum_frequency_overlap: float = 0.0
    time_weight: float = 1.0
    frequency_weight: float = 1.0
    # Amplitude ratio is not a shape mismatch: the same signal reaches two
    # detectors with amplitudes set by their antenna responses, routinely
    # differing by a factor of a few. Penalizing it makes the assignment prefer
    # a quieter, better-matched pair over the louder true one, so it is off by
    # default; the frequency term already carries the shape test.
    morphology_weight: float = 0.0

    @property
    def window_s(self):
        return self.light_travel_time_s + 2.0 * self.timing_jitter_s


class IndexedCoincidenceFinder:
    """Indexed one-to-one H1-L1 coincidence.

    Candidate intervals are obtained with searchsorted. Ambiguous local
    bipartite components are resolved by minimum-cost one-to-one assignment.
    """

    def __init__(self, config: CoincidenceConfig | None = None, **kwargs):
        if config is None:
            config = CoincidenceConfig(**kwargs)
        self.config = config

    def coincidence_window(self, *_):
        return self.config.window_s

    def _candidate_edges(self, left, right):
        lt = _numeric(left, ("gpsPeak",))
        rt = _numeric(right, ("gpsPeak",))
        lf0, lfm, lf1 = _frequency_interval(left)
        rf0, rfm, rf1 = _frequency_interval(right)
        le = np.maximum(_coefficient_energy(left), EPS)
        re = np.maximum(_coefficient_energy(right), EPS)

        right_order = np.argsort(rt, kind="mergesort")
        rt_sorted = rt[right_order]
        edges = []

        for i, t in enumerate(lt):
            lo = np.searchsorted(rt_sorted, t - self.config.window_s, side="left")
            hi = np.searchsorted(rt_sorted, t + self.config.window_s, side="right")
            for position in range(lo, hi):
                j = int(right_order[position])
                overlap = _overlap_fraction(lf0[i], lf1[i], rf0[j], rf1[j])
                if overlap < self.config.minimum_frequency_overlap:
                    continue
                dt_cost = abs(t - rt[j]) / max(self.config.window_s, EPS)
                scale = max(lf1[i] - lf0[i], rf1[j] - rf0[j], 1.0)
                df_cost = abs(lfm[i] - rfm[j]) / scale
                morphology_cost = abs(np.log(le[i] / re[j]))
                cost = (
                    self.config.time_weight * dt_cost
                    + self.config.frequency_weight * df_cost
                    + self.config.morphology_weight * morphology_cost
                )
                edges.append((i, j, cost, t - rt[j], overlap))
        return edges

    @staticmethod
    def _components(n_left, n_right, edges):
        uf = _UnionFind(n_left + n_right)
        for i, j, *_ in edges:
            uf.union(i, n_left + j)
        by_root = {}
        for edge in edges:
            root = uf.find(edge[0])
            by_root.setdefault(root, []).append(edge)
        return list(by_root.values())

    def find(self, events_by_ifo):
        ifos = list(events_by_ifo)
        if len(ifos) != 2:
            raise ValueError("IndexedCoincidenceFinder.find requires exactly two IFOs")
        left_ifo, right_ifo = ifos
        left = events_by_ifo[left_ifo].reset_index(drop=True)
        right = events_by_ifo[right_ifo].reset_index(drop=True)
        if left.empty or right.empty:
            return pd.DataFrame()

        edges = self._candidate_edges(left, right)
        selected = []

        for component in self._components(len(left), len(right), edges):
            left_ids = sorted({edge[0] for edge in component})
            right_ids = sorted({edge[1] for edge in component})
            li = {value: index for index, value in enumerate(left_ids)}
            rj = {value: index for index, value in enumerate(right_ids)}
            cost = np.full((len(left_ids), len(right_ids)), np.inf)
            metadata = {}

            for i, j, value, dt, overlap in component:
                if value < cost[li[i], rj[j]]:
                    cost[li[i], rj[j]] = value
                    metadata[(i, j)] = (dt, overlap)

            finite = np.isfinite(cost)
            if not finite.any():
                continue

            penalty = float(np.nanmax(cost[finite]) + 1e6)
            row_index, col_index = linear_sum_assignment(np.where(finite, cost, penalty))

            for a, b in zip(row_index, col_index):
                if not finite[a, b]:
                    continue
                i, j = left_ids[a], right_ids[b]
                dt, overlap = metadata[(i, j)]
                selected.append((i, j, cost[a, b], dt, overlap))

        rows = []
        for i, j, cost, dt, overlap in selected:
            a, b = left.iloc[i], right.iloc[j]
            sa = float(a.get("EnWDF", a.get("snrMax", 0.0)))
            sb = float(b.get("EnWDF", b.get("snrMax", 0.0)))
            ta = float(a.get("gpsMax", a.get("gpsPeak")))
            tb = float(b.get("gpsMax", b.get("gpsPeak")))
            rows.append(
                {
                    f"index_{left_ifo}": int(i),
                    f"index_{right_ifo}": int(j),
                    f"EnWDF_{left_ifo}": sa,
                    f"EnWDF_{right_ifo}": sb,
                    f"snr_{left_ifo}": sa,
                    f"snr_{right_ifo}": sb,
                    "gps_candidate": 0.5 * (ta + tb),
                    "delta_t": float(dt),
                    "dt_s": float(dt),
                    "frequency_overlap": float(overlap),
                    "coincidence_cost": float(cost),
                    "network_enwdf": float(np.hypot(sa, sb)),
                    "network_min_enwdf": float(min(sa, sb)),
                    "network_geometric_enwdf": float(np.sqrt(max(sa * sb, 0.0))),
                    "network_snr": float(np.hypot(sa, sb)),
                    "ifos_involved": f"{left_ifo},{right_ifo}",
                    "n_ifos": 2,
                }
            )

        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows).sort_values(
            ["network_enwdf", "coincidence_cost"],
            ascending=[False, True],
        ).reset_index(drop=True)
        out.insert(0, "candidate_id", np.arange(len(out), dtype=int))
        return out

    def find_network(self, events_by_ifo, min_ifos=2):
        pair_results = []
        for a, b in combinations(events_by_ifo, 2):
            result = self.find({a: events_by_ifo[a], b: events_by_ifo[b]})
            if len(result):
                pair_results.append(result)
        if not pair_results:
            return pd.DataFrame()
        return pd.concat(pair_results, ignore_index=True)


@dataclass
class FARConfig:
    n_slides: int = 100
    min_shift_s: float = 2.0
    seed: int = 1


class TimeSlideFAR:
    """Time-slide background and FAR ranking.

    FAR(score) = N_background(>=score) / total_slide_livetime.
    FAP = 1 - exp(-FAR * foreground_duration).
    """

    def __init__(self, coincidence_finder, config: FARConfig | None = None, **kwargs):
        self.coincidence_finder = coincidence_finder
        self.config = FARConfig(**kwargs) if config is None else config
        self.n_slides = self.config.n_slides

    def background_distribution(self, events_by_ifo, segment_bounds):
        ifos = list(events_by_ifo)
        if len(ifos) != 2:
            raise ValueError("TimeSlideFAR currently requires exactly two IFOs")
        ref, shifted_ifo = ifos
        start, end = map(float, segment_bounds[shifted_ifo])
        span = end - start
        if span <= 2.0 * self.config.min_shift_s:
            raise ValueError("Segment is too short for the requested time slides")

        rng = np.random.default_rng(self.config.seed)
        rows = []
        shifts = []

        for slide_index in range(self.config.n_slides):
            magnitude = rng.uniform(self.config.min_shift_s, span - self.config.min_shift_s)
            shift = magnitude if rng.integers(0, 2) else -magnitude
            shifts.append(shift)

            shifted = events_by_ifo[shifted_ifo].copy()
            for column in ("gpsMax", "gpsPeak", "gpsStart", "gpsEnd"):
                if column in shifted:
                    shifted[column] = start + (
                        (pd.to_numeric(shifted[column], errors="coerce") - start + shift)
                        % span
                    )

            candidates = self.coincidence_finder.find(
                {ref: events_by_ifo[ref], shifted_ifo: shifted}
            )
            if len(candidates):
                candidates = candidates.copy()
                candidates["slide_index"] = slide_index
                candidates["slide_shift_s"] = shift
                rows.append(candidates)

        background = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        background.attrs["n_slides"] = self.config.n_slides
        background.attrs["segment_duration_s"] = span
        background.attrs["total_livetime_s"] = self.config.n_slides * span
        background.attrs["shifts_s"] = tuple(shifts)
        return background

    def rank_candidates(
        self,
        candidates,
        background,
        segment_duration_s,
        score_column="network_min_enwdf",
    ):
        out = candidates.copy()
        if out.empty:
            return out

        if score_column not in out:
            score_column = "network_enwdf" if "network_enwdf" in out else "network_snr"

        if background.empty:
            out["n_background_ge"] = 0
            total_livetime = self.config.n_slides * float(segment_duration_s)
            out["background_livetime_s"] = total_livetime
            out["far_hz"] = 1.0 / max(total_livetime, EPS)
            out["far_per_day"] = out["far_hz"] * 86400.0
            out["fap"] = -np.expm1(-out["far_hz"] * float(segment_duration_s))
            return out

        bg_score_column = score_column if score_column in background else "network_snr"
        bg = np.sort(background[bg_score_column].to_numpy(dtype=float))
        scores = out[score_column].to_numpy(dtype=float)
        n_ge = bg.size - np.searchsorted(bg, scores, side="left")

        total_livetime = float(
            background.attrs.get(
                "total_livetime_s",
                self.config.n_slides * float(segment_duration_s),
            )
        )
        # A finite background cannot establish a zero rate.  The +1 count is
        # the conservative resolution of the measured time-slide livetime.
        far_hz = (n_ge + 1) / max(total_livetime, EPS)

        out["n_background_ge"] = n_ge
        out["background_livetime_s"] = total_livetime
        out["far_hz"] = far_hz
        out["far_per_day"] = far_hz * 86400.0
        out["fap"] = -np.expm1(-far_hz * float(segment_duration_s))

        return out.sort_values(
            ["far_hz", score_column],
            ascending=[True, False],
        ).reset_index(drop=True)

    def false_alarm_probability(
        self,
        candidate,
        background,
        segment_duration_s,
        score_column="network_min_enwdf",
    ):
        ranked = self.rank_candidates(
            pd.DataFrame([candidate]),
            background,
            segment_duration_s,
            score_column=score_column,
        )
        return {
            "far_hz": float(ranked.iloc[0]["far_hz"]),
            "far_per_day": float(ranked.iloc[0]["far_per_day"]),
            "fap": float(ranked.iloc[0]["fap"]),
            "n_background_ge": int(ranked.iloc[0]["n_background_ge"]),
        }


# Backward-compatible names used by the notebooks.
CoincidenceFinder = IndexedCoincidenceFinder
BackgroundEstimator = TimeSlideFAR
