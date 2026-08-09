 
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from wdf.analysis.detectors import network_light_travel_time
from wdf.analysis.pairs import cross_pairs, neighbour_pairs

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
    fmin = _numeric(frame, ("freqMin", "freqMean"), default=0.0)
    fmax = _numeric(frame, ("freqMax", "freqMean"), default=0.0)
    fmean = _numeric(frame, ("freqMean",), default=0.5 * (fmin + fmax))
    fmin, fmax = np.minimum(fmin, fmax), np.maximum(fmin, fmax)
    return fmin, fmean, fmax


def _coefficient_energy(frame: pd.DataFrame) -> np.ndarray:
    if "wt_value" in frame:
        energy = np.zeros(len(frame), dtype=float)
        for row, value in enumerate(frame["wt_value"].to_numpy()):
            value = np.asarray(value, dtype=float)
            value = value[np.isfinite(value)]
            energy[row] = value @ value
        return energy
    rank = _numeric(frame, ("EnWDF", "mSNR"), default=0.0)
    return rank * rank


def _overlap_fraction(a0, a1, b0, b1):
    overlap = np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))
    width = np.maximum(np.minimum(a1 - a0, b1 - b0), EPS)
    return overlap / width


def _shifted_overlap_fraction(a0, a1, b0, b1, tolerance):
    """Overlap of two intervals, allowing one to shift by up to `tolerance`.

    Normalised by the shorter of the two undilated widths, so an interval
    contained in the other overlaps fully -- which is the case of a real
    coincidence between detectors of unequal sensitivity, where the weaker one
    keeps fewer coefficients and its support is a subset of the stronger one's.

    :param a0: start of the first interval.
    :param a1: end of the first interval.
    :param b0: start of the second interval.
    :param b1: end of the second interval.
    :param tolerance: how far the second interval may shift either way.
    :return: the overlap fraction, between 0 and 1.
    """
    width = np.maximum(np.minimum(a1 - a0, b1 - b0), EPS)
    overlap = np.minimum(a1, b1 + tolerance) - np.maximum(a0, b0 - tolerance)
    return np.clip(np.minimum(overlap, width) / width, 0.0, 1.0)


def _intervals_touch(a0, a1, b0, b1, tolerance):
    """Whether two intervals meet once one may shift by `tolerance`.

    Stated as an intersection rather than as a fraction of either width, so an
    event of no measured extent --- one tile, or a catalogue that records only
    an instant --- is admitted when it falls inside the other, which a fraction
    normalised by a zero width cannot express.

    :param a0: start of the first interval.
    :param a1: end of the first interval.
    :param b0: start of the second interval.
    :param b1: end of the second interval.
    :param tolerance: how far the second interval may shift either way.
    :return: boolean array.
    """
    return np.minimum(a1, b1 + tolerance) >= np.maximum(a0, b0 - tolerance)


def _group_bounds(size: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(size)[:-1]))


def _group_weighted_mean(values, weights, group_index, n_groups):
    total = np.bincount(group_index, weights=weights, minlength=n_groups)
    weighted = np.bincount(group_index, weights=values * weights, minlength=n_groups)
    return np.divide(weighted, total, out=np.full(n_groups, np.nan), where=total > 0)


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

    Adjacency is measured between the analysis windows themselves, on `gps`,
    which advances by exactly one stride. Measuring it between the windows'
    peaks cannot work: a peak sits anywhere inside its own window, the stride is
    shorter than the window by the overlap, so two adjacent windows' peaks can
    differ by more than a stride and the pair is refused although the windows
    are consecutive.
    """
    config = ClusterConfig() if config is None else config
    cleaned = clean_triggers_robust(
        triggers,
        segment_bounds=segment_bounds,
        edge_margin_s=config.edge_margin_s,
    )

    if cleaned.empty:
        return cleaned.assign(cluster_id=pd.Series(dtype=int)), pd.DataFrame()

    window_time = _numeric(cleaned, ("gps", "gpsPeak", "gpsMax"))
    order = np.argsort(window_time, kind="mergesort")
    cleaned = cleaned.iloc[order].reset_index(drop=True)
    window_time = window_time[order]
    time = _numeric(cleaned, ("gpsPeak", "gpsMax", "gps"))

    fmin, fmean, fmax = _frequency_interval(cleaned)
    energy = np.maximum(_coefficient_energy(cleaned), EPS)
    rank = _numeric(cleaned, ("EnWDF", "mSNR"), default=0.0)

    stride = stride_seconds(parameters)
    # The window origins sit on an exact grid, so this is a count of windows
    # rather than a tolerance: half a stride of slack keeps the comparison
    # clear of rounding without ever reaching the next window.
    time_eps = stride * (1.5 + int(config.max_missing_windows))
    uf = _UnionFind(len(cleaned))
    log_energy = np.log(energy)

    for left, right in neighbour_pairs(window_time, time_eps):
        keep_pair = (
            _overlap_fraction(fmin[left], fmax[left], fmin[right], fmax[right])
            >= config.minimum_frequency_overlap
        ) & (
            np.abs(log_energy[left] - log_energy[right])
            <= config.maximum_log_energy_jump
        )
        for i, j in zip(left[keep_pair], right[keep_pair]):
            uf.union(i, j)

    roots = np.array([uf.find(i) for i in range(len(cleaned))])
    _, cluster_ids = np.unique(roots, return_inverse=True)
    cleaned["cluster_id"] = cluster_ids

    sizes = np.bincount(cluster_ids, minlength=cluster_ids.max() + 1)
    n_clusters = len(sizes)
    by_cluster = np.argsort(cluster_ids, kind="stable")
    starts = _group_bounds(sizes)
    grouped = cluster_ids[by_cluster]

    weights = np.maximum(energy, EPS)[by_cluster]
    group_rank = rank[by_cluster]

    # The peak member of each cluster is the loudest, so ordering by cluster and
    # then by decreasing rank puts it first in every group.
    peak_at = np.lexsort((-rank, cluster_ids))[starts]

    start_time = _numeric(cleaned, ("gpsStart", "gps"))
    start_time = np.where(np.isfinite(start_time), start_time, window_time)
    gps_start = np.minimum.reduceat(start_time[by_cluster], starts)

    if "gpsEnd" in cleaned:
        end_time = _numeric(cleaned, ("gpsEnd",))
    elif "duration" in cleaned:
        end_time = start_time + _numeric(cleaned, ("duration",), default=0.0)
    else:
        end_time = window_time + float(parameters.window) / float(parameters.resampling)
    gps_end = np.maximum.reduceat(np.nan_to_num(end_time[by_cluster], nan=-np.inf), starts)

    peak_time = time[peak_at]
    span = np.maximum(0.0, gps_end - gps_start)

    # The cluster's energy centroid and spread in time are the moments of the
    # union of its members' tiles, which compose exactly from the members' own:
    # the centroid is their energy-weighted mean, and the spread adds each
    # member's spread to its distance from the common centroid.
    member_centroid = _numeric(cleaned, ("gpsCentroid",), default=np.nan)
    member_centroid = np.where(np.isfinite(member_centroid), member_centroid, time)
    member_spread = _numeric(cleaned, ("tSpread",), default=0.0)
    member_spread = np.where(np.isfinite(member_spread), member_spread, 0.0)

    cluster_centroid = _group_weighted_mean(
        member_centroid[by_cluster], weights, grouped, n_clusters)
    offset = member_centroid[by_cluster] - cluster_centroid[grouped]
    cluster_spread = np.sqrt(np.maximum(_group_weighted_mean(
        member_spread[by_cluster] ** 2 + offset ** 2, weights, grouped, n_clusters), 0.0))

    snr_peak = _numeric(cleaned, ("snrPeak",), default=0.0)
    sigma = _numeric(cleaned, ("sigmaWin", "sigma", "mSigma"), default=np.nan)
    finite_sigma = np.isfinite(sigma) & (sigma > 0.0)

    sigma_weights = np.where(finite_sigma, 1.0, 0.0)[by_cluster] * weights
    cluster_sigma = _group_weighted_mean(
        np.where(finite_sigma, sigma, 0.0)[by_cluster],
        sigma_weights,
        grouped,
        n_clusters,
    )

    cluster_snr_peak = np.maximum.reduceat(
        np.nan_to_num(snr_peak[by_cluster], nan=-np.inf), starts
    )
    cluster_snr_peak[~np.isfinite(cluster_snr_peak)] = np.nan

    ifo_column = (
        cleaned["ifo"].to_numpy()[peak_at]
        if "ifo" in cleaned
        else np.full(n_clusters, getattr(parameters, "itf", ""))
    )
    if "wave" in cleaned:
        wave_column = cleaned["wave"].to_numpy()[peak_at]
    elif "mWave" in cleaned:
        wave_column = cleaned["mWave"].to_numpy()[peak_at]
    else:
        wave_column = np.full(n_clusters, "")

    member_indices = np.split(
        cleaned["trigger_index"].to_numpy()[by_cluster], np.cumsum(sizes)[:-1]
    )

    events = pd.DataFrame(
        {
            "cluster_id": np.arange(n_clusters, dtype=int),
            "ifo": ifo_column,
            "gps": gps_start,
            "gpsStart": gps_start,
            "gpsEnd": gps_end,
            "gpsMax": peak_time,
            "gpsPeak": peak_time,
            # Coincidence time. Which tile is loudest depends on the noise
            # realisation and on which basis won, so the peak need not fall at
            # the same instant in two detectors seeing the same signal.
            "gpsCentroid": cluster_centroid,
            "tSpread": cluster_spread,
            "duration": span,
            "gps_span_s": span,
            "freqMin": np.minimum.reduceat(fmin[by_cluster], starts),
            "freqMean": _group_weighted_mean(
                fmean[by_cluster], weights, grouped, n_clusters),
            "freqMax": np.maximum.reduceat(fmax[by_cluster], starts),
            # Primary cluster ranking: do not add EnWDF values in quadrature
            # across overlapping WDF windows, because that double-counts common
            # samples.
            "EnWDF": np.maximum.reduceat(group_rank, starts),
            "cluster_sum_enwdf": np.sqrt(
                np.bincount(grouped, weights=group_rank * group_rank,
                            minlength=n_clusters)),
            "snrPeak": cluster_snr_peak,
            "sigmaWin": cluster_sigma,
            "coefficient_energy": np.bincount(
                grouped, weights=energy[by_cluster], minlength=n_clusters),
            "n_triggers": sizes.astype(int),
            "singleton": sizes == 1,
            "member_indices": [tuple(int(v) for v in m) for m in member_indices],
            "wave": wave_column,
        }
    )

    return cleaned, events


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
    """How close two single-detector events have to be to form a candidate.

    The timing tolerance is not a fixed number: it is the light travel time
    between the detectors plus the two events' own declared timing spreads
    combined in quadratura, which is what makes one window fit a blip of a few
    milliseconds and a chirp of several seconds alike. `timing_jitter_s` is the
    floor on a single event's spread -- an event living in one tile declares
    zero spread, which is a statement about the tiling, not about the signal.

    :param light_travel_time_s: largest arrival-time difference a real signal
        can have between the detectors, seconds. None takes it from the
        detectors named in `ifos`, which is a distance rather than a number
        chosen for one pair; a value overrides that.
    :param ifos: the detectors in the network, used to resolve the light travel
        time when it is not given.
    :param timing_jitter_s: floor on an event's timing spread, seconds. It is a
        floor and not the timing error: an event living in one tile declares
        zero spread, which is a statement about the tiling. Set so that the
        smallest tolerance any pair can claim is 25 ms, a little over twice the
        light travel time, rather than being set by the floor.
    :param timing_sigma: how many combined spreads to accept.
    :param minimum_frequency_overlap: least overlap of the two bands.
    :param minimum_time_overlap: least overlap of the two time supports, once
        one of them is allowed to shift by the light travel time.
    :param maximum_tolerance_travel_times: largest time difference any pair may
        claim, in units of that pair's own light travel time, whatever their
        spreads. The arrival times of one signal differ by at most the light
        travel time; an event's spread is uncertainty on its own centroid, and
        for a signal seen in both detectors that uncertainty is largely common
        and cancels in the difference. Without a cap a pair of long events
        claims seconds, which no signal can produce and which lets the
        accidental rate grow with the events' duration. Expressed as a multiple
        because a cap in seconds is a cap for one baseline: 2.5 is 25 ms across
        Hanford and Livingston and 68 ms across Hanford and Virgo.
    :param maximum_tolerance_s: an explicit cap in seconds, overriding the
        multiple when given.
    """

    light_travel_time_s: float | None = None
    timing_jitter_s: float = 0.0035
    timing_sigma: float = 3.0
    minimum_frequency_overlap: float = 0.0
    minimum_time_overlap: float = 0.0
    maximum_tolerance_travel_times: float = 2.5
    maximum_tolerance_s: float | None = None
    time_weight: float = 1.0
    frequency_weight: float = 1.0
    # Amplitude ratio is not a shape mismatch: the same signal reaches two
    # detectors with amplitudes set by their antenna responses, routinely
    # differing by a factor of a few. Penalizing it makes the assignment prefer
    # a quieter, better-matched pair over the louder true one, so it is off by
    # default; the frequency term already carries the shape test.
    morphology_weight: float = 0.0

    def travel_time(self, ifos=None) -> float:
        """The largest arrival-time difference this pair's geometry allows.

        :param ifos: the detectors being paired; None with no explicit value
            set means no geometry is known and the time is zero.
        :return: float -- seconds.
        """
        if self.light_travel_time_s is not None:
            return float(self.light_travel_time_s)
        return network_light_travel_time(ifos or ())

    def maximum_tolerance(self, ifos=None) -> float:
        """The cap on any pair's timing tolerance, seconds.

        :param ifos: the detectors being paired.
        :return: float -- seconds.
        """
        if self.maximum_tolerance_s is not None:
            return float(self.maximum_tolerance_s)
        return self.maximum_tolerance_travel_times * self.travel_time(ifos)

    def timing_tolerance(self, left_spread, right_spread, ifos=None):
        """Largest time difference the two events may have and still pair.

        :param left_spread: one event's timing spread, seconds.
        :param right_spread: the other event's timing spread, seconds.
        :param ifos: the detectors being paired, which set the light travel
            time and the cap.
        :return: the tolerance, seconds.
        """
        left = np.maximum(left_spread, self.timing_jitter_s)
        right = np.maximum(right_spread, self.timing_jitter_s)
        return np.minimum(
            self.travel_time(ifos) + self.timing_sigma * np.hypot(left, right),
            self.maximum_tolerance(ifos))


class IndexedCoincidenceFinder:
    """Indexed one-to-one H1-L1 coincidence.

    Candidate intervals are obtained with searchsorted. Ambiguous local
    bipartite components are resolved by minimum-cost one-to-one assignment.
    """

    def __init__(self, config: CoincidenceConfig | None = None, **kwargs):
        if config is None:
            config = CoincidenceConfig(**kwargs)
        self.config = config

    @staticmethod
    def _pair(left=None, right=None) -> tuple:
        """The detectors two event frames belong to.

        The geometry of a coincidence is a property of the pair being tested,
        so it is read from the events rather than configured once for the whole
        network.

        :param left: one detector's events.
        :param right: the other detector's events.
        :return: tuple -- the detector names found, in the order given.
        """
        names = []
        for events in (left, right):
            if events is not None and len(events) and "ifo" in events:
                names.append(str(events["ifo"].iloc[0]))
        return tuple(names)

    def coincidence_window(self, left=None, right=None):
        """How far apart two of these events can sit and still overlap in time.

        The test a pair has to pass is that the stretches of time the two events
        cover overlap, so the pairs worth forming are those whose extents can
        reach each other: the two longest, plus the light travel time. A window
        set by the timing tolerance alone would search a fraction of a second
        and never form the pair a signal lasting seconds produces.

        :param left: one detector's events, or None for the configured floor.
        :param right: the other detector's events, or None.
        :return: the reach, seconds.
        """
        reach = self.config.travel_time(self._pair(left, right))
        for events in (left, right):
            if events is not None and len(events):
                extent = _numeric(events, ("duration",), default=0.0)
                extent = extent[np.isfinite(extent)]
                reach += float(extent.max()) if extent.size else 0.0
        spreads = [self.config.timing_jitter_s]
        for events in (left, right):
            if events is not None and len(events):
                spread = _numeric(events, ("tSpread",), default=0.0)
                spread = spread[np.isfinite(spread)]
                if spread.size:
                    spreads.append(float(spread.max()))
        widest = max(spreads)
        return float(max(reach, self.config.timing_tolerance(
            widest, widest, self._pair(left, right))))

    def candidate_edges(self, left, right):
        """Every pair of events a signal could physically have produced.

        The pair must cover the same stretch of time once one of the two is
        allowed to shift by the light travel time, and share enough of its band.

        The test is on the events' extents and not on any single instant of
        them. An extended transient has no arrival time: a chirp lasting
        seconds is spread over all of them, and which instant a detector calls
        its centroid or its peak depends on its own noise, its antenna response
        and which coefficients survived threshold. Measured on the simulated
        set, two detectors seeing one compact binary put their centroids a
        median of 81 ms apart and their peaks 15.6 ms, against a light travel
        time of 10 ms --- so no instant agrees, while the stretches covered do.
        For a transient shorter than the light travel time the two statements
        coincide, which is why this is the general one.

        `dt` is still measured and carried, and ranks the survivors; it no
        longer decides which pairs exist.

        What survives is the candidate set; deciding among the survivors is a
        separate question, and this is what both the one-to-one assignment and
        the graph stage start from, so that the two admit the same pairs.

        :param left: one detector's events.
        :param right: the other detector's events.
        :return: list of (i, j, cost, dt, frequency_overlap, time_overlap).
        """
        lt = _numeric(left, ("gpsCentroid", "gpsPeak"))
        rt = _numeric(right, ("gpsCentroid", "gpsPeak"))
        ls = _numeric(left, ("tSpread",), default=0.0)
        rs = _numeric(right, ("tSpread",), default=0.0)
        ls = np.where(np.isfinite(ls), ls, 0.0)
        rs = np.where(np.isfinite(rs), rs, 0.0)
        l_start = _numeric(left, ("gpsStart", "gpsCentroid", "gpsPeak"))
        r_start = _numeric(right, ("gpsStart", "gpsCentroid", "gpsPeak"))
        l_end = l_start + _numeric(left, ("duration",), default=0.0)
        r_end = r_start + _numeric(right, ("duration",), default=0.0)
        lf0, lfm, lf1 = _frequency_interval(left)
        rf0, rfm, rf1 = _frequency_interval(right)
        le = np.maximum(_coefficient_energy(left), EPS)
        re = np.maximum(_coefficient_energy(right), EPS)

        pair = self._pair(left, right)
        widest = self.coincidence_window(left, right)

        # Formed as arrays over the admissible pairs rather than one pair at a
        # time: the per-pair tolerance is the same expression evaluated
        # elementwise, and at these event rates the Python call dominated the
        # whole graph build.
        left_order = np.argsort(lt, kind="mergesort")
        right_order = np.argsort(rt, kind="mergesort")
        blocks = []
        for a, b in cross_pairs(lt[left_order], rt[right_order], widest):
            i, j = left_order[a], right_order[b]
            tolerance = self.config.timing_tolerance(ls[i], rs[j], pair)
            dt = lt[i] - rt[j]
            overlap = _overlap_fraction(lf0[i], lf1[i], rf0[j], rf1[j])
            time_overlap = _shifted_overlap_fraction(
                l_start[i], l_end[i], r_start[j], r_end[j],
                self.config.travel_time(pair))
            keep = (
                _intervals_touch(l_start[i], l_end[i], r_start[j], r_end[j],
                                 tolerance)
                & (overlap >= self.config.minimum_frequency_overlap)
                & (time_overlap >= self.config.minimum_time_overlap)
            )
            if not keep.any():
                continue
            i, j, dt = i[keep], j[keep], dt[keep]
            overlap, time_overlap = overlap[keep], time_overlap[keep]
            scale = np.maximum(np.maximum(lf1[i] - lf0[i], rf1[j] - rf0[j]), 1.0)
            cost = (
                self.config.time_weight * np.abs(dt) / np.maximum(tolerance[keep], EPS)
                + self.config.frequency_weight * np.abs(lfm[i] - rfm[j]) / scale
                + self.config.morphology_weight * np.abs(np.log(le[i] / re[j]))
            )
            blocks.append(np.column_stack([i, j, cost, dt, overlap, time_overlap]))

        if not blocks:
            return []
        rows = np.concatenate(blocks)
        rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
        return [(int(r[0]), int(r[1]), float(r[2]), float(r[3]), float(r[4]),
                 float(r[5])) for r in rows]

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

        edges = self.candidate_edges(left, right)
        selected = []

        for component in self._components(len(left), len(right), edges):
            left_ids = sorted({edge[0] for edge in component})
            right_ids = sorted({edge[1] for edge in component})
            li = {value: index for index, value in enumerate(left_ids)}
            rj = {value: index for index, value in enumerate(right_ids)}
            cost = np.full((len(left_ids), len(right_ids)), np.inf)
            metadata = {}

            for i, j, value, dt, overlap, time_overlap in component:
                if value < cost[li[i], rj[j]]:
                    cost[li[i], rj[j]] = value
                    metadata[(i, j)] = (dt, overlap, time_overlap)

            finite = np.isfinite(cost)
            if not finite.any():
                continue

            penalty = float(np.nanmax(cost[finite]) + 1e6)
            row_index, col_index = linear_sum_assignment(np.where(finite, cost, penalty))

            for a, b in zip(row_index, col_index):
                if not finite[a, b]:
                    continue
                i, j = left_ids[a], right_ids[b]
                dt, overlap, time_overlap = metadata[(i, j)]
                selected.append((i, j, cost[a, b], dt, overlap, time_overlap))

        if not selected:
            return pd.DataFrame()

        # Built column by column. Taking one row per candidate through .iloc
        # costs a pandas row materialisation each time, which at the event rates
        # a search without a per-detector threshold produces is most of the run.
        chosen = np.asarray(selected, dtype=float)
        i = chosen[:, 0].astype(int)
        j = chosen[:, 1].astype(int)
        cost, dt = chosen[:, 2], chosen[:, 3]
        overlap, time_overlap = chosen[:, 4], chosen[:, 5]

        sa = _numeric(left, ("EnWDF", "snrMax"), default=0.0)[i]
        sb = _numeric(right, ("EnWDF", "snrMax"), default=0.0)[j]
        ta = _numeric(left, ("gpsCentroid", "gpsPeak"))[i]
        tb = _numeric(right, ("gpsCentroid", "gpsPeak"))[j]

        out = pd.DataFrame({
            f"index_{left_ifo}": i,
            f"index_{right_ifo}": j,
            f"EnWDF_{left_ifo}": sa,
            f"EnWDF_{right_ifo}": sb,
            f"snr_{left_ifo}": sa,
            f"snr_{right_ifo}": sb,
            "gps_candidate": 0.5 * (ta + tb),
            "delta_t": dt,
            "dt_s": dt,
            "frequency_overlap": overlap,
            "time_overlap": time_overlap,
            "coincidence_cost": cost,
            "network_enwdf": np.hypot(sa, sb),
            "network_min_enwdf": np.minimum(sa, sb),
            "network_geometric_enwdf": np.sqrt(np.maximum(sa * sb, 0.0)),
            "network_snr": np.hypot(sa, sb),
            "ifos_involved": f"{left_ifo},{right_ifo}",
            "n_ifos": 2,
        })
        out = out.sort_values(

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

        template = None
        for slide_index in range(self.config.n_slides):
            magnitude = rng.uniform(self.config.min_shift_s, span - self.config.min_shift_s)
            shift = magnitude if rng.integers(0, 2) else -magnitude
            shifts.append(shift)

            shifted = events_by_ifo[shifted_ifo].copy()
            # One displacement per event, applied to all of its times. Wrapping
            # each column on its own moves an event that straddles the seam by
            # different amounts in each, so its start no longer precedes its end
            # and its extent becomes the length of the segment.
            reference = _numeric(shifted, ("gpsCentroid", "gpsPeak", "gps"))
            wrapped = start + ((reference - start + shift) % span)
            displacement = wrapped - reference
            for column in ("gpsMax", "gpsPeak", "gpsCentroid", "gpsStart", "gpsEnd"):
                if column in shifted:
                    shifted[column] = (
                        pd.to_numeric(shifted[column], errors="coerce") + displacement)

            candidates = self.coincidence_finder.find(
                {ref: events_by_ifo[ref], shifted_ifo: shifted}
            )
            if template is None:
                template = candidates.iloc[0:0]
            if len(candidates):
                candidates = candidates.copy()
                candidates["slide_index"] = slide_index
                candidates["slide_shift_s"] = shift
                rows.append(candidates)

        # Slides that produced nothing still say what a candidate looks like, so
        # an empty background keeps its columns and is empty rather than
        # unrecognisable.
        background = pd.concat(rows, ignore_index=True) if rows else (
            pd.DataFrame() if template is None else template.copy())
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
