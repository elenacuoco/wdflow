"""Events, coincidence and background, from trigger files to a false-alarm rate.

The stages that turn one detector's triggers into a network candidate with a
rate attached: cleaning a trigger file, grouping consecutive triggers into
events, admitting pairs of events across detectors on the geometry a real
signal must satisfy, and measuring how often that admission happens by accident.

The timing tolerance of a pair is not a fixed number. It is the light travel
time between the detectors, widened by the two events' own declared timing
spreads and capped, so that one rule fits a burst of a few milliseconds and a
transient of several seconds without either being treated as a special case.

The accidental rate is measured by shifting the detectors against each other by
more than any signal could explain, so every coincidence a shift produces is
accidental by construction and none of them has to be identified as such.
"""
 
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from wdf.analysis.detectors import network_light_travel_time
from wdf.analysis.pairs import cross_pairs, interval_pairs, neighbour_pairs

EPS = np.finfo(float).tiny


#: Where an event's instant is read from, in the order it is preferred. The
#: usual one is the second, the centre of the tile carrying the event's largest
#: coefficient, which is what the search itself produces; it lasts one over the
#: upper edge of its band and so cannot resolve a network's light travel time
#: wherever that band is low. The first is offered for a catalogue that has had
#: an instant read below the tile, on the event's own reconstruction (see
#: `wdf.analysis.timing`), and no stage here writes it. The last is the energy
#: centroid, which two detectors do not agree on. Every one of them is a
#: property of one event, so a time slide carries it with the event.
INSTANT_COLUMNS = ("gpsEnvelope", "gpsPeak", "gpsCentroid")


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
        can have between the detectors, seconds. None resolves it from the
        detectors the pair is being evaluated for, passed to the methods below,
        which is a distance rather than a number chosen for one pair; a value
        set here overrides that for every pair.
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

        The test is on the events' stretches of time and not on any single
        instant of them. A transient longer than one analysis window is
        assembled as several events, and the two detectors do not keep the
        same one: measured on injections both detectors recovered, the
        smallest difference between two such instants is milliseconds for a
        signal shorter than a window and hundreds of milliseconds for one
        lasting minutes. Gating on it would discard the second, which is a
        stage taking evidence away from a candidate the detectors assembled.

        The difference is measured on every admitted pair, from each event's
        own instant --- `INSTANT_COLUMNS` names where that is read from --- and
        ranks the survivors. It does not decide which pairs exist.

        What survives is the candidate set; deciding among the survivors is a
        separate question, and this is what both the one-to-one assignment and
        the graph stage start from, so that the two admit the same pairs.

        :param left: one detector's events.
        :param right: the other detector's events.
        :return: list of (i, j, cost, dt, frequency_overlap, time_overlap).
        """
        # The instant the pair is ranked on. `INSTANT_COLUMNS` is an order of
        # preference and not a requirement: the search itself produces the
        # centre of the tile carrying the event's largest coefficient, and a
        # catalogue that has had an instant read below the tile --- see
        # `wdf.analysis.timing` --- is timed on that one instead, without
        # anything here having to know which happened. Both are node
        # quantities, so the difference of two of them is one too and a time
        # slide carries it. The energy centroid is last because it measures how
        # much of a transient survived threshold in that detector, so two
        # detectors at different projected amplitudes place it differently and
        # the difference lands in dt, where it is indistinguishable from
        # geometry.
        lt = _numeric(left, INSTANT_COLUMNS)
        rt = _numeric(right, INSTANT_COLUMNS)
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

        # Formed as arrays over the admissible pairs rather than one pair at a
        # time: the per-pair tolerance is the same expression evaluated
        # elementwise, and at these event rates the Python call dominated the
        # whole graph build.
        #
        # Enumerated on each event's own extent, not on a global window. The
        # pairs that can survive are those whose stretches touch within the
        # tolerance cap, so each left event needs the right events whose
        # anchor falls between its own start, less the cap and the farthest
        # any right event reaches past its anchor, and its own end plus the
        # cap. One long event then widens its own enumeration and nobody
        # else's; a window of the longest duration either side multiplies that
        # event's reach onto every event of both detectors, and on data whose
        # noise chains long events the enumeration alone exhausts the memory.
        tol_up = float(self.config.maximum_tolerance(pair))
        l_lo = np.where(np.isfinite(l_start), l_start, lt) - tol_up
        l_hi = np.where(np.isfinite(l_end), l_end, lt) + tol_up
        r_anchor = np.where(np.isfinite(r_start), r_start, rt)
        r_extent = np.where(np.isfinite(r_end - r_anchor), r_end - r_anchor, 0.0)
        r_reach = float(r_extent.max()) if r_extent.size else 0.0
        right_order = np.argsort(r_anchor, kind="mergesort")
        blocks = []
        for i, b in interval_pairs(l_lo, l_hi, r_anchor[right_order], r_reach):
            j = right_order[b]
            tolerance = self.config.timing_tolerance(ls[i], rs[j], pair)
            dt = lt[i] - rt[j]
            overlap = _overlap_fraction(lf0[i], lf1[i], rf0[j], rf1[j])
            time_overlap = _shifted_overlap_fraction(
                l_start[i], l_end[i], r_start[j], r_end[j],
                self.config.travel_time(pair))
            # The test is on the events' stretches of time, not on the
            # difference of their instants. A transient longer than one
            # analysis window is assembled as several events, and the two
            # detectors do not keep the same one: their instants then differ
            # by far more than the light travel time although both belong to
            # one signal. Gating on that difference would discard the pair,
            # which is a stage taking evidence away from a candidate the
            # detectors did assemble. The difference is measured on every
            # admitted pair and ranks it instead.
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
        # The same clock the admissibility used, so a candidate's recorded dt
        # is the quantity it was admitted and penalised on.
        ta = _numeric(left, INSTANT_COLUMNS)[i]
        tb = _numeric(right, INSTANT_COLUMNS)[j]

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
    """How the accidental background is drawn.

    :param n_slides: how many displacements to draw. The accidental livetime
        is this times the span, and the lowest rate a background resolves is
        one over that livetime.
    :param min_shift_s: the step between one displacement and the next,
        seconds --- a stated constant, as published burst searches state
        theirs. It does two jobs, and they ask for the same number: a
        displacement below the window a pair is admitted in leaves a real
        coincidence admissible, so the signal enters the background it is
        measured against; and two lags closer together than the length the
        trigger stream is correlated over re-use the same clusters, so the
        tail is redrawn rather than sampled. Both scales are measured from the
        data and a step below either is refused, so the constant is checked
        rather than trusted. None derives it from those measurements instead,
        which makes the step a property of the run and not of the analysis:
        two runs then draw backgrounds a reader cannot compare.
    :param seed: the generator's seed. The grid is regular, so this chooses
        only the phase it starts at.
    """

    n_slides: int = 100
    min_shift_s: float | None = 4.0
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

    def _admission_window(self, ifos):
        """The widest timing tolerance the coincidence allows, seconds.

        A displacement smaller than this leaves a real coincidence admissible,
        and two lags closer together than this form the same pairings twice.
        Admission also widens by the events' own extents, which is the second
        scale the step is held against below; this is the part of it the
        coincidence configuration fixes, so a background drawn for one
        admission rule cannot be read against another.

        :param ifos: the detectors being paired.
        :return: float -- the widest tolerance over the pairs, seconds.
        :raises TypeError: if the finder does not carry a coincidence
            configuration to read it from.
        """
        finder = self.coincidence_finder
        config = getattr(finder, "config", None)
        if config is None:
            builder = getattr(finder, "builder", None)
            config = getattr(builder, "coincidence", None)
        if config is None or not hasattr(config, "maximum_tolerance"):
            raise TypeError(
                f"{type(finder).__name__} carries no coincidence "
                f"configuration, so the window a pair is admitted in --- which "
                f"is the smallest displacement a slide may use --- cannot be "
                f"read")
        names = [str(i) for i in ifos]
        return max(float(config.maximum_tolerance((a, b)))
                   for a, b in combinations(names, 2))

    def _shift_grid(self, rng, n_shifted, span, least):
        """Every displacement the span holds, one row per slide.

        Enumerated and not drawn. A displacement drawn at random can repeat a
        pairing an earlier draw already formed, and two draws of one pairing
        are two perfectly correlated rows in a distribution a threshold reads
        as an order statistic: the tail is then redrawn rather than sampled,
        and the livetime credited overstates what was measured. A grid of
        multiples of one step gives each configuration once, which is what
        published burst searches do.

        The step is `min_shift_s`, one distance for every lag, and the span
        holds as many as fit --- so a short segment yields fewer lags than
        asked for rather than the same ones twice. `seed` chooses the phase the
        grid starts at, which keeps a background reproducible and lets two runs
        sample different offsets.

        Every detector but the reference takes its own multiple of the step, so
        every difference between two shifted detectors is a multiple of the
        step as well and clears the floor by construction --- two detectors
        moved by nearly the same amount would stay in step with each other and
        their own pair would keep its real coincidences.

        Only positive multiples are enumerated: a displacement wraps inside the
        span, so a shift of `-s` is the shift of `span - s` and the positive
        multiples already cover every distinct configuration.

        :param rng: the generator the phase is drawn from.
        :type n_shifted: int
        :param n_shifted: how many detectors are displaced.
        :type span: float
        :param span: length of the segment, seconds.
        :type least: float
        :param least: the smallest displacement, and the step between one lag
            and the next: the window a pair is admitted in.
        :return: numpy.ndarray -- shape ``(slides, n_shifted)``, the
            displacement of each shifted detector in each slide. Fewer rows
            than `n_slides` where the span cannot hold that many.
        :raises ValueError: if the span holds no displacement at all.
        """
        # A fixed step, as published burst searches use: every lag is the same
        # distance from the last, and a span holds as many as fit. Spreading a
        # requested number over the whole span instead would make the step a
        # function of how many were asked for, which is not a property of the
        # data.
        step = float(least)
        asked = int(self.config.n_slides)
        # Zero and the whole span are the identity, and one step of room is
        # left for the phase.
        available = int(np.floor(span / step)) - n_shifted - 1
        if available < 1:
            raise ValueError(
                f"a span of {span:g} s holds no displacement of at least "
                f"{least:g} s for {n_shifted} detector(s) beside the reference")
        used = min(asked, available)
        phase = float(rng.uniform(0.0, step))
        k = np.arange(1, used + 1, dtype=float)
        return step * (k[:, None] + np.arange(n_shifted)[None, :]) + phase

    def background_distribution(self, events_by_ifo, segment_bounds,
                                reduce=None):
        """Accidental coincidences, by displacing every detector but the first.

        :param events_by_ifo: `{ifo: events}`, two or more detectors. The first
            is the reference and is never moved.
        :param segment_bounds: `{ifo: (start, end)}`, the stretch each
            detector's events were found in.
        :param reduce: called with each slide's candidates as they are formed,
            as `reduce(candidates, livetime_s)`. Given one, the slides are not
            accumulated: what a rate needs is an order statistic, and the
            caller keeps only that, so the memory a background costs stops
            growing with the number of slides. The frame returned is then empty
            and carries the run's `attrs` alone.
        :type reduce: callable or None
        :return: pandas.DataFrame -- the accidental candidates, carrying
            `slide_index` and the shifts applied, with the number of slides and
            the total slid livetime in `attrs`; empty when `reduce` was given.
        :raises ValueError: with fewer than two detectors, or a segment too
            short for the requested shifts.
        """
        ifos = list(events_by_ifo)
        if len(ifos) < 2:
            raise ValueError("time slides need at least two IFOs")
        ref, shifted_ifos = ifos[0], ifos[1:]
        start, end = map(float, segment_bounds[shifted_ifos[0]])
        span = end - start
        # The window a pair is admitted in. A pair is admitted when the two
        # events' instants differ by no more than the coincidence's tolerance,
        # so the widest that tolerance can be is the displacement below which
        # a real coincidence stays admissible --- and equally the spacing below
        # which two lags form the same pairings twice. It is read from the
        # coincidence itself rather than assumed of it.
        # A caller that states a floor has said what it wants; one that does
        # not is asking the coincidence, and a coincidence that cannot answer
        # is refused rather than guessed at.
        stated = self.config.min_shift_s
        try:
            admission = self._admission_window(ifos)
        except TypeError:
            if stated is None:
                raise
            admission = None

        # Clearing the admission window keeps a real coincidence out of the
        # background and stops two lags forming the same pairing twice, but it
        # does not make two lags independent. Triggers arrive in clusters as
        # long as the events themselves, so two displacements closer together
        # than that length re-use the same clusters in nearly the same
        # configuration. The scale is taken from the events: the extent all
        # but the longest hundredth of them fit inside.
        correlated = 0.0
        for frame in events_by_ifo.values():
            if not len(frame):
                continue
            extent = _numeric(frame, ("duration",), default=0.0)
            extent = extent[np.isfinite(extent)]
            if extent.size:
                correlated = max(correlated, float(np.quantile(extent, 0.99)))
        floor = correlated if admission is None else max(admission, correlated)
        least = floor if stated is None else float(stated)
        if admission is not None and least < admission:
            raise ValueError(
                f"the smallest displacement is {least:g} s and a pair is "
                f"admitted within {admission:g} s; a slide shorter than that "
                f"keeps real coincidences in the background")
        if least < correlated:
            raise ValueError(
                f"the smallest displacement is {least:g} s and the events "
                f"reach {correlated:g} s; lags closer than that are not "
                f"independent realisations of the trigger stream")
        if span <= 2.0 * least:
            raise ValueError(
                f"a span of {span:g} s cannot hold displacements of at least "
                f"{least:g} s in either direction")

        rng = np.random.default_rng(self.config.seed)
        grid = self._shift_grid(rng, len(shifted_ifos), span, least)
        rows = []
        shifts = []

        template = None
        for slide_index, drawn in enumerate(grid):
            shifts.append(float(drawn[0]) if len(drawn) == 1
                          else tuple(float(s) for s in drawn))

            slid = {ref: events_by_ifo[ref]}
            for ifo, shift in zip(shifted_ifos, drawn):
                shifted = events_by_ifo[ifo].copy()
                # One displacement per event, applied to all of its times.
                # Wrapping each column on its own moves an event that straddles
                # the seam by different amounts in each, so its start no longer
                # precedes its end and its extent becomes the whole segment.
                reference = _numeric(shifted, ("gpsCentroid", "gpsPeak", "gps"))
                wrapped = start + ((reference - start + shift) % span)
                displacement = wrapped - reference
                for column in ("gpsMax", "gpsPeak", "gpsEnvelope",
                               "gpsCentroid", "gpsStart", "gpsEnd"):
                    if column in shifted:
                        shifted[column] = (pd.to_numeric(shifted[column],
                                                         errors="coerce")
                                           + displacement)
                slid[ifo] = shifted

            candidates = self.coincidence_finder.find(slid)
            if template is None:
                template = candidates.iloc[0:0]
            if len(candidates):
                candidates = candidates.copy()
                candidates["slide_index"] = slide_index
                # One number with two detectors, one per shifted detector with
                # more, so the column says what was actually applied.
                candidates["slide_shift_s"] = (
                    float(drawn[0]) if len(drawn) == 1
                    else [tuple(float(s) for s in drawn)] * len(candidates))
                if reduce is None:
                    rows.append(candidates)
                else:
                    reduce(candidates, span)
                    # The slide is released here: holding it is what makes the
                    # background cost grow with the number of displacements,
                    # and nothing downstream reads it again.
                    del candidates

        # Slides that produced nothing still say what a candidate looks like, so
        # an empty background keeps its columns and is empty rather than
        # unrecognisable.
        if reduce is not None:
            background = (template.copy() if template is not None
                          else pd.DataFrame())
        else:
            background = pd.concat(rows, ignore_index=True) if rows else (
                pd.DataFrame() if template is None else template.copy())
        # What the span held, which is what the livetime is credited on --- not
        # what was asked for. A short segment yields fewer distinct lags, and a
        # rate divided by the number requested would claim a livetime the
        # slides never produced.
        background.attrs["n_slides"] = int(len(grid))
        background.attrs["n_slides_requested"] = int(self.config.n_slides)
        background.attrs["segment_duration_s"] = span
        background.attrs["min_shift_s"] = float(least)
        background.attrs["admission_window_s"] = (
            float("nan") if admission is None else float(admission))
        background.attrs["correlated_length_s"] = float(correlated)
        background.attrs["total_livetime_s"] = float(len(grid)) * span
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
            # The slides the span held, not the number asked of it: a stretch
            # too short for the request yields fewer, and crediting the request
            # claims a livetime the slides never produced.
            total_livetime = (
                int(background.attrs.get("n_slides", self.config.n_slides))
                * float(segment_duration_s))
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
                int(background.attrs.get("n_slides", self.config.n_slides))
                * float(segment_duration_s),
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
