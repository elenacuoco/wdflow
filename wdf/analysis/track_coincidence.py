"""Stages 3 and 4 of the track path: pairs of track events admitted on their
tracks and waveforms, and ranked on the energy along the tracks.

`TrackCoincidenceFinder` is the track path's counterpart of
`wdf.analysis.network_graph.WavegramCoincidenceFinder`. The candidate pairs
are those `IndexedCoincidenceFinder.candidate_edge_array` forms; each is
described by `TRACK_EDGE_FEATURES`:

- **track compatibility** (`track_compatibility`): the time over which the
  two f(t) agree within `TrackFrequencyTolerance` octaves, once one may shift
  by the light travel time plus `TrackTimingJitter` (`track_common_s`);
- **waveform coherence** (`waveform_coherence`): the complex
  cross-correlation of the two reconstructed waveforms' analytic signals,
  normalised by their energies, searched over ``|tau| <= TrackLagSearch``:
  the lag of its largest modulus, that modulus ``|Gamma|`` (0 to 1) and its
  phase.

A pair is **admitted** when (a) and (b) hold, and (c) when a coherence
threshold is set:

(a) the tracks are compatible: the time their f(t) agree, `track_common_s`,
    is at least `TrackMinCommonFraction` of the shorter track, holds at least
    `TrackMinCommonCycles` cycles at the tracks' frequency there, and is at
    least `TrackMinCommonDuration` (an absolute floor);
(b) the coherence lag is physical: ``|track_lag_s| <=`` light travel time
    plus `TrackTimingJitter` (the search is wider, so a lag at its edge
    fails);
(c) ``track_coherence >= TrackCoherenceMin``.

With `TrackCoherenceMin` "none" (the ``[track.coincidence]`` default of
wdf/config/defaults.toml), there is no coherence cut. The coherence then
enters the ranking only, through `track_statistic`, and the false-alarm rate
of the time slides decides. `TrackCoherenceMin` may instead be a number, or
"calibrate": the `TrackCoherencePercentile` of ``|Gamma|`` over time-slid
pairs that pass (a) and (b) (`calibrate_coherence_min`).

The pair's statistic is `track_statistic`, ``min(rho_1, rho_2) |Gamma|``,
``rho = sqrt(track_energy)`` each detector's SNR integrated along its track
(the energy of the reconstructed waveform in noise units). The minimum, not
the product or the quadrature sum, asks both detectors to carry energy along
their tracks: a loud glitch in one detector paired with a weak track in the
other scores no more than the weak one. ``|Gamma|`` is the fraction of that
energy the two waveforms share at a physical lag, so a pair of loud,
unrelated tracks scores low. The track path ranks on it; `network_enwdf`, the
seed triggers' statistic, is kept as a column. `dt_s` is the coherence lag
`track_lag_s`; the classical finder's difference of instants is
`dt_instant_s`. Every lag has one sign: the first detector of `ifos_involved`
minus the second, ``t_H1 - t_L1`` for "H1,L1", as the classical `dt_s`,
`network_wavegram_match_dt` and `wdf.analysis.timing.arrival_time_difference`;
it is positive when L1 leads, as at GW150914.

A time slide (`TimeSlideFAR`) moves an event's times; the finder reads each
event's displacement from its `gpsPeak` against the unslid events it was built
with and moves the event's track and waveform by it.

With a scorer (`GNNCoincidenceScorer`), the pairs are those of a
`TriggerGraphBuilder` graph of the track events, whose node maps are drawn
from the waveforms (`track_event_coefficients`), the track features are
appended to the graph's edge features (the model must have been fitted with
``cross_edge_dim = N_EDGE_FEATURES + len(TRACK_EDGE_FEATURES)``), and the same
admission applies.
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import pandas as pd

from wdf.config.constants import PERCENT
from wdf.config.defaults import defaults

_TC = defaults("track.coincidence")

#: The features a track pair adds to the classical ones.
TRACK_EDGE_FEATURES = ["track_time_overlap", "track_common_s", "track_common_fraction",
                       "track_common_cycles", "track_f_distance",
                       "track_f_overlap", "track_compatible", "track_lag_s", "track_coherence",
                       "track_phase"]
#: The track path's ranking column (`NetworkRankColumn` default).
TRACK_STATISTIC = "track_statistic"
#: What a finder admits: "full" (a, b and c), "tracks" (a and b, for the
#: calibration), "classic" (every pair the classical rule forms).
GATES = ("full", "tracks", "classic")
#: `TrackCommonRule`: how the common-span minima of admission (a) combine.
COMMON_RULES = ("all", "duration-or-fraction")


def _coherence_min(value):
    """`TrackCoherenceMin` as a number: 0.0 for "none"/"off" or None (no
    cut), None for "calibrate", else the number."""
    word = str(value).strip().lower() if isinstance(value, str) else value
    return 0.0 if word in (None, "none", "off") else None if word == "calibrate" else float(value)


@dataclass
class TrackCoincidenceSettings:
    """What admits a pair of tracks.

    Field defaults: the ``[track.coincidence]`` table of
    wdf/config/defaults.toml.

    :param jitter_s: timing jitter added to the light travel time, s
        (`TrackTimingJitter`).
    :param frequency_tolerance: largest ``|d log2 f|`` between the two tracks'
        f(t), octaves (`TrackFrequencyTolerance`).
    :param light_travel_time_s: overrides the pair's light travel time, s
        (`TrackLightTravelTime`); None reads it from the detectors.
    :param min_common_fraction: least fraction of the shorter track over
        which the two f(t) agree within the tolerance
        (`TrackMinCommonFraction`). The rule is relative because
        a track is as long as its signal: a compact binary's merger lasts a
        few tenths of a second, an inspiral seconds. Half the shorter track
        asks the two detectors to see the same part of the signal, not
        merely to touch.
    :param min_common_cycles: least number of cycles at the tracks'
        frequency over that time (`TrackMinCommonCycles`). The
        coherence of two waveforms is measured on their cycles: over less
        than about two, any two oscillations of similar frequency correlate,
        so a shared span shorter than that says nothing about the phase.
        2 cycles are 50 ms at 40 Hz and 10 ms at 200 Hz.
    :param min_common_s: absolute floor on that time, s
        (`TrackMinCommonDuration`; 0: none).
    :param common_rule: how the three minima combine (`TrackCommonRule`):
        "all" needs every one of them; "duration-or-fraction" needs
        the cycles and either the fraction or the absolute duration (the
        latter only when `min_common_s` is positive), so that two tracks
        sharing a stated length of signal pair even when one of them was
        prolonged by noise and the shared part is less than half of the
        shorter.
    :param slot_tolerance: fraction of a slot by which a slot centre of one
        track may lie outside the other track's first-to-last slot centres
        and still be compared (`TrackSlotTolerance`; 0: none). The
        two detectors' slot grids coincide at zero lag (the same analysis
        grid) and not under a time slide of an arbitrary displacement; with 0
        the comparison reads an aligned pair more generously than a
        misaligned one (the boundary slots), so the slides underestimate the
        background; 0.5 reads both alike, the frequency of the other track
        then taken at its nearest slot.
    :param lag_search_s: half-width of the coherence's lag search, s
        (`TrackLagSearch`); at least the light travel time plus the jitter, or
        "physical": the light travel time plus the jitter itself, so that the
        lag is the best physical one and admission (b) always holds.
    :param coherence_min: least ``|Gamma|`` admitted (`TrackCoherenceMin`):
        0.0 ("none") for no coherence cut, the coherence then
        ranking only; None ("calibrate") sets it with
        `calibrate_coherence_min`.
    :param percentile: the noise percentile the calibration takes
        (`TrackCoherencePercentile`).
    :param calibration_slides: time slides of the calibration
        (`TrackCalibrationSlides`).
    :param calibration_minimum: slid pairs passing (a) and (b) needed; with
        fewer, every slid pair the classical rule forms is used
        (`TrackCalibrationMinimum`).
    """

    jitter_s: float = _TC["TrackTimingJitter"]
    frequency_tolerance: float = _TC["TrackFrequencyTolerance"]
    light_travel_time_s: float = _TC["TrackLightTravelTime"]
    min_common_fraction: float = _TC["TrackMinCommonFraction"]
    min_common_cycles: float = _TC["TrackMinCommonCycles"]
    min_common_s: float = _TC["TrackMinCommonDuration"]
    common_rule: str = _TC["TrackCommonRule"]
    slot_tolerance: float = _TC["TrackSlotTolerance"]
    lag_search_s: object = _TC["TrackLagSearch"]
    coherence_min: float = _coherence_min(_TC["TrackCoherenceMin"])
    percentile: float = _TC["TrackCoherencePercentile"]
    calibration_slides: int = _TC["TrackCalibrationSlides"]
    calibration_minimum: int = _TC["TrackCalibrationMinimum"]

    KEYS = dict(jitter_s="TrackTimingJitter", frequency_tolerance="TrackFrequencyTolerance",
                light_travel_time_s="TrackLightTravelTime",
                min_common_fraction="TrackMinCommonFraction",
                min_common_cycles="TrackMinCommonCycles",
                min_common_s="TrackMinCommonDuration", common_rule="TrackCommonRule",
                slot_tolerance="TrackSlotTolerance", lag_search_s="TrackLagSearch",
                coherence_min="TrackCoherenceMin", percentile="TrackCoherencePercentile",
                calibration_slides="TrackCalibrationSlides",
                calibration_minimum="TrackCalibrationMinimum")

    @classmethod
    def from_parameters(cls, par):
        """The settings a run asks for, defaults filled in and written back.

        `TrackCoherenceMin` is "none" (no cut), "calibrate", or a number;
        unset, the ``[track.coincidence]`` default.
        """
        out = cls()
        for name, key in cls.KEYS.items():
            value = getattr(par, key, None)
            if name == "coherence_min":
                value = _TC["TrackCoherenceMin"] if value is None else value
                word = str(value).strip().lower() if isinstance(value, str) else value
                out.coherence_min = _coherence_min(value)
                setattr(par, key, "none" if out.coherence_min == 0.0 and word in (None, "none", "off")
                        else "calibrate" if out.coherence_min is None else out.coherence_min)
                continue
            if name == "common_rule":
                if value is not None:
                    word = str(value).strip().lower()
                    if word not in COMMON_RULES:
                        raise ValueError(f"TrackCommonRule {value!r}: one of {COMMON_RULES}")
                    out.common_rule = word
            elif name == "lag_search_s" and isinstance(value, str):
                if value.strip().lower() != "physical":
                    raise ValueError(f"TrackLagSearch {value!r}: seconds, or 'physical'")
                out.lag_search_s = "physical"
            elif value is not None:
                value = int(value) if name.startswith("calibration") else float(value)
                setattr(out, name, value)
            setattr(par, key, getattr(out, name))
        if not 0 < out.percentile < PERCENT:
            raise ValueError(f"TrackCoherencePercentile {out.percentile}: between 0 and 100")
        if not 0.0 <= out.slot_tolerance <= 1.0:
            raise ValueError(f"TrackSlotTolerance {out.slot_tolerance}: a fraction of a slot, 0 to 1")
        return out

    def window(self, a, b):
        """Largest lag between detectors `a` and `b`, s.

        :raises ValueError: if `lag_search_s` does not reach it.
        """
        from wdf.analysis.detectors import light_travel_time

        travel = (self.light_travel_time_s if self.light_travel_time_s is not None
                  else light_travel_time(a, b))
        window = float(travel) + float(self.jitter_s)
        if self.lag_search_s != "physical" and float(self.lag_search_s) < window:
            raise ValueError(f"TrackLagSearch {self.lag_search_s} s is shorter than the "
                             f"{a}-{b} light travel time plus jitter, {window:.4f} s")
        return window

    def lag_search(self, a, b):
        """Half-width of the lag search between `a` and `b`, s: `lag_search_s`,
        or the window itself with "physical"."""
        window = self.window(a, b)
        return window if self.lag_search_s == "physical" else float(self.lag_search_s)


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
                        tolerance, min_common_s=0.0, min_common_fraction=0.0,
                        min_common_cycles=0.0, common_rule=_TC["TrackCommonRule"],
                        slot_tolerance=0.0):
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
    :param min_common_s: least time the two f(t) agree within the tolerance.
    :param min_common_fraction: least fraction of the shorter track it is.
    :param min_common_cycles: least number of cycles it holds.
    :param common_rule: "all" (every minimum) or "duration-or-fraction" (the
        cycles, and the fraction or -- when `min_common_s` is positive -- the
        absolute time), see `TrackCoincidenceSettings`.
    :param slot_tolerance: fraction of the left track's slot by which a left
        slot centre may lie outside the right track's centre range and still
        count (0: none, the boundary read exactly; the right track's f(t) is
        interpolated); with a positive value the right track's frequency is
        read at its nearest slot centre, so that the comparison does not
        depend on how the two slot grids are aligned.
    :return: dict -- `track_time_overlap` (the shared span over the shorter
        track), `track_f_distance` (median ``|d log2 f|`` over it, octaves; the
        smallest over the shifts -window, 0, +window), `track_f_overlap` (the
        fraction of the shared slots within the tolerance), `track_common_s`
        (the time they are within it: those slots times the slot),
        `track_common_fraction` (that time over the shorter track's duration),
        `track_common_cycles` (the cycles over it: the sum, over those slots,
        of the slot times the left track's frequency) and `track_compatible`
        (1 when the three minima are met).
    """
    s1 = float(left["track_t_start"]) + displacement_left
    e1 = float(left["track_t_end"]) + displacement_left
    s2 = float(right["track_t_start"]) + displacement_right
    e2 = float(right["track_t_end"]) + displacement_right
    shared = min(e1, e2 + window_s) - max(s1, s2 - window_s)
    shortest = max(min(e1 - s1, e2 - s2), 1e-12)
    out = dict(track_time_overlap=float(np.clip(shared / shortest, 0.0, 1.0)),
               track_f_distance=float("nan"), track_f_overlap=0.0, track_common_s=0.0,
               track_common_fraction=0.0, track_common_cycles=0.0, track_compatible=0.0)
    if shared <= 0:
        return out
    t1 = np.asarray(left["track_times"], dtype=float) + displacement_left
    f1 = np.log2(np.asarray(left["track_f"], dtype=float))
    t2 = np.asarray(right["track_times"], dtype=float) + displacement_right
    f2 = np.log2(np.asarray(right["track_f"], dtype=float))
    slot = float(np.median(np.diff(t1))) if t1.size > 1 else float(e1 - s1)
    pad = float(slot_tolerance) * slot
    best = None
    for shift in (-window_s, 0.0, window_s):
        here = (t1 >= t2[0] + shift - pad) & (t1 <= t2[-1] + shift + pad)
        if not here.any():
            continue
        if pad > 0:
            nearest = np.clip(np.searchsorted(t2, t1[here] - shift), 1, t2.size - 1) if t2.size > 1 else \
                np.zeros(int(here.sum()), int)
            if t2.size > 1:
                left_closer = np.abs(t1[here] - shift - t2[nearest - 1]) <= np.abs(t1[here] - shift - t2[nearest])
                nearest = np.where(left_closer, nearest - 1, nearest)
            d = np.abs(f1[here] - f2[nearest])
        else:
            d = np.abs(f1[here] - np.interp(t1[here] - shift, t2, f2))
        if best is None or np.median(d) < np.median(best):
            best, best_f = d, f1[here]
    if best is None:
        return out
    out["track_f_distance"] = float(np.median(best))
    out["track_f_overlap"] = float(np.mean(best <= tolerance))
    agree = best <= tolerance
    common = float(np.sum(agree) * slot)
    out["track_common_s"] = common
    out["track_common_fraction"] = float(min(common / shortest, 1.0))
    out["track_common_cycles"] = float(np.sum(2.0 ** best_f[agree]) * slot)
    if common_rule not in COMMON_RULES:
        raise ValueError(f"common_rule {common_rule!r}: one of {COMMON_RULES}")
    enough_fraction = out["track_common_fraction"] >= min_common_fraction
    enough_duration = common >= min_common_s
    if common_rule == "all":
        extent = enough_fraction and enough_duration
    else:
        extent = enough_fraction or (min_common_s > 0 and enough_duration)
    out["track_compatible"] = float(common > 0 and extent
                                    and out["track_common_cycles"] >= min_common_cycles)
    return out


def track_event_coefficients(events, waveforms, fs, window, basis=_TC["graph_basis"],
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
    edge = window // _TC["graph_edge_divisor"]
    t_lo, t_hi = coeff_time_bounds(window, fs, basis)
    centre = 0.5 * (t_lo + t_hi) * fs
    central = (centre >= edge) & (centre < window - edge)
    rows, labels = [], []
    ids = events["cluster_id"].astype(int).tolist()
    for label, cid in enumerate(ids):
        t0, y = waveforms.get(cid, (np.nan, np.zeros(0)))
        y = np.asarray(y, dtype=float)
        pad = np.concatenate([np.zeros(edge), y, np.zeros(window)])
        made = 0
        for a in range(0, max(pad.size - window + 1, 1), stride):
            seg = pad[a:a + window]
            if seg.size < window:
                break
            c = np.where(central, local_cosine(seg[None, :], m)[0], 0.0)
            index, value = from_dense(c)
            if not len(index) and made:
                continue
            gps = t0 + (a - edge) / fs
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
    :type gate: str
    :param gate: what is admitted, one of `GATES`; "full" applies
        `settings.coherence_min` (0: no coherence cut), which must be set.
    """

    def __init__(self, finder, events, waveforms, fs, settings=None, scorer=None, builder=None,
                 coefficients=None, comparison=None, gate="full"):
        if gate not in GATES:
            raise ValueError(f"gate {gate!r}: use one of {GATES}")
        if gate == "full" and (settings is None or settings.coherence_min is None):
            raise ValueError("TrackCoherenceMin is to be calibrated: calibrate it on noise "
                             "with calibrate_coherence_min, or give a value (0 for no cut)")
        self.gate = gate
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
        st = self.settings
        window = st.window(ifo_a, ifo_b)
        out = track_compatibility(row_a, row_b, da, db, window, st.frequency_tolerance,
                                  st.min_common_s, st.min_common_fraction,
                                  st.min_common_cycles, st.common_rule, st.slot_tolerance)
        wa = self.waveforms.get(ifo_a, {}).get(ca)
        wb = self.waveforms.get(ifo_b, {}).get(cb)
        if wa is None or wb is None:
            lag = coherence = phase = float("nan")
        else:
            lag, coherence, phase = waveform_coherence((wa[0] + da, wa[1]), (wb[0] + db, wb[1]),
                                                       self.fs, st.lag_search(ifo_a, ifo_b))
        out.update(track_lag_s=lag, track_coherence=coherence, track_phase=phase)
        rho_a = float(np.sqrt(max(float(row_a.get("track_energy", np.nan)), 0.0)))
        rho_b = float(np.sqrt(max(float(row_b.get("track_energy", np.nan)), 0.0)))
        out.update(track_snr_i=rho_a, track_snr_j=rho_b,
                   track_statistic=min(rho_a, rho_b) * coherence,
                   track_lag_physical=float(np.isfinite(lag) and abs(lag) <= window))
        return out

    def _admit(self, table):
        """The rows the gate admits; `dt_s` becomes the coherence lag."""
        if table.empty:
            return table
        keep = np.ones(len(table), bool)
        if self.gate in ("full", "tracks"):
            keep &= (table["track_compatible"].to_numpy(float) == 1.0)
            keep &= (table["track_lag_physical"].to_numpy(float) == 1.0)
        if self.gate == "full":
            keep &= table["track_coherence"].to_numpy(float) >= self.settings.coherence_min
        out = table[keep].reset_index(drop=True)
        out["dt_instant_s"] = out["dt_s"]
        out["dt_s"] = out["track_lag_s"]
        if "candidate_id" in out:
            out["candidate_id"] = np.arange(len(out))
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
            blocks.append(pd.concat([table, pd.DataFrame(feats)], axis=1))
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
        everything = pd.DataFrame(feats)
        extra = everything[TRACK_EDGE_FEATURES]
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
        return pd.concat([table.reset_index(drop=True), everything], axis=1)

    def find(self, events_by_ifo):
        """The admitted pairs with their track features.

        :type events_by_ifo: dict
        :param events_by_ifo: ``{ifo: track event table}``, possibly slid;
            the tracks a later track superseded (`track_supersedes`) are
            skipped.
        :return: pandas.DataFrame -- one row per admitted pair; empty when
            none is.
        """
        from wdf.analysis.track_reconstruction import drop_superseded

        events_by_ifo = {ifo: drop_superseded(ev) for ifo, ev in events_by_ifo.items()}
        if any(ev.empty for ev in events_by_ifo.values()):
            return pd.DataFrame()
        found = (self._classic(events_by_ifo) if self.scorer is None
                 else self._scored(events_by_ifo))
        return self._admit(found)


def calibrate_coherence_min(events, waveforms, fs, settings, segment_bounds, step_s,
                            finder=None):
    """`TrackCoherenceMin` from noise alone: the `settings.percentile` of
    ``|Gamma|`` over time-slid track pairs.

    The events of every detector but the first are slid by
    `settings.calibration_slides` multiples of `step_s` (`TimeSlideFAR`), so
    no pair is a real coincidence; the population is the slid pairs that pass
    admission (a) and (b), the ones the threshold acts on. With fewer than
    `settings.calibration_minimum` of them, every slid pair the classical rule
    forms is used, and the record says so.

    :param events: ``{ifo: track event table}``.
    :param waveforms: ``{ifo: {cluster_id: (t0, waveform)}}``.
    :param fs: sampling rate of the waveforms, Hz.
    :type settings: TrackCoincidenceSettings
    :param segment_bounds: ``{ifo: (start, end)}`` the slides wrap in.
    :param step_s: step between two slides, s (`NetworkSlideStep`).
    :param finder: the classical finder; `IndexedCoincidenceFinder()` when None.
    :return: tuple -- (the threshold, a record: ``pairs``, ``population``
        ("tracks" or "classic"), ``percentile``, ``slides``).
    :raises ValueError: if no slid pair is formed at all.
    """
    from wdf.analysis.robust_events import FARConfig, IndexedCoincidenceFinder, TimeSlideFAR

    finder = IndexedCoincidenceFinder() if finder is None else finder
    open_settings = replace(settings, coherence_min=None)
    values, population, slides = np.zeros(0), None, 0
    for gate in ("tracks", "classic"):
        track_finder = TrackCoincidenceFinder(finder, events, waveforms, fs, open_settings,
                                              gate=gate)
        far = TimeSlideFAR(track_finder, FARConfig(n_slides=settings.calibration_slides,
                                                   min_shift_s=step_s))
        background = far.background_distribution(events, segment_bounds)
        slides = int(background.attrs.get("n_slides", settings.calibration_slides))
        values = (background["track_coherence"].to_numpy(float) if len(background)
                  else np.zeros(0))
        values = values[np.isfinite(values)]
        population = gate
        if values.size >= settings.calibration_minimum:
            break
    if not values.size:
        raise ValueError("no time-slid pair is formed: TrackCoherenceMin cannot be calibrated "
                         "on these events; give it a value")
    value = float(np.percentile(values, settings.percentile))
    return value, dict(pairs=int(values.size), population=population,
                       percentile=settings.percentile, slides=slides)
