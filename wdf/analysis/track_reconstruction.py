"""Stage 2 of the track path: a detector's triggers as seeds of tracks across
windows, and the waveform along each track.

The paper path (`wdf.analysis.detector_graph`) groups a detector's triggers
into events and describes each event by the coefficients its triggers kept.
The track path describes it by a track through the detector's full,
unthresholded cell map (`wdf.analysis.ridge_tracking`): each trigger is a
seed, the track is the local alignment through the seed's loudest tile,
followed back into the past (the inspiral) and forward (the merger), and the
event's waveform is the inverse transform of the coefficients under the track.

The chain:

1. **Seed** (`seed_window`): a trigger becomes a slice of the map, from
   `TrackLookback` seconds before its window to `TrackHorizon` seconds after
   it, and the cell of its loudest coefficient.
2. **Track** (`seeded_tracks`): `offline_ridge` runs on each seed's slice; the
   candidate of highest score whose path passes the seed's cell (within
   `TrackSeedStates` states) is the track. Where none does, the track is the
   anchored alignment through that cell: the same scores, penalty, drift and
   X-drop, followed backward and forward from the cell (`anchored_path`).
   Seeds whose triggers overlap in time and band are one seed group, and two
   tracks that share a cell are merged. The penalty is symmetric in the
   direction of the sweep: a falling track scores as a rising one.
3. **Waveform** (`track_waveform`): the coefficients under the track's cells,
   widened by `RidgeWaveformStates` states, in the basis `TrackBasis`:
   ``LocalCos<M>`` (the map's own coefficients when it is the map's basis,
   recomputed on the whitened stream otherwise) or ``CosinePackets<m>``, the
   cosine-packet best basis (`py4tsa.tsa.CosinePackets`, segments from `m`
   samples, cost `TrackPacketCost`) of the whitened stream over the track's
   span. The waveform is in units of the noise scale.
4. **Events** (`track_events`): one row per track, `TRACK_EVENT_COLUMNS`:
   the columns of `DETECTOR_EVENT_COLUMNS`, measured on the track and its
   waveform, then the track's own columns (`TRACK_COLUMNS`).

The same functions run offline, on the segment's map (`wdfUnitOffline`), and
online, on the rows and coefficients the streaming map holds
(`wdfUnitOnline`, `TrackObserver`). Online the seeds come in batches, one per
drain, and a track of a later batch may share cells with a track already
emitted: `TrackOverlapPolicy` says whether it is dropped ("drop") or, when its
score is higher, emitted in the old tracks' place with their seeds and their
ids in `track_supersedes` ("supersede"; `TrackObserver`). Superseded rows are
kept where they were written and skipped by `load_tracks` and the coincidence
(`drop_superseded`). `TrackEventInstant` chooses the event's instant
(`gpsPeak`): the peak of the waveform's envelope ("peak") or the end of the
track's last slot ("end").
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from wdf.analysis.detector_graph import DETECTOR_EVENT_COLUMNS
from wdf.analysis.ridge import RIDGE_FEATURES, event_ridge_features
from wdf.analysis.ridge_tracking import (RidgeSettings, calibrate_drift, dense_coefficients,
                                         offline_ridge, ridge_waveform, tile_geometry,
                                         window_sigma)

#: Prefix of the cosine-packet track basis, followed by the smallest segment.
COSINE_PACKETS = "CosinePackets"
#: Costs of the cosine-packet best basis (`py4tsa.tsa.CosinePackets.Cost`).
PACKET_COSTS = ("l1", "entropy", "wdfUniversal", "wdfBlock")

#: The track's own columns, after those of `DETECTOR_EVENT_COLUMNS`.
TRACK_COLUMNS = [
    "track_t_start", "track_t_end", "track_times", "track_f", "track_energy", "track_snr",
    "track_score", "track_z_sum", "track_n_slots", "track_basis", "track_method",
    "track_n_seeds", "track_seed_gps", "track_waveform_t0", "track_supersedes",
]
#: Columns of a track event table.
TRACK_EVENT_COLUMNS = DETECTOR_EVENT_COLUMNS + TRACK_COLUMNS


# ================================================================ the track's penalty
#: G Msun / c^3, s.
SOLAR_MASS_S = 4.925490947e-6


def chirp_allowance(f, chirp_mass, slot_s):
    """The largest sweep of an inspiral in one slot from frequency `f`,
    octaves: ``df/dt = (96/5) pi^(8/3) (G Mc / c^3)^(5/3) f^(11/3)``
    integrated over `slot_s`, ``f'^(-8/3) = f^(-8/3) - (8/3) K slot_s``;
    infinite where the inspiral would reach coalescence within the slot.

    :param f: frequency, Hz (array).
    :param chirp_mass: chirp mass, solar masses (detector frame).
    :param slot_s: slot, s.
    :return: numpy.ndarray -- octaves.
    """
    f = np.asarray(f, dtype=float)
    k = (96.0 / 5.0) * np.pi ** (8.0 / 3.0) * (chirp_mass * SOLAR_MASS_S) ** (5.0 / 3.0)
    rest = f ** (-8.0 / 3.0) - (8.0 / 3.0) * k * slot_s
    with np.errstate(divide="ignore", invalid="ignore"):
        end = np.where(rest > 0, rest ** (-3.0 / 8.0), np.inf)
    return np.log2(end / f)


@dataclass
class TrackRidgeSettings(RidgeSettings):
    """The map's settings with the track's penalty.

    A move between states s and s' costs ``lam * max(0, |d log2 f| - c)``,
    ``c`` the largest sweep an inspiral of chirp mass up to the top of
    `mc_range` makes in one slot from the LOWER of the two states' centres
    (`chirp_allowance`), at most `allowance_max`. The allowance depends on
    the pair of states only, not on the direction of the move, so the
    penalty is symmetric: a falling track costs as a rising one. A sweep
    grows with the chirp mass, so the top of the range bounds it over the
    range. `lam` is the stiff penalty beyond the allowance (`TrackLambda`).

    :param mc_range: (smallest, largest) chirp mass, solar masses
        (`TrackMcRange`).
    :param allowance_max: largest allowance, octaves per slot
        (`TrackAllowanceMax`), which bounds it near coalescence.
    :param kept_quantile: the percentile of the slots kept into noise that
        the drift calibration holds at `calibration_kept`
        (`TrackCalibrationQuantile`; 50 is the median).
    :param track_calibration_seeds: random starts of the track's drift
        calibration (`TrackCalibrationSeeds`, default 500); the map's own
        calibration takes `calibration_seeds` (`RidgeCalibrationSeeds`).
    """

    mc_range: tuple = (1.0, 30.0)
    allowance_max: float = 0.27
    kept_quantile: float = 50.0
    track_calibration_seeds: int = 500

    def allowances(self):
        """``[from, to]`` allowance, octaves."""
        centre = np.sqrt(self.edges()[:-1] * self.edges()[1:])
        low = np.minimum(centre[:, None], centre[None, :])
        return np.minimum(chirp_allowance(low, max(self.mc_range), self.slot_s),
                          self.allowance_max)

    def penalty(self):
        """``[from, to]`` cost of a move: ``lam * max(0, |d log2 f| - c(f))``."""
        centre = np.log2(self.edges()[:-1]) + 0.5 * self.octave_step
        return self.lam * np.maximum(0.0, np.abs(centre[:, None] - centre[None, :])
                                     - self.allowances())


# ================================================================ settings
@dataclass
class TrackSettings:
    """Every choice of the track reconstruction, read from the run parameters.

    :param lookback: seconds of the map before a seed's window searched for
        its track (`TrackLookback`).
    :param horizon: seconds after it (`TrackHorizon`); online, a track is
        built once this much of the map past the seed's window is complete,
        so it adds to the declared latency.
    :param basis: basis of the waveform (`TrackBasis`): ``LocalCos<M>`` or
        ``CosinePackets<m>``; default the map's `RidgeBasis`.
    :param seed_states: states on either side of the seed's cell the track
        may pass through and still contain it (`TrackSeedStates`).
    :param packet_cost: cost of the cosine-packet best basis
        (`TrackPacketCost`), one of `PACKET_COSTS`.
    :param overlap_policy: online, a track of a later batch sharing cells
        with emitted tracks (`TrackOverlapPolicy`): "drop" = discarded, its
        seeds counted as absorbed; "supersede" = emitted when its score is
        higher than every emitted track it overlaps, carrying their seeds and
        their ids in `track_supersedes`, else discarded.
    :param overlap_memory: online, how many of the most recently emitted
        tracks a later track is checked against for overlap
        (`TrackOverlapMemory`, default 256).
    :param event_instant: the event's instant, `gpsPeak`
        (`TrackEventInstant`): "peak" = centre of the largest sample of the
        waveform's envelope; "end" = end of the track's last slot.
    :param ridge: the map's settings with the track's penalty
        (`TrackRidgeSettings`): `TrackLambda` (default 3.7 per octave),
        `TrackMcRange` (default 1 to 30 solar masses), `TrackAllowanceMax`
        (default 0.27 octave), and the track's own drift and X-drop
        (`TrackDrift`, `TrackXDrop`, default calibrated on the background
        with this penalty).
    """

    lookback: float
    horizon: float
    basis: str
    seed_states: int
    packet_cost: str
    ridge: RidgeSettings
    overlap_policy: str = "drop"
    event_instant: str = "peak"
    overlap_memory: int = 256

    KEYS = dict(lookback="TrackLookback", horizon="TrackHorizon", basis="TrackBasis",
                seed_states="TrackSeedStates", packet_cost="TrackPacketCost",
                overlap_policy="TrackOverlapPolicy", event_instant="TrackEventInstant",
                overlap_memory="TrackOverlapMemory")
    DEFAULTS = dict(lookback=4.0, horizon=1.0, seed_states=1, packet_cost="entropy",
                    overlap_policy="drop", event_instant="peak", overlap_memory=256)
    #: Values of `TrackOverlapPolicy` and `TrackEventInstant`.
    OVERLAP_POLICIES = ("drop", "supersede")
    EVENT_INSTANTS = ("peak", "end")
    #: The track penalty's parameters and defaults: the stiff penalty of
    #: ridge v1, which followed inspirals, beyond a chirp allowance.
    PENALTY = dict(TrackLambda=3.7, TrackMcRange=(1.0, 30.0), TrackAllowanceMax=0.27,
                   TrackDrift="calibrate", TrackXDrop="calibrate", TrackCalibrationQuantile=90.0,
                   TrackCalibrationSeeds=500)

    @classmethod
    def from_parameters(cls, par, rate, ridge=None):
        """The settings a run asks for, defaults filled in and written back.

        :param par: run configuration; it receives every value used.
        :type rate: float
        :param rate: sampling rate of the searched stream, Hz.
        :type ridge: RidgeSettings or None
        :param ridge: the map's settings, read from `par` when None.
        :return: TrackSettings
        :raises ValueError: on a basis or cost that is not known, or a
            negative lookback or horizon.
        """
        ridge = RidgeSettings.from_parameters(par, rate) if ridge is None else ridge
        ridge = cls.track_ridge(par, ridge)

        def get(name, default):
            value = getattr(par, cls.KEYS[name], None)
            return default if value is None else value

        d = cls.DEFAULTS
        out = cls(lookback=float(get("lookback", d["lookback"])),
                  horizon=float(get("horizon", d["horizon"])),
                  basis=str(get("basis", ridge.basis)),
                  seed_states=int(get("seed_states", d["seed_states"])),
                  packet_cost=str(get("packet_cost", d["packet_cost"])), ridge=ridge,
                  overlap_policy=str(get("overlap_policy", d["overlap_policy"])),
                  event_instant=str(get("event_instant", d["event_instant"])),
                  overlap_memory=int(get("overlap_memory", d["overlap_memory"])))
        out.check(rate)
        for name, key in cls.KEYS.items():
            setattr(par, key, getattr(out, name))
        return out

    @classmethod
    def track_ridge(cls, par, ridge):
        """`ridge` with the track's penalty, drift and X-drop, from `par`
        (values written back)."""
        values = {}
        for key, default in cls.PENALTY.items():
            value = getattr(par, key, None)
            values[key] = default if value is None else value
        mc = tuple(float(m) for m in values["TrackMcRange"])
        if len(mc) != 2 or not 0 < mc[0] <= mc[1]:
            raise ValueError(f"TrackMcRange {values['TrackMcRange']}: two positive chirp masses, "
                             "smallest first")
        if float(values["TrackLambda"]) <= 0 or float(values["TrackAllowanceMax"]) < 0:
            raise ValueError("TrackLambda must be positive and TrackAllowanceMax not negative")
        drift, xdrop = values["TrackDrift"], values["TrackXDrop"]
        base = {k: v for k, v in asdict(ridge).items()}
        base.update(lam=float(values["TrackLambda"]),
                    drift=None if drift in (None, "calibrate") else float(drift),
                    xdrop=None if xdrop in (None, "calibrate") else float(xdrop))
        for key, value in values.items():
            setattr(par, key, list(value) if isinstance(value, tuple) else value)
        par.TrackMcRange = list(mc)
        quantile = float(values["TrackCalibrationQuantile"])
        if not 0 < quantile < 100:
            raise ValueError(f"TrackCalibrationQuantile {quantile}: between 0 and 100")
        seeds = int(values["TrackCalibrationSeeds"])
        if seeds < 1:
            raise ValueError(f"TrackCalibrationSeeds {seeds}: at least 1")
        par.TrackCalibrationSeeds = seeds
        return TrackRidgeSettings(**base, mc_range=mc,
                                  allowance_max=float(values["TrackAllowanceMax"]),
                                  kept_quantile=quantile, track_calibration_seeds=seeds)

    def check(self, rate):
        """Refuse settings the reconstruction cannot run."""
        from wdf.analysis.wavelets import LOCAL_COSINE, _local_cosine_segment

        if self.lookback < 0 or self.horizon < 0:
            raise ValueError(f"TrackLookback {self.lookback} and TrackHorizon {self.horizon} "
                             "must not be negative")
        if self.basis.startswith(LOCAL_COSINE):
            m = _local_cosine_segment(self.basis)
            if self.ridge.window % m:
                raise ValueError(f"TrackBasis {self.basis}: M must divide RidgeWindow "
                                 f"{self.ridge.window}")
        elif self.basis.startswith(COSINE_PACKETS):
            packet_segment(self.basis)
        else:
            raise ValueError(f"TrackBasis {self.basis!r}: use LocalCos<M> or "
                             f"{COSINE_PACKETS}<m>")
        if self.packet_cost not in PACKET_COSTS:
            raise ValueError(f"TrackPacketCost {self.packet_cost!r}: use one of {PACKET_COSTS}")
        if self.overlap_policy not in self.OVERLAP_POLICIES:
            raise ValueError(f"TrackOverlapPolicy {self.overlap_policy!r}: use one of "
                             f"{self.OVERLAP_POLICIES}")
        if self.overlap_memory < 1:
            raise ValueError(f"TrackOverlapMemory {self.overlap_memory}: at least 1")
        if self.event_instant not in self.EVENT_INSTANTS:
            raise ValueError(f"TrackEventInstant {self.event_instant!r}: use one of "
                             f"{self.EVENT_INSTANTS}")

    def completion_s(self, rate):
        """Seconds of whitened stream after a slot's end that the map reads
        before the slot is complete: one slot, plus the part of the last
        window reaching past the start of its central stretch,
        ``(RidgeWindow - RidgeOverlap / 2) / rate``."""
        return self.ridge.slot_s + (self.ridge.window - 0.5 * self.ridge.overlap) / float(rate)

    def record(self):
        """The settings as a dict (the map's under ``ridge``)."""
        out = {k: getattr(self, k) for k in self.KEYS}
        out["ridge"] = self.ridge.record()
        return out


def packet_segment(basis):
    """The smallest segment ``m`` of a ``CosinePackets<m>`` name."""
    digits = str(basis)[len(COSINE_PACKETS):]
    if not digits.isdigit() or int(digits) < 2 or int(digits) & (int(digits) - 1):
        raise ValueError(f"TrackBasis {basis!r}: CosinePackets takes a power of 2, "
                         "as CosinePackets32")
    return int(digits)


# ================================================================ seeds
@dataclass
class Seed:
    """A trigger as the start of a track.

    :param gps_peak: GPS centre of its loudest tile.
    :param f_peak: frequency of its loudest tile, Hz.
    :param t_lo: start of the trigger's window (its `gps`).
    :param t_hi: end of the trigger's window.
    :param f_lo: lower edge of its band, Hz (`freqMin`).
    :param f_hi: upper edge, Hz (`freqMax`).
    :param slot_lo: first slot of the slice its track is searched in.
    :param slot_hi: slot after the last.
    :param peak_slot: slot of the loudest tile.
    :param peak_state: state of the loudest tile.
    :param enwdf: the trigger's `EnWDF`.
    :param record: the trigger's record, without its coefficients.
    """

    gps_peak: float
    f_peak: float
    t_lo: float
    t_hi: float
    f_lo: float
    f_hi: float
    slot_lo: int
    slot_hi: int
    peak_slot: int
    peak_state: int
    enwdf: float
    record: dict = field(default_factory=dict)


def _get(trigger, name, default=np.nan):
    value = trigger.get(name, default) if hasattr(trigger, "get") else getattr(trigger, name, default)
    return default if value is None else value


def loudest_tile(trigger):
    """GPS centre and frequency of a trigger's largest coefficient.

    Read from its coefficients (`wt_index`, `wt_value`, laid out in its basis
    `wave`) when it carries them, from `gpsPeak` and `freqMean` otherwise.

    :return: tuple -- (GPS, Hz).
    """
    from wdf.analysis.wavelets import coeff_freq_bands, coeff_time_bounds

    index = _get(trigger, "wt_index", None)
    value = _get(trigger, "wt_value", None)
    if index is not None and value is not None and len(index):
        index = np.asarray(index, dtype=int)
        value = np.asarray(value, dtype=float)
        n, fs = int(_get(trigger, "n_coeff")), float(_get(trigger, "fs"))
        wave = _get(trigger, "wave", None)
        t_lo, t_hi = coeff_time_bounds(n, fs, wave)
        f_lo, f_hi = coeff_freq_bands(n, fs, wave)
        k = index[int(np.argmax(np.abs(value)))]
        f = np.sqrt(f_lo[k] * f_hi[k]) if f_lo[k] > 0 else 0.5 * f_hi[k]
        return float(_get(trigger, "gps")) + 0.5 * (t_lo[k] + t_hi[k]), float(f)
    return float(_get(trigger, "gpsPeak")), float(_get(trigger, "freqMean"))


def seed_window(trigger, settings, origin):
    """A trigger mapped onto the map: the slice its track is searched in and
    the cell of its loudest tile.

    The slice runs from `lookback` seconds before the trigger's window to
    `horizon` seconds after its end.

    :param trigger: the trigger's record (dict or row): `gps`, `n_coeff`,
        `fs`, `EnWDF`, `freqMin`, `freqMax`, and its coefficients or
        `gpsPeak` and `freqMean`.
    :type settings: TrackSettings
    :param settings: lookback, horizon, the map's slots and states.
    :type origin: float
    :param origin: GPS start of the map's slot 0.
    :return: Seed
    """
    rs = settings.ridge
    t_lo = float(_get(trigger, "gps"))
    t_hi = t_lo + float(_get(trigger, "n_coeff")) / float(_get(trigger, "fs"))
    gps_peak, f_peak = loudest_tile(trigger)
    edges = rs.edges()
    state = int(np.clip(np.searchsorted(edges, f_peak, side="right") - 1, 0, edges.size - 2))
    record = {k: v for k, v in (trigger.items() if hasattr(trigger, "items") else [])
              if k not in ("wt_index", "wt_value")}
    return Seed(gps_peak=gps_peak, f_peak=f_peak, t_lo=t_lo, t_hi=t_hi,
                f_lo=float(_get(trigger, "freqMin")), f_hi=float(_get(trigger, "freqMax")),
                slot_lo=int(np.floor((t_lo - settings.lookback - origin) / rs.slot_s + 1e-9)),
                slot_hi=int(np.ceil((t_hi + settings.horizon - origin) / rs.slot_s - 1e-9)),
                peak_slot=int(np.floor((gps_peak - origin) / rs.slot_s + 1e-9)),
                peak_state=state, enwdf=float(_get(trigger, "EnWDF")), record=record)


def group_seeds(seeds):
    """Seeds whose triggers overlap in time and in band, as groups.

    :type seeds: list[Seed]
    :return: list[list[Seed]] -- each group in time order.
    """
    order = sorted(seeds, key=lambda s: (s.t_lo, s.t_hi))
    groups = []
    for s in order:
        for g in groups:
            if any(s.t_lo < o.t_hi and o.t_lo < s.t_hi
                   and not (s.f_hi < o.f_lo or o.f_hi < s.f_lo) for o in g):
                g.append(s)
                break
        else:
            groups.append([s])
    return groups


# ================================================================ tracks
@dataclass
class Track:
    """A path through a detector's cell map, seeded by triggers.

    :param slots: slot indices of the map, consecutive.
    :param states: one state per slot.
    :param score: the alignment's score.
    :param z_sum: sum of z over the path's cells.
    :param cells: each cell's energy (NaN where not known).
    :param t_start: GPS start of the first slot.
    :param t_end: GPS end of the last slot.
    :param f_low: lower edge of each slot's state, Hz.
    :param f_high: upper edge, Hz.
    :param seeds: the seeds it contains.
    :param method: "ridge" (an `offline_ridge` candidate) or "anchored".
    :param waveform: (GPS of the first sample, samples, energy, basis), once
        reconstructed.
    """

    slots: np.ndarray
    states: np.ndarray
    score: float
    z_sum: float
    cells: np.ndarray
    t_start: float
    t_end: float
    f_low: np.ndarray
    f_high: np.ndarray
    seeds: list
    method: str
    waveform: tuple = None

    def contains(self, slot, state, tolerance):
        """Whether the path passes cell (slot, state) within `tolerance` states."""
        hit = self.slots == int(slot)
        return bool(np.any(np.abs(self.states[hit] - int(state)) <= int(tolerance)))

    def times(self):
        """Centre of each slot, GPS."""
        return self.t_start + (np.arange(self.slots.size) + 0.5) * (self.t_end - self.t_start) / self.slots.size

    def frequencies(self):
        """Geometric centre of each slot's state, Hz."""
        return np.sqrt(self.f_low * self.f_high)


def _directed(Z, k0, s0, settings, pen, direction, stop):
    """The best path leaving cell (k0, s0) in one direction, X-drop."""
    ns = Z.shape[1]
    H = np.full(ns, -np.inf)
    H[s0] = 0.0
    back, best, best_step, best_state = [], 0.0, 0, s0
    for step in range(1, int(settings.max_slots) + 1):
        k = k0 + direction * step
        if k < 0 or k >= stop or not np.all(np.isfinite(Z[k])):
            break
        cand = H[:, None] - pen
        arg = np.argmax(cand, axis=0)
        H = Z[k] - settings.drift + cand[arg, np.arange(ns)]
        back.append(arg)
        s = int(np.argmax(H))
        if H[s] > best:
            best, best_step, best_state = float(H[s]), step, s
        if H[s] < best - settings.xdrop:
            break
    path, s = [], best_state
    for step in range(best_step, 0, -1):
        path.append((k0 + direction * step, s))
        s = int(back[step - 1][s])
    return path[::-1] if direction > 0 else path, best


def anchored_path(Z, k0, s0, settings):
    """The alignment through cell (k0, s0): the best path backward and the
    best path forward from it, each ended by the X-drop.

    :type Z: numpy.ndarray
    :param Z: z map.
    :param k0: the cell's slot.
    :param s0: its state.
    :type settings: RidgeSettings
    :param settings: penalty, drift, X-drop, max_slots.
    :return: tuple -- (slots, states, score).
    """
    pen = settings.penalty()
    forward, f_best = _directed(Z, k0, s0, settings, pen, +1, len(Z))
    backward, b_best = _directed(Z, k0, s0, settings, pen.T, -1, len(Z))
    cells = backward + [(k0, s0)] + forward
    slots = np.array([c[0] for c in cells], dtype=int)
    states = np.array([c[1] for c in cells], dtype=int)
    return slots, states, float(Z[k0, s0] - settings.drift + f_best + b_best)


def _make_track(slots, states, score, Z, X, origin, rs, seeds, method):
    edges = rs.edges()
    cells = (np.full(slots.size, np.nan) if X is None
             else np.asarray(X, dtype=float)[slots, states])
    return Track(slots=slots, states=states, score=float(score),
                 z_sum=float(np.sum(Z[slots, states])), cells=cells,
                 t_start=origin + slots[0] * rs.slot_s, t_end=origin + (slots[-1] + 1) * rs.slot_s,
                 f_low=edges[states], f_high=edges[states + 1], seeds=list(seeds), method=method)


def calibrate_track_drift(Z, settings, seed=1):
    """The drift b and the X-drop of the track's alignment, from noise.

    As `calibrate_drift`, with the alignment the track uses (the best path
    over every state sequence from a start, `anchored_path`) in place of the
    greedy extension: from `track_calibration_seeds` random starts
    (`TrackCalibrationSeeds`) in the background, at the start's loudest state, the best score after k
    slots is followed for `calibration_steps` slots. X is the
    `xdrop_percentile` of its largest drop within `xdrop_slots` slots, and b
    the smallest drift of `calibration_grid` for which the `kept_quantile`
    percentile of the slots kept before a drop of X is at most
    `calibration_kept`. A track's ends are where its alignment stops gaining,
    so the higher that percentile, the less a track runs on into the noise
    around a signal.

    :return: tuple -- (b, X).
    """
    Z = np.asarray(Z, dtype=float)
    Z = Z[np.all(np.isfinite(Z), axis=1)]
    steps = settings.calibration_steps
    if len(Z) < steps + 2:
        raise ValueError(f"{len(Z)} slots of background cannot calibrate the drift over {steps}")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(Z) - steps - 1, settings.track_calibration_seeds)
    pen = settings.penalty()
    s0 = np.argmax(Z[starts], axis=1)
    result = None
    for b in settings.calibration_grid:
        H = np.full((starts.size, Z.shape[1]), -np.inf)
        H[np.arange(starts.size), s0] = 0.0
        best = np.zeros((starts.size, steps + 1))
        for k in range(1, steps + 1):
            H = Z[starts + k] - b + (H[:, :, None] - pen[None]).max(axis=1)
            best[:, k] = H.max(axis=1)
        drops = np.zeros(starts.size)
        for d in range(1, settings.xdrop_slots + 1):
            drops = np.maximum(drops, (best[:, :-d] - best[:, d:]).max(axis=1))
        X = float(np.percentile(drops, settings.xdrop_percentile))
        peak = np.maximum.accumulate(best, axis=1)
        stopped = best < peak - X
        stop = np.where(stopped.any(1), stopped.argmax(1), best.shape[1])
        masked = np.where(np.arange(best.shape[1])[None, :] < stop[:, None], best, -np.inf)
        kept = np.argmax(masked, axis=1)
        result = (float(b), X)
        if np.percentile(kept, getattr(settings, "kept_quantile", 50.0)) <= settings.calibration_kept:
            break
    return result


def calibrated(settings, Z):
    """The ridge settings with drift and X-drop, calibrated on `Z` where they
    are left to calibrate: by `calibrate_track_drift` for the track's
    penalty (`TrackRidgeSettings`), by `calibrate_drift` otherwise."""
    rs = settings.ridge
    if rs.drift is None or rs.xdrop is None:
        calibrate = calibrate_track_drift if isinstance(rs, TrackRidgeSettings) else calibrate_drift
        b, x = calibrate(Z, rs)
        rs = replace(rs, drift=rs.drift if rs.drift is not None else b,
                     xdrop=rs.xdrop if rs.xdrop is not None else x)
    return replace(settings, ridge=rs)


def seeded_tracks(Z, X, origin, seeds, settings, first_slot=0):
    """The tracks of a detector's seeds.

    Seeds overlapping in time and band form one group, whose slice is the
    union of theirs and whose cell is that of its loudest trigger. On each
    group's slice `offline_ridge` runs; its candidate of highest score passing
    the cell within `seed_states` states is the track, the anchored alignment
    through the cell (`anchored_path`) where none does. Tracks sharing a cell
    are then merged into the one of higher score.

    :type Z: numpy.ndarray
    :param Z: z map, ``(slots, states)``; NaN rows are not scored.
    :type X: numpy.ndarray or None
    :param X: the cells' energy, same shape.
    :type origin: float
    :param origin: GPS start of slot 0 of the map the seeds were placed on.
    :type seeds: list[Seed]
    :param seeds: as `seed_window` returns them.
    :type settings: TrackSettings
    :param settings: the reconstruction's settings; a drift or X-drop left to
        calibrate is calibrated on `Z`.
    :type first_slot: int
    :param first_slot: the map slot `Z[0]` is (online, the rows held start
        later than slot 0).
    :return: list[Track] -- by decreasing score, slots in map indices.
    """
    Z = np.asarray(Z, dtype=float)
    if not seeds or Z.size == 0:
        return []
    settings = calibrated(settings, Z)
    rs = settings.ridge
    n = len(Z)
    tracks = []
    for group in group_seeds(seeds):
        head = max(group, key=lambda s: s.enwdf)
        a = max(min(s.slot_lo for s in group) - first_slot, 0)
        b = min(max(s.slot_hi for s in group) - first_slot, n)
        k0 = head.peak_slot - first_slot
        if not (a <= k0 < b) or not np.all(np.isfinite(Z[k0])):
            continue
        slice_origin = origin + (first_slot + a) * rs.slot_s
        found, _ = offline_ridge(Z[a:b], slice_origin, rs, None if X is None else X[a:b])
        chosen = None
        for c in found:
            hit = c.slots == (k0 - a)
            if np.any(np.abs(c.states[hit] - head.peak_state) <= settings.seed_states):
                chosen = c
                break
        if chosen is not None:
            slots, states, score, method = chosen.slots + a, chosen.states, chosen.score, "ridge"
        else:
            slots, states, score = anchored_path(Z[a:b], k0 - a, head.peak_state, rs)
            slots, method = slots + a, "anchored"
        track = _make_track(slots, states, score, Z, X, origin + first_slot * rs.slot_s, rs,
                            group, method)
        track.slots = track.slots + first_slot
        tracks.append(track)
    return merge_tracks(tracks)


def merge_tracks(tracks):
    """Tracks that share a cell, merged into the one of higher score, which
    takes the other's seeds.

    :return: list[Track] -- by decreasing score.
    """
    kept = []
    for t in sorted(tracks, key=lambda t: -t.score):
        cells = set(zip(t.slots.tolist(), t.states.tolist()))
        for k in kept:
            if cells & set(zip(k.slots.tolist(), k.states.tolist())):
                k.seeds.extend(t.seeds)
                break
        else:
            kept.append(t)
    return kept


# ================================================================ waveforms
def _inverse(c, basis, fs, alpha=0.0):
    """One window from its coefficients in `basis`."""
    from wdf.analysis.wavelets import LOCAL_COSINE, _local_cosine_segment

    if str(basis).startswith(LOCAL_COSINE):
        from wdf.analysis.dechirp import inverse_dechirped_local_cosine
        return inverse_dechirped_local_cosine(c, fs, alpha, _local_cosine_segment(basis))
    from wdf.analysis.reconstruction import inverse_transform
    return inverse_transform(c, basis)


def held_stream(coefficients, starts, fs, settings):
    """The whitened stream the map's windows describe: every window inverted
    whole and its central stretch laid down.

    :type coefficients: numpy.ndarray
    :param coefficients: dense coefficients of consecutive map windows.
    :type starts: numpy.ndarray
    :param starts: GPS first sample of each window.
    :type fs: float
    :param fs: sampling rate, Hz.
    :type settings: RidgeSettings
    :param settings: the map's basis, window and overlap.
    :return: tuple -- (GPS of the first sample, samples).
    """
    C = np.atleast_2d(np.asarray(coefficients, dtype=float))
    starts = np.asarray(starts, dtype=float)
    if not len(starts):
        return float("nan"), np.zeros(0)
    n, half = settings.window, settings.overlap // 2
    t0 = starts[0] + half / fs
    size = int(round((starts[-1] - starts[0]) * fs)) + n - 2 * half
    x = np.zeros(size)
    for c, s in zip(C, starts):
        w = _inverse(c, settings.basis, fs)
        a = int(round((s - starts[0]) * fs))
        x[a:a + n - 2 * half] = w[half:n - half]
    return t0, x


def _windows_over(stream, t_start, t_end, fs, rs):
    """Map windows of the stream covering [t_start, t_end): their samples and
    GPS starts."""
    t0, x = stream
    stride = rs.window - rs.overlap
    half = rs.overlap // 2
    first = int(np.floor(((t_start - t0) * fs - half) / stride)) - 1
    last = int(np.ceil(((t_end - t0) * fs - half) / stride)) + 1
    offs = [o for o in range(first * stride, last * stride + 1, stride)
            if o >= 0 and o + rs.window <= x.size]
    if not offs:
        return np.zeros((0, rs.window)), np.zeros(0)
    offs = np.asarray(offs)
    return x[offs[:, None] + np.arange(rs.window)[None, :]], t0 + offs / fs


def _packet_waveform(track, stream, fs, settings):
    """The waveform under the track in the cosine-packet best basis of the
    stream over its span."""
    from py4tsa.tsa import CosinePackets

    rs = settings.ridge
    m = packet_segment(settings.basis)
    t0, x = stream
    need = int(np.ceil((track.t_end - track.t_start) * fs)) + 4 * m
    n = 1 << int(np.ceil(np.log2(max(need, 2 * m))))
    centre = int(round((0.5 * (track.t_start + track.t_end) - t0) * fs))
    a = centre - n // 2
    piece = np.zeros(n)
    lo, hi = max(a, 0), min(a + n, x.size)
    if hi > lo:
        piece[lo - a:hi - a] = x[lo:hi]
    cp = CosinePackets(n, m)
    fine = np.asarray(cp.Analyse(piece, [m] * (n // m)), dtype=float).reshape(n // m, m)
    above = (np.arange(m) + 1) * fs / (2.0 * m) > rs.f_low + 1e-9
    sigma = max(float(np.median(np.abs(fine[:, above])) / 0.6745), 1e-300)
    cp.BestBasis(piece / sigma, getattr(CosinePackets.Cost, settings.packet_cost), 1.0)
    seg = [int(s) for s in cp.GetSegmentation()]
    c = np.array(cp.GetCoefficients(), dtype=float)
    edges = rs.edges()
    ns = edges.size - 1
    margin = rs.waveform_states
    lo_s = edges[np.clip(track.states - margin, 0, ns - 1)]
    hi_s = edges[np.clip(track.states + margin, 0, ns - 1) + 1]
    slot_t0 = track.t_start + np.arange(track.slots.size) * rs.slot_s
    keep = np.zeros(n, bool)
    pos = 0
    start_gps = t0 + a / fs
    for length in seg:
        tl = start_gps + pos / fs
        th = tl + length / fs
        k = np.arange(length)
        fl, fh = k * fs / (2.0 * length), (k + 1) * fs / (2.0 * length)
        inside = (slot_t0 < th) & (slot_t0 + rs.slot_s > tl)
        sel = np.zeros(length, bool)
        for j in np.flatnonzero(inside):
            sel |= (fh > lo_s[j]) & (fl < hi_s[j])
        keep[pos:pos + length] = sel
        pos += length
    y = np.asarray(cp.Inverse(np.where(keep, c, 0.0), seg), dtype=float)
    i0 = int(round((track.t_start - start_gps) * fs))
    size = int(np.ceil((track.t_end - track.t_start) * fs))
    out = np.zeros(size)
    j0, j1 = max(i0, 0), min(i0 + size, n)
    if j1 > j0:
        out[j0 - i0:j1 - i0] = y[j0:j1]
    return track.t_start, out, float(np.sum(out ** 2))


def track_waveform(track, fs, origin, settings, coefficients=None, sigmas=None, starts=None,
                   stream=None):
    """The waveform under a track, in `settings.basis`, in noise units.

    ``LocalCos<M>`` equal to the map's basis reads the map's own coefficients
    (`ridge_waveform`); another ``LocalCos<M>`` recomputes them on the
    whitened stream over the map's windows; ``CosinePackets<m>`` takes the
    best basis of the stream over the track's span. The stream is `stream`
    when given, else the map's windows inverted (`held_stream`).

    :type track: Track
    :param track: the path.
    :type fs: float
    :param fs: sampling rate, Hz.
    :type origin: float
    :param origin: GPS start of the map's slot 0.
    :type settings: TrackSettings
    :param settings: the basis and the map's settings (drift and X-drop
        unused).
    :param coefficients: the map windows' dense coefficients.
    :param sigmas: each window's noise scale.
    :param starts: each window's first sample, GPS.
    :param stream: (GPS of the first sample, whitened samples), or None.
    :return: tuple -- (GPS of the first sample, waveform, energy, basis);
        the waveform is empty where the data held do not cover the track.
    """
    rs = settings.ridge
    basis = settings.basis
    if basis == rs.basis and coefficients is not None and len(starts):
        if starts[0] > track.t_start or starts[-1] + rs.window / fs < track.t_end:
            return track.t_start, np.zeros(0), float("nan"), basis
        t0, y, energy = ridge_waveform(track, coefficients, sigmas, starts, fs, origin, rs)
        return t0, y, energy, basis
    if stream is None:
        if coefficients is None or not len(starts):
            return track.t_start, np.zeros(0), float("nan"), basis
        stream = held_stream(coefficients, starts, fs, rs)
    t0, x = stream
    if t0 > track.t_start or t0 + x.size / fs < track.t_end:
        return track.t_start, np.zeros(0), float("nan"), basis
    if basis.startswith(COSINE_PACKETS):
        return _packet_waveform(track, stream, fs, settings) + (basis,)
    other = replace(rs, basis=basis)
    windows, wstarts = _windows_over(stream, track.t_start, track.t_end, fs, other)
    if not len(wstarts):
        return track.t_start, np.zeros(0), float("nan"), basis
    C = dense_coefficients(windows, basis, fs)
    g = tile_geometry(other.window, fs, basis)
    sig = window_sigma(C, g["f_hi"], other.f_low)
    t0, y, energy = ridge_waveform(track, C, sig, wstarts, fs, origin, other)
    return t0, y, energy, basis


# ================================================================ events
def _quantile(values, weights, q):
    order = np.argsort(values)
    c = np.cumsum(weights[order])
    if not c.size or c[-1] <= 0:
        return float("nan")
    return float(values[order][np.searchsorted(c, q * c[-1])])


def track_row(track, cluster_id, ifo, fs, instant="peak", supersedes=()):
    """One track as a row of `TRACK_EVENT_COLUMNS`.

    The time columns are moments of the waveform's energy (`gpsCentroid`,
    `tSpread`, `duration90`) and the event's instant `gpsPeak`: the centre of
    the waveform's largest envelope sample (`instant` "peak") or the end of
    the track's last slot ("end"); `gpsStart`, `gps` and `duration` are the
    track's span. `supersedes` are the ids of the emitted tracks this one
    replaces (`track_supersedes`, empty but online under "supersede"). The
    frequency columns are the path's states, weighted by the cells' energy.
    `EnWDF` is the square root of the waveform's energy (`track_snr`);
    `EnWDF_window`, `sigma`, `snrPeak`, `significance`, `n_coeff` and
    `scale_best` are those of its loudest seed.
    """
    from scipy.signal import hilbert

    seeds = [s.record for s in track.seeds]
    head = max(track.seeds, key=lambda s: s.enwdf).record if track.seeds else {}
    t0, y, energy, basis = track.waveform if track.waveform is not None else (
        track.t_start, np.zeros(0), float("nan"), "")
    f = track.frequencies()
    w = track.cells if np.all(np.isfinite(track.cells)) else np.ones(f.size)
    w = np.maximum(w, 0.0)
    if y.size and np.sum(y ** 2) > 0:
        e = y ** 2
        t = t0 + np.arange(y.size) / fs
        centroid = float(np.sum(t * e) / np.sum(e))
        spread = float(np.sqrt(np.sum((t - centroid) ** 2 * e) / np.sum(e)))
        peak = float(t[int(np.argmax(np.abs(hilbert(y))))])
        c = np.cumsum(e) / np.sum(e)
        d90 = float((np.searchsorted(c, 0.95) - np.searchsorted(c, 0.05)) / fs)
    else:
        centroid = spread = peak = d90 = float("nan")
    if instant == "end":
        peak = float(track.t_end)
    elif instant != "peak":
        raise ValueError(f"TrackEventInstant {instant!r}: use one of "
                         f"{TrackSettings.EVENT_INSTANTS}")
    snr = float(np.sqrt(energy)) if np.isfinite(energy) else float("nan")
    half = 0.5 * (track.t_end - track.t_start) / track.slots.size
    ridge = event_ridge_features(track.times() - half, track.times() + half,
                                 track.f_low, track.f_high, w)
    n_coeffs = {int(s.get("n_coeff", 0)) for s in seeds if s.get("n_coeff") is not None}
    enwdf_window = max((float(s.get("EnWDF", np.nan)) for s in seeds), default=float("nan"))
    row = dict(
        cluster_id=int(cluster_id), ifo=ifo, gps=track.t_start, gpsStart=track.t_start,
        gpsCentroid=centroid, tSpread=spread, gpsPeak=peak,
        duration=track.t_end - track.t_start, duration90=d90,
        freqMin=float(track.f_low.min()), freqMean=float(np.exp(np.average(np.log(f), weights=w + 1e-300))),
        freqMax=float(track.f_high.max()), freqQ05=_quantile(f, w, 0.05), freqQ95=_quantile(f, w, 0.95),
        EnWDF=snr, sigma=float(head.get("sigma", np.nan)),
        snrPeak=max((float(s.get("snrPeak", np.nan)) for s in seeds), default=float("nan")),
        significance=float(np.log1p(enwdf_window)), n_pixels=int(track.slots.size),
        n_triggers=len(seeds), n_scales=len(n_coeffs),
        scale_best=int(head.get("n_coeff", 0) or 0), n_coeff=int(head.get("n_coeff", 0) or 0),
        fs=float(fs), EnWDF_window=enwdf_window,
        track_t_start=track.t_start, track_t_end=track.t_end,
        track_times=track.times().tolist(), track_f=f.tolist(), track_energy=float(energy),
        track_snr=snr, track_score=track.score, track_z_sum=track.z_sum,
        track_n_slots=int(track.slots.size), track_basis=basis, track_method=track.method,
        track_n_seeds=len(seeds), track_seed_gps=[float(s.get("gps", np.nan)) for s in seeds],
        track_waveform_t0=float(t0), track_supersedes=[int(i) for i in supersedes])
    row.update({k: ridge.get(k, float("nan")) for k in RIDGE_FEATURES})
    return row


def track_events(tracks, ifo, fs, first_id=0, instant="peak"):
    """A detector's tracks as its event table.

    :type tracks: list[Track]
    :param tracks: tracks with their waveform (`track_waveform`).
    :type ifo: str
    :param ifo: the detector.
    :type fs: float
    :param fs: sampling rate of the waveforms, Hz.
    :type first_id: int
    :param first_id: `cluster_id` of the first track.
    :type instant: str
    :param instant: the event's instant (`TrackEventInstant`, `track_row`).
    :return: pandas.DataFrame -- `TRACK_EVENT_COLUMNS`, one row per track.
    """
    rows = [track_row(t, first_id + i, ifo, fs, instant) for i, t in enumerate(tracks)]
    return pd.DataFrame(rows, columns=TRACK_EVENT_COLUMNS)


def superseded_ids(events):
    """The `cluster_id` values named in the `track_supersedes` of a track
    event table: tracks replaced by a later emission (`TrackOverlapPolicy`
    "supersede"), to be skipped by the coincidence.

    :type events: pandas.DataFrame
    :return: set[int]
    """
    out = set()
    if events is None or "track_supersedes" not in events:
        return out
    for value in events["track_supersedes"]:
        if isinstance(value, str):
            import json
            value = json.loads(value) if value else []
        if isinstance(value, (list, tuple, np.ndarray)):
            out.update(int(i) for i in value)
    return out


def drop_superseded(events):
    """`events` without the rows a later track superseded (`superseded_ids`).

    :type events: pandas.DataFrame
    :return: pandas.DataFrame -- the same table when nothing is superseded.
    """
    gone = superseded_ids(events)
    if not gone or "cluster_id" not in events:
        return events
    keep = ~events["cluster_id"].astype(int).isin(gone)
    return events[keep.to_numpy()].reset_index(drop=True)


def track_waveforms(tracks, first_id=0):
    """``{cluster_id: (GPS of the first sample, waveform)}`` of the tracks, in
    the order `track_events` numbers them."""
    return {first_id + i: (float(t.waveform[0]), np.asarray(t.waveform[1]))
            for i, t in enumerate(tracks) if t.waveform is not None}


def reconstruct(triggers, Z, X, origin, fs, settings, ifo, coefficients=None, sigmas=None,
                starts=None, stream=None, first_slot=0, first_id=0):
    """Stage 2 of the track path for one detector: seeds, tracks, waveforms,
    events.

    :param triggers: the detector's triggers with their coefficients
        (pandas.DataFrame or list of records).
    :param Z: z map; ``X`` the cells' energy; ``origin`` GPS of slot 0;
        ``first_slot`` the slot `Z[0]` is.
    :type settings: TrackSettings
    :return: tuple -- (event table, ``{cluster_id: (t0, waveform)}``,
        tracks).
    """
    records = (triggers.to_dict("records") if isinstance(triggers, pd.DataFrame)
               else list(triggers))
    seeds = [seed_window(r, settings, origin) for r in records]
    tracks = seeded_tracks(Z, X, origin, seeds, settings, first_slot=first_slot)
    for t in tracks:
        t.waveform = track_waveform(t, fs, origin, settings, coefficients, sigmas, starts, stream)
    return (track_events(tracks, ifo, fs, first_id, settings.event_instant),
            track_waveforms(tracks, first_id), tracks)


def save_tracks(events, waveforms, csv_path, npz_path):
    """Write a track event table (list columns as JSON) and its waveforms."""
    import json

    out = events.copy()
    for column in ("track_times", "track_f", "track_seed_gps"):
        out[column] = [json.dumps(list(map(float, v))) for v in out[column]]
    if "track_supersedes" in out:
        out["track_supersedes"] = [json.dumps([int(i) for i in v]) if isinstance(v, (list, tuple, np.ndarray))
                                   else "[]" for v in out["track_supersedes"]]
    out.to_csv(csv_path, index=False)
    arrays = {}
    for cid, (t0, y) in waveforms.items():
        arrays[f"t0_{cid}"] = np.array([t0])
        arrays[f"y_{cid}"] = np.asarray(y)
    np.savez(npz_path, **arrays)


def load_tracks(csv_path, npz_path=None, superseded="drop"):
    """Read what `save_tracks` wrote.

    :type superseded: str
    :param superseded: "drop" = the tracks a later track superseded
        (`track_supersedes`) are left out, with their waveforms, so that a
        signal yields one track; "keep" = every row written.
    :return: tuple -- (event table, ``{cluster_id: (t0, waveform)}``).
    """
    import json
    import os

    if superseded not in ("drop", "keep"):
        raise ValueError(f"superseded {superseded!r}: 'drop' or 'keep'")
    events = pd.read_csv(csv_path) if os.path.getsize(csv_path) > 1 else pd.DataFrame(
        columns=TRACK_EVENT_COLUMNS)
    for column in ("track_times", "track_f", "track_seed_gps"):
        if column in events:
            events[column] = [json.loads(v) if isinstance(v, str) else v for v in events[column]]
    if "track_supersedes" in events:
        events["track_supersedes"] = [[int(i) for i in json.loads(v)] if isinstance(v, str) and v
                                      else (list(v) if isinstance(v, (list, tuple)) else [])
                                      for v in events["track_supersedes"]]
    else:
        events["track_supersedes"] = [[] for _ in range(len(events))]
    if superseded == "drop":
        events = drop_superseded(events)
    ids = set(events["cluster_id"].astype(int)) if len(events) else set()
    waveforms = {}
    if npz_path is not None and os.path.exists(npz_path):
        z = np.load(npz_path)
        for key in z.files:
            if key.startswith("t0_"):
                cid = int(key[3:])
                if superseded == "drop" and cid not in ids:
                    continue
                waveforms[cid] = (float(z[key][0]), z[f"y_{cid}"])
    return events, waveforms
