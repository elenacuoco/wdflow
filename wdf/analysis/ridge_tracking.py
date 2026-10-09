"""Following a track across windows: cell maps, a robust per-row z, and a path.

`wdf.analysis.ridge` describes the track of an event already found; this
module finds tracks. A transient whose energy per window is too small to fire
the search can still stand out when it is summed along its time-frequency
track, which is what the path does.

The chain, the same offline and online:

1. **Cells.** The dense coefficients of each analysis window (in a basis such
   as `LocalCos128`, or the dechirped local cosine) give tiles, each with its
   energy over the window's noise scale, ``(c / sigma)^2``
   (`window_tiles`). A tile's energy is shared among the cells it overlaps,
   slots of `slot_s` seconds by states of `octave_step` octaves from
   `f_low`, in proportion to the overlap in time and in log frequency
   (`cell_sums`). Only the central stretch of each window is used (`keep`),
   so that a slot is covered once and the window's circular edges are not.
2. **Background.** Each state row is standardised by its sigma-clipped mean
   and standard deviation over a rolling background (`background_z`): centred on the slot offline,
   strictly behind it online (`causal=True`), recomputed every
   `background_update_s`.
3. **Network.** Detectors are combined cell by cell by the minimum of their
   z (`combine_min`): a cell stands out only where every detector has it, so
   one detector's glitch does not make a track.
4. **Path.** A local alignment (Smith-Waterman) over the rows: each cell
   scores ``z - b``, ``b`` the drift that keeps a path from growing in noise,
   and a move between slots costs ``lam * max(0, |d log2 f| - c)``, symmetric
   up and down, with the free allowance ``c`` (`StreamingRidge`). Scores
   start anew wherever they would fall below zero. An episode is the run of
   slots from where the best path starts growing to where its score drops
   `xdrop` below its maximum; its candidate is the path that reached the
   maximum.

Two variants share that path: `offline_ridge` runs it over a whole segment and
then extends every candidate greedily backward and forward with the same
X-drop; `StreamingRidge` runs it slot by slot, using only the cells up to the
current slot, and keeps a running best path ending at the current slot. Its
backward extension reaches into the past only.

`ridge_waveform` inverts the coefficients under a path's cells and returns the
waveform and its integrated energy.

Ported from the catalogue's scripts/ridge_stage.py (cells, rolling z, DP) and
scripts/ridge_v2.py (local alignment, allowance, drift, X-drop extension).
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from wdf.config.constants import MAD_GAUSSIAN, MAD_TO_SIGMA
from wdf.config.defaults import defaults

_R = defaults("ridge")


@dataclass
class RidgeSettings:
    """Every choice of the ridge, read from the run parameters.

    :param slot_s: slot duration, s (`RidgeSlot`).
    :param octave_step: state width, octaves (`RidgeOctaveStep`).
    :param f_low: lowest state edge, Hz (`RidgeLowFrequency`, default the
        search's `MinFrequency`, else `LowFrequencyCut`).
    :param f_high: highest state edge, Hz (`RidgeHighFrequency`, default the
        Nyquist frequency of the searched stream).
    :param basis: basis of the dense window coefficients (`RidgeBasis`):
        ``LocalCos<M>``, or a p4TSA basis name.
    :param window: analysis window of the map, samples (`RidgeWindow`).
    :param overlap: overlap of consecutive map windows, samples
        (`RidgeOverlap`); the central ``window - overlap`` samples of each
        are used.
    :param alphas: sweeps of the dechirped local cosine, Hz/s
        (`RidgeDechirpAlphas`); empty for none. With sweeps, each cell takes
        its largest value over them.
    :param background_s: rolling background, s (`RidgeBackground`).
    :param background_update_s: its update period, s (`RidgeBackgroundUpdate`).
    :param background_min_s: background needed before a slot is scored, s
        (`RidgeBackgroundMinimum`).
    :param background_clip: the background's clipping level, standard
        deviations (`RidgeBackgroundClip`).
    :param background_iterations: clipping iterations
        (`RidgeBackgroundIterations`).
    :param lam: penalty per octave of a move beyond the allowance
        (`RidgeLambda`).
    :param allowance: free move per slot, octaves (`RidgeAllowance`).
    :param drift: per-cell drift b (`RidgeDrift`); None calibrates it on the
        background (`calibrate_drift`).
    :param xdrop: the drop below an episode's maximum that ends it
        (`RidgeXDrop`); None calibrates it with b.
    :param threshold: smallest candidate score kept (`RidgeThreshold`).
    :param max_slots: longest path, slots (`RidgeMaxSlots`), which bounds the
        memory the streaming path holds.
    :param waveform_states: states added on each side of a path's cells when
        its waveform is reconstructed (`RidgeWaveformStates`).
    :param calibration_seeds: random starts of the drift calibration
        (`RidgeCalibrationSeeds`).
    :param calibration_steps: slots each start is followed for
        (`RidgeCalibrationSteps`).
    :param calibration_kept: the largest median extension into noise the
        calibrated drift allows, slots (`RidgeCalibrationKept`).
    :param calibration_step: step of the drifts tried, from 0
        (`RidgeCalibrationGridStep`).
    :param calibration_max: largest drift tried (`RidgeCalibrationGridMax`).
    :param xdrop_slots: span over which the drop is measured, slots
        (`RidgeXDropSlots`).
    :param xdrop_percentile: percentile of the noise drops that sets X
        (`RidgeXDropPercentile`).
    """

    slot_s: float
    octave_step: float
    f_low: float
    f_high: float
    basis: str
    window: int
    overlap: int
    alphas: tuple
    background_s: float
    background_update_s: float
    background_min_s: float
    background_clip: float
    background_iterations: int
    lam: float
    allowance: float
    drift: object
    xdrop: object
    threshold: float
    max_slots: int
    waveform_states: int
    calibration_seeds: int
    calibration_steps: int
    calibration_kept: float
    calibration_step: float
    calibration_max: float
    xdrop_slots: int
    xdrop_percentile: float

    #: The run parameter each field is read from.
    KEYS = dict(slot_s="RidgeSlot", octave_step="RidgeOctaveStep", f_low="RidgeLowFrequency",
                f_high="RidgeHighFrequency", basis="RidgeBasis", window="RidgeWindow",
                overlap="RidgeOverlap", alphas="RidgeDechirpAlphas", background_s="RidgeBackground",
                background_update_s="RidgeBackgroundUpdate",
                background_min_s="RidgeBackgroundMinimum", background_clip="RidgeBackgroundClip",
                background_iterations="RidgeBackgroundIterations", lam="RidgeLambda",
                allowance="RidgeAllowance", drift="RidgeDrift", xdrop="RidgeXDrop",
                threshold="RidgeThreshold", max_slots="RidgeMaxSlots",
                waveform_states="RidgeWaveformStates", calibration_seeds="RidgeCalibrationSeeds",
                calibration_steps="RidgeCalibrationSteps", calibration_kept="RidgeCalibrationKept",
                calibration_step="RidgeCalibrationGridStep",
                calibration_max="RidgeCalibrationGridMax", xdrop_slots="RidgeXDropSlots",
                xdrop_percentile="RidgeXDropPercentile")

    #: Defaults, from the ``[ridge]`` table of wdf/config/defaults.toml. The
    #: penalty `lam` is ``1 / (ridge_merger_octaves - RidgeAllowance)`` unless
    #: `RidgeLambda` is set there: a move of `ridge_merger_octaves` in one
    #: slot costs one cell sigma.
    DEFAULTS = dict(slot_s=_R["RidgeSlot"], octave_step=_R["RidgeOctaveStep"], basis=_R["RidgeBasis"],
                    alphas=_R["RidgeDechirpAlphas"],
                    background_s=_R["RidgeBackground"], background_update_s=_R["RidgeBackgroundUpdate"],
                    background_min_s=_R["RidgeBackgroundMinimum"],
                    background_clip=_R["RidgeBackgroundClip"],
                    background_iterations=_R["RidgeBackgroundIterations"],
                    lam=(1.0 / (_R["ridge_merger_octaves"] - _R["RidgeAllowance"])
                         if _R["RidgeLambda"] is None else _R["RidgeLambda"]),
                    allowance=_R["RidgeAllowance"], drift=_R["RidgeDrift"], xdrop=_R["RidgeXDrop"],
                    threshold=_R["RidgeThreshold"], waveform_states=_R["RidgeWaveformStates"],
                    calibration_seeds=_R["RidgeCalibrationSeeds"],
                    calibration_steps=_R["RidgeCalibrationSteps"],
                    calibration_kept=_R["RidgeCalibrationKept"],
                    calibration_step=_R["RidgeCalibrationGridStep"],
                    calibration_max=_R["RidgeCalibrationGridMax"],
                    xdrop_slots=_R["RidgeXDropSlots"], xdrop_percentile=_R["RidgeXDropPercentile"])

    @classmethod
    def from_parameters(cls, par, rate):
        """The settings a run asks for, defaults filled in and written back.

        Unset in ``[ridge]``, `RidgeWindow` and `RidgeOverlap` default to the
        search's `window` and `overlap` (the first of a schedule),
        `RidgeMaxSlots` to `ridge_max_path_s` of slots, `RidgeHighFrequency`
        to `ridge_high_fraction_of_nyquist` of the Nyquist frequency.

        :param par: run configuration; it receives every value used.
        :type rate: float
        :param rate: sampling rate of the searched stream, Hz.
        :return: RidgeSettings
        """
        from wdf.config.Parameters import low_frequency_cut

        def get(field, default):
            value = getattr(par, cls.KEYS[field], None)
            return default if value is None else value

        def unset(key, derived):
            return derived if _R[key] is None else _R[key]

        window = par.window[0] if isinstance(par.window, (list, tuple)) else par.window
        overlap = par.overlap[0] if isinstance(par.overlap, (list, tuple)) else par.overlap
        d = cls.DEFAULTS
        slot = float(get("slot_s", d["slot_s"]))
        drift, xdrop = get("drift", d["drift"]), get("xdrop", d["xdrop"])
        out = cls(
            slot_s=slot,
            octave_step=float(get("octave_step", d["octave_step"])),
            f_low=float(get("f_low", unset("RidgeLowFrequency",
                                           getattr(par, "MinFrequency", None) or low_frequency_cut(par)))),
            f_high=float(get("f_high", unset("RidgeHighFrequency",
                                             _R["ridge_high_fraction_of_nyquist"] * 0.5 * float(rate)))),
            basis=str(get("basis", d["basis"])),
            window=int(get("window", unset("RidgeWindow", window))),
            overlap=int(get("overlap", unset("RidgeOverlap", overlap))),
            alphas=tuple(float(a) for a in get("alphas", d["alphas"])),
            background_s=float(get("background_s", d["background_s"])),
            background_update_s=float(get("background_update_s", d["background_update_s"])),
            background_min_s=float(get("background_min_s", d["background_min_s"])),
            background_clip=float(get("background_clip", d["background_clip"])),
            background_iterations=int(get("background_iterations", d["background_iterations"])),
            lam=float(get("lam", d["lam"])),
            allowance=float(get("allowance", d["allowance"])),
            drift=None if drift in (None, "calibrate") else float(drift),
            xdrop=None if xdrop in (None, "calibrate") else float(xdrop),
            threshold=float(get("threshold", d["threshold"])),
            max_slots=int(get("max_slots", unset("RidgeMaxSlots",
                                                 int(round(_R["ridge_max_path_s"] / slot))))),
            waveform_states=int(get("waveform_states", d["waveform_states"])),
            calibration_seeds=int(get("calibration_seeds", d["calibration_seeds"])),
            calibration_steps=int(get("calibration_steps", d["calibration_steps"])),
            calibration_kept=float(get("calibration_kept", d["calibration_kept"])),
            calibration_step=float(get("calibration_step", d["calibration_step"])),
            calibration_max=float(get("calibration_max", d["calibration_max"])),
            xdrop_slots=int(get("xdrop_slots", d["xdrop_slots"])),
            xdrop_percentile=float(get("xdrop_percentile", d["xdrop_percentile"])))
        for field, key in cls.KEYS.items():
            value = getattr(out, field)
            if field in ("drift", "xdrop") and value is None:
                value = "calibrate"
            setattr(par, key, list(value) if isinstance(value, tuple) else value)
        return out

    def edges(self):
        """The state edges, Hz: `f_low` times powers of 2 by `octave_step`,
        up to `f_high`."""
        octaves = np.log2(self.f_high / self.f_low)
        count = int(np.floor(octaves / self.octave_step + 1e-9))
        if count < 1:
            raise ValueError(f"no state between {self.f_low} and {self.f_high} Hz")
        return self.f_low * 2.0 ** (np.arange(count + 1) * self.octave_step)

    @property
    def n_states(self):
        """Number of states."""
        return self.edges().size - 1

    @property
    def calibration_grid(self):
        """The drifts the calibration tries: 0 to `calibration_max` by
        `calibration_step`."""
        count = int(np.floor(self.calibration_max / self.calibration_step + 1e-9))
        return np.round(np.arange(count + 1) * self.calibration_step, 10)

    def penalty(self):
        """``[from, to]`` cost of a move: ``lam * max(0, |d log2 f| - c)``."""
        centre = np.log2(self.edges()[:-1]) + 0.5 * self.octave_step
        return self.lam * np.maximum(0.0, np.abs(centre[:, None] - centre[None, :]) - self.allowance)

    def slots(self, seconds):
        """A duration in slots, rounded."""
        return max(int(round(float(seconds) / self.slot_s)), 1)

    def record(self):
        """The settings as a dict."""
        return asdict(self)


# ================================================================ tiles and cells
def dense_coefficients(x, basis, fs):
    """The dense coefficients of windows in one basis.

    :type x: numpy.ndarray
    :param x: windows, ``(windows, N)``.
    :type basis: str
    :param basis: ``LocalCos<M>`` (NumPy, `wdf.analysis.local_cosine`), or a
        p4TSA basis name (``"DaubC8"``, ``"Sym8P6"``), transformed by
        `py4tsa.WaveletTransform` window by window.
    :type fs: float
    :param fs: sampling rate, Hz.
    :return: numpy.ndarray -- ``(windows, N)``, laid out as p4TSA lays it.
    """
    from wdf.analysis.wavelets import LOCAL_COSINE, _local_cosine_segment

    X = np.atleast_2d(np.asarray(x, dtype=float))
    if str(basis).startswith(LOCAL_COSINE):
        from wdf.analysis.local_cosine import local_cosine
        return local_cosine(X, _local_cosine_segment(str(basis)))
    from py4tsa.tsa import WaveletTransform
    from wdf.analysis.wavelets import mother_wavelet, packet_depth
    from wdf.structures.array2SeqView import array2SeqView

    n = X.shape[1]
    depth = packet_depth(basis)
    mother = getattr(WaveletTransform, mother_wavelet(basis))
    transform = WaveletTransform(n, mother, depth) if depth else WaveletTransform(n, mother)
    out = np.empty_like(X)
    for i, row in enumerate(X):
        view = array2SeqView(0.0, fs, n)
        view.Fill(0.0, row.copy())
        transform.Forward(view.SV)
        out[i] = [view.SV.GetY(0, j) for j in range(n)]
    return out


def tile_geometry(n, fs, basis, alpha=0.0):
    """Each coefficient's tile in a window: time from the window's start and
    frequency; for a dechirped basis, its sweep.

    :type n: int
    :param n: window length, samples.
    :type fs: float
    :param fs: sampling rate, Hz.
    :type basis: str
    :param basis: the basis name.
    :type alpha: float
    :param alpha: the dechirp sweep, Hz/s (LocalCos bases only).
    :return: dict -- ``t_lo``, ``t_hi``, ``f_lo``, ``f_hi``, ``sweep`` (Hz/s,
        the band moves by ``sweep (t - tc)``), ``tc`` (s from the window's
        start).
    """
    from wdf.analysis.wavelets import coeff_freq_bands, coeff_time_bounds

    t_lo, t_hi = coeff_time_bounds(n, fs, basis)
    f_lo, f_hi = coeff_freq_bands(n, fs, basis)
    return dict(t_lo=t_lo, t_hi=t_hi, f_lo=f_lo, f_hi=f_hi,
                sweep=np.full(n, float(alpha)), tc=np.full(n, (n - 1) / (2.0 * fs)))


def window_sigma(coefficients, f_hi, f_low):
    """Each window's noise scale: the median of ``|c|`` over the coefficients
    whose tile reaches above `f_low`, over `MAD_GAUSSIAN`.

    :return: numpy.ndarray -- one scale per window.
    """
    keep = np.asarray(f_hi) > f_low + 1e-9
    return np.median(np.abs(np.atleast_2d(coefficients)[:, keep]), axis=-1) / MAD_GAUSSIAN


def window_tiles(coefficients, starts, fs, settings, alpha=0.0):
    """The tiles of windows, with their energy over each window's scale, kept
    to each window's central stretch and above `settings.f_low`.

    :type coefficients: numpy.ndarray
    :param coefficients: dense coefficients, ``(windows, N)``.
    :type starts: numpy.ndarray
    :param starts: GPS time of each window's first sample.
    :type fs: float
    :param fs: sampling rate, Hz.
    :type settings: RidgeSettings
    :param settings: the ridge's settings (basis, overlap, f_low).
    :type alpha: float
    :param alpha: the dechirp sweep the coefficients were taken at, Hz/s.
    :return: dict -- flat arrays ``t_lo``, ``t_hi`` (GPS), ``f_lo``, ``f_hi``,
        ``sweep``, ``tc`` (GPS), ``E``, and ``sigma`` per window.
    """
    C = np.atleast_2d(np.asarray(coefficients, dtype=float))
    w, n = C.shape
    g = tile_geometry(n, fs, settings.basis, alpha)
    sigma = window_sigma(C, g["f_hi"], settings.f_low)
    half = 0.5 * settings.overlap / fs
    centre = 0.5 * (g["t_lo"] + g["t_hi"])
    use = (centre >= half) & (centre < n / fs - half) & (g["f_hi"] > settings.f_low + 1e-9)
    starts = np.asarray(starts, dtype=float)[:, None]
    E = (C[:, use] / np.maximum(sigma, 1e-300)[:, None]) ** 2
    shape = (w, int(use.sum()))
    return dict(t_lo=(starts + g["t_lo"][use]).ravel(), t_hi=(starts + g["t_hi"][use]).ravel(),
                f_lo=np.broadcast_to(g["f_lo"][use], shape).ravel(),
                f_hi=np.broadcast_to(g["f_hi"][use], shape).ravel(),
                sweep=np.broadcast_to(g["sweep"][use], shape).ravel(),
                tc=(starts + g["tc"][use]).ravel(), E=E.ravel(), sigma=sigma)


def cell_sums(tiles, t0, n_slots, settings):
    """``X[slot, state]``: the tiles' energy shared among the cells.

    Each tile gives its energy E to the cells it overlaps, in proportion to
    the overlap in time over its duration and the overlap in log2 frequency
    over its log2 width; a tile with a sweep takes, in each slot, its band
    moved by ``sweep (t - tc)`` over the slot's overlap.

    :type tiles: dict
    :param tiles: as `window_tiles` returns them.
    :type t0: float
    :param t0: GPS time of slot 0's start.
    :type n_slots: int
    :param n_slots: slots of the map.
    :type settings: RidgeSettings
    :param settings: slot and states.
    :return: numpy.ndarray -- ``(n_slots, n_states)``.
    """
    edges = np.log2(settings.edges())
    ns, step, slot = edges.size - 1, settings.octave_step, settings.slot_s
    X = np.zeros((int(n_slots), ns))
    if tiles["E"].size == 0:
        return X
    tl, th, E = tiles["t_lo"], tiles["t_hi"], tiles["E"]
    fl, fh, sw, tc = tiles["f_lo"], tiles["f_hi"], tiles["sweep"], tiles["tc"]
    k0 = np.floor((tl - t0) / slot + 1e-9).astype(int)
    nk = np.ceil((th - t0) / slot - 1e-9).astype(int) - k0
    dur = th - tl
    for dk in range(int(nk.max())):
        k = k0 + dk
        ta = np.maximum(tl, t0 + k * slot)
        tb = np.minimum(th, t0 + (k + 1) * slot)
        wt = np.clip(tb - ta, 0.0, None) / dur
        live = (wt > 0) & (k >= 0) & (k < n_slots)
        if not live.any():
            continue
        half = 0.5 * (fh - fl)
        fa, fb = 0.5 * (fl + fh) + sw * (ta - tc), 0.5 * (fl + fh) + sw * (tb - tc)
        lo, hi = np.minimum(fa, fb) - half, np.maximum(fa, fb) + half
        floor = _R["cell_frequency_floor"]
        llo, lhi = np.log2(np.maximum(lo, floor)), np.log2(np.maximum(hi, floor))
        width = np.maximum(lhi - llo, 1e-12)
        s0 = np.clip(np.floor((llo - edges[0]) / step).astype(int), 0, None)
        nst = np.ceil((lhi - edges[0]) / step).astype(int) - s0
        top = int(max(nst[live].max(), 0))
        for ds in range(top):
            s = s0 + ds
            ov = np.clip(np.minimum(lhi, edges[0] + (s + 1) * step)
                         - np.maximum(llo, edges[0] + s * step), 0.0, None) / width
            ok = live & (ov > 0) & (s >= 0) & (s < ns)
            np.add.at(X, (k[ok], s[ok]), E[ok] * wt[ok] * ov[ok])
    return X


def window_starts(t0, n_samples, fs, settings):
    """First-sample times of the map's windows over a stream: a stride of
    ``window - overlap`` samples from `t0`.

    :return: numpy.ndarray -- sample offsets of the windows that fit.
    """
    stride = settings.window - settings.overlap
    if stride <= 0:
        raise ValueError("RidgeOverlap must be smaller than RidgeWindow")
    return np.arange(0, int(n_samples) - settings.window + 1, stride)


def stream_cells(x, t0, fs, settings):
    """The cell map of a whitened stream: windows, dense coefficients, tiles,
    cells; with sweeps, each cell's largest value over them.

    Slot 0 starts at ``t0 + overlap / 2 / fs``, where the first window's
    central stretch begins.

    :type x: numpy.ndarray
    :param x: the whitened stream.
    :type t0: float
    :param t0: GPS time of its first sample.
    :type fs: float
    :param fs: its sampling rate, Hz.
    :type settings: RidgeSettings
    :param settings: the ridge's settings.
    :return: tuple -- (X ``(slots, states)``, GPS time of slot 0, window
        offsets, dense coefficients per sweep ``{alpha: (windows, N)}``,
        scales per sweep ``{alpha: (windows,)}``).
    """
    x = np.asarray(x, dtype=float)
    offsets = window_starts(t0, x.size, fs, settings)
    origin = t0 + 0.5 * settings.overlap / fs
    if offsets.size == 0:
        return np.zeros((0, settings.n_states)), origin, offsets, {}, {}
    end = t0 + (offsets[-1] + settings.window - 0.5 * settings.overlap) / fs
    n_slots = int(np.floor((end - origin) / settings.slot_s + 1e-9))
    windows = x[offsets[:, None] + np.arange(settings.window)[None, :]]
    starts = t0 + offsets / fs
    X, coeffs, sigmas = None, {}, {}
    for alpha in (settings.alphas or (0.0,)):
        C = _coefficients(windows, fs, settings, alpha)
        tiles = window_tiles(C, starts, fs, settings, alpha)
        cells = cell_sums(tiles, origin, n_slots, settings)
        X = cells if X is None else np.maximum(X, cells)
        coeffs[alpha], sigmas[alpha] = C, tiles["sigma"]
    return X, origin, offsets, coeffs, sigmas


def _coefficients(windows, fs, settings, alpha):
    """Dense coefficients at one sweep (the dechirped local cosine for
    alpha != 0)."""
    if float(alpha) == 0.0:
        return dense_coefficients(windows, settings.basis, fs)
    from wdf.analysis.dechirp import dechirped_local_cosine
    from wdf.analysis.wavelets import _local_cosine_segment

    return dechirped_local_cosine(windows, fs, alpha, _local_cosine_segment(settings.basis))


class StreamingCells:
    """`stream_cells` one block of whitened samples at a time.

    Windows of `window` samples are taken every ``window - overlap`` samples
    as soon as their last sample has arrived; their tiles are added to the
    cells, and a slot is emitted once no later window can reach it (the end of
    the last window's central stretch has passed it). The dense coefficients
    of the last `buffer_windows` windows are kept in a ring, for the energy
    along a path (`ridge_waveform`). The rows emitted equal `stream_cells`'s
    over the same stream.

    :type settings: RidgeSettings
    :param settings: the map's settings.
    :type t0: float
    :param t0: GPS time of the stream's first sample.
    :type fs: float
    :param fs: its sampling rate, Hz.
    :type buffer_windows: int or None
    :param buffer_windows: windows held in the ring; enough for `max_slots`
        when None.
    """

    def __init__(self, settings, t0, fs, buffer_windows=None):
        self.settings, self.t0, self.fs = settings, float(t0), float(fs)
        self.stride = settings.window - settings.overlap
        if self.stride <= 0:
            raise ValueError("RidgeOverlap must be smaller than RidgeWindow")
        self.origin = self.t0 + 0.5 * settings.overlap / self.fs
        self.samples = np.zeros(0)
        self.first = 0
        self.next_window = 0
        self.next_slot = 0
        self.alphas = settings.alphas or (0.0,)
        self.pending = {a: np.zeros((0, settings.n_states)) for a in self.alphas}
        if buffer_windows is None:
            buffer_windows = int(np.ceil(settings.max_slots * settings.slot_s * self.fs / self.stride)) + 2
        self.ring = deque(maxlen=int(buffer_windows))

    def push(self, samples):
        """Add whitened samples; return the slots completed by them.

        :type samples: numpy.ndarray
        :param samples: the next samples of the stream.
        :return: numpy.ndarray -- ``(slots, states)``, the completed rows in
            order (possibly none).
        """
        s = self.settings
        self.samples = np.concatenate([self.samples, np.asarray(samples, dtype=float)])
        n = s.window
        done_until = None
        while self.next_window - self.first + n <= self.samples.size:
            a = self.next_window - self.first
            window = self.samples[a:a + n][None, :]
            start = self.t0 + self.next_window / self.fs
            keep_lo = start + 0.5 * s.overlap / self.fs
            keep_hi = start + (n - 0.5 * s.overlap) / self.fs
            k_lo = int(np.floor((keep_lo - self.origin) / s.slot_s + 1e-9))
            k_hi = int(np.ceil((keep_hi - self.origin) / s.slot_s - 1e-9))
            coeffs, sigmas = {}, {}
            for alpha in self.alphas:
                C = _coefficients(window, self.fs, s, alpha)
                tiles = window_tiles(C, [start], self.fs, s, alpha)
                cells = cell_sums(tiles, self.origin + k_lo * s.slot_s, k_hi - k_lo, s)
                need = k_hi - self.next_slot
                p = self.pending[alpha]
                if p.shape[0] < need:
                    p = np.vstack([p, np.zeros((need - p.shape[0], s.n_states))])
                p[k_lo - self.next_slot:k_hi - self.next_slot] += cells
                self.pending[alpha] = p
                coeffs[alpha], sigmas[alpha] = C[0], float(tiles["sigma"][0])
            self.ring.append((start, coeffs, sigmas))
            done_until = int(np.floor((keep_hi - self.origin) / s.slot_s + 1e-9))
            self.next_window += self.stride
        drop = self.next_window - self.first
        if drop > 0:
            self.samples = self.samples[drop:]
            self.first += drop
        if done_until is None or done_until <= self.next_slot:
            return np.zeros((0, s.n_states))
        count = done_until - self.next_slot
        rows = np.maximum.reduce([self.pending[a][:count] for a in self.alphas])
        for a in self.alphas:
            self.pending[a] = self.pending[a][count:]
        self.next_slot = done_until
        return rows

    def held(self, alpha=0.0):
        """The ring's windows: (GPS starts, coefficients, scales) at a sweep.

        :return: tuple of numpy.ndarray
        """
        if not self.ring:
            return np.zeros(0), np.zeros((0, self.settings.window)), np.zeros(0)
        return (np.array([r[0] for r in self.ring]), np.array([r[1][alpha] for r in self.ring]),
                np.array([r[2][alpha] for r in self.ring]))


# ================================================================ background and network
def _robust(rows, clip, iterations):
    """Mean and standard deviation of each column of `rows`, sigma-clipped.

    The cells' energy is a sum of a few squared coefficients, skewed to the
    right; its median and MAD would put the noise's own tail several sigma
    out. The clipped mean and standard deviation follow the noise's own
    moments while a loud transient in the background, beyond `clip` standard
    deviations, is left out of them.

    :return: tuple -- (mean, standard deviation) per column.
    """
    rows = np.asarray(rows, dtype=float)
    keep = np.ones(rows.shape, bool)
    for _ in range(int(iterations) + 1):
        n = np.maximum(keep.sum(0), 1)
        mean = np.where(keep, rows, 0.0).sum(0) / n
        sd = np.sqrt(np.where(keep, (rows - mean) ** 2, 0.0).sum(0) / np.maximum(n - 1, 1))
        keep = np.abs(rows - mean) <= clip * np.maximum(sd, 1e-30)
    return mean, np.maximum(sd, 1e-30)


def background_z(X, settings, causal=False):
    """Each state row standardised by its sigma-clipped mean and standard
    deviation over a rolling background (`_robust`).

    The slots are taken in blocks of `background_update_s`; each block is
    standardised with the statistics of the `background_s` around it
    (offline, centred) or strictly before its first slot (`causal`). A causal
    slot without `background_min_s` of past is NaN.

    :type X: numpy.ndarray
    :param X: cells, ``(slots, states)``.
    :type settings: RidgeSettings
    :param settings: the background's settings.
    :type causal: bool
    :param causal: use only the past.
    :return: numpy.ndarray -- z, same shape.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    Z = np.full(X.shape, np.nan)
    block, span = settings.slots(settings.background_update_s), settings.slots(settings.background_s)
    minimum = settings.slots(settings.background_min_s)
    for a in range(0, n, block):
        b = min(a + block, n)
        if causal:
            lo, hi = max(0, a - span), a
            if hi - lo < minimum:
                continue
        else:
            centre = (a + b) // 2
            lo, hi = max(0, centre - span // 2), min(n, centre + span // 2)
            if hi - lo < minimum:
                lo, hi = max(0, min(lo, n - minimum)), min(n, max(hi, minimum))
        med, sd = _robust(X[lo:hi], settings.background_clip, settings.background_iterations)
        Z[a:b] = (X[a:b] - med) / sd
    return Z


class CausalBackground:
    """`background_z` with ``causal=True``, one row at a time.

    The statistics are recomputed every `background_update_s` from the rows
    strictly before the update, kept in a ring of `background_s`; a row
    before `background_min_s` of past has none (NaN).
    """

    def __init__(self, settings):
        self.settings = settings
        self.rows = deque(maxlen=settings.slots(settings.background_s))
        self.block = settings.slots(settings.background_update_s)
        self.minimum = settings.slots(settings.background_min_s)
        self.count = 0
        self.stats = None

    def update(self, row):
        """The z of the next row.

        :type row: numpy.ndarray
        :param row: cells of one slot.
        :return: numpy.ndarray -- its z, NaN before enough past.
        """
        row = np.asarray(row, dtype=float)
        if self.count % self.block == 0:
            self.stats = (_robust(np.array(self.rows), self.settings.background_clip,
                                  self.settings.background_iterations)
                          if len(self.rows) >= self.minimum else None)
        self.rows.append(row)
        self.count += 1
        if self.stats is None:
            return np.full(row.shape, np.nan)
        med, sd = self.stats
        return (row - med) / sd


def combine_min(*zs):
    """The network's cell value: the smallest z over the detectors.

    :param zs: z maps of the same shape, one per detector.
    :return: numpy.ndarray -- their element-wise minimum (NaN where any is).
    """
    return np.minimum.reduce([np.asarray(z, dtype=float) for z in zs])


# ================================================================ drift calibration
def _greedy_gains(Z, starts, steps, pen, b, direction=1):
    """Greedy extension gains from (start, argmax state), vectorised over
    starts."""
    s = np.argmax(Z[starts], axis=1)
    G = np.zeros((len(starts), steps))
    for k in range(steps):
        v = Z[starts + direction * (k + 1)] - b - pen[s]
        s = np.argmax(v, axis=1)
        G[:, k] = v[np.arange(len(starts)), s]
    return G


def _drops_and_kept(G, span):
    """The largest drop of the cumulative gain within `span` slots, and the
    slots kept up to the maximum before a drop of X, per start."""
    Cm = np.hstack([np.zeros((len(G), 1)), np.cumsum(G, 1)])
    drops = np.zeros(len(G))
    for d in range(1, span + 1):
        drops = np.maximum(drops, (Cm[:, :-d] - Cm[:, d:]).max(1))
    return Cm, drops


def calibrate_drift(Z, settings, seed=1):
    """The drift b and the X-drop from noise, as ridge v2 fixes them.

    b is the smallest of `calibration_grid` (0 to `calibration_max` by
    `calibration_step`) for which a greedy extension from
    random starts into the background keeps, in median, at most
    `calibration_kept` slots before a drop of X; X is the
    `xdrop_percentile` of the largest drop of the cumulative gain within
    `xdrop_slots` slots, at that b.

    :type Z: numpy.ndarray
    :param Z: z rows of noise, ``(slots, states)``; NaN rows are left out.
    :type settings: RidgeSettings
    :param settings: the calibration's settings.
    :type seed: int
    :param seed: of the random starts.
    :return: tuple -- (b, X).
    :raises ValueError: if the noise is shorter than one calibration span.
    """
    Z = np.asarray(Z, dtype=float)
    Z = Z[np.all(np.isfinite(Z), axis=1)]
    steps = settings.calibration_steps
    if len(Z) < steps + 2:
        raise ValueError(f"{len(Z)} slots of background cannot calibrate the drift over {steps}")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(Z) - steps - 1, settings.calibration_seeds)
    pen = settings.penalty()
    result = None
    for b in settings.calibration_grid:
        G = _greedy_gains(Z, starts, steps, pen, b)
        Cm, drops = _drops_and_kept(G, settings.xdrop_slots)
        X = float(np.percentile(drops, settings.xdrop_percentile))
        best = np.maximum.accumulate(Cm, axis=1)
        stopped = Cm < best - X
        stop = np.where(stopped.any(1), stopped.argmax(1), Cm.shape[1])
        masked = np.where(np.arange(Cm.shape[1])[None, :] < stop[:, None], Cm, -np.inf)
        kept = np.argmax(masked, axis=1)
        result = (float(b), X)
        if np.median(kept) <= settings.calibration_kept:
            break
    return result


# ================================================================ the path
@dataclass
class RidgeCandidate:
    """A path through the cells.

    :param slots: slot indices, consecutive.
    :param states: one state per slot.
    :param score: the local alignment's score at the path's end.
    :param z_sum: sum of z over the path's cells.
    :param energy: sum of the cells' energy over the path, when the cells are
        known (noise units).
    :param t_start: GPS start of the first slot.
    :param t_end: GPS end of the last slot.
    :param emitted_at: GPS time of the slot when the candidate was emitted.
    :param latency: ``emitted_at - t_end``, s.
    :param f_low: lower edge of each slot's state, Hz.
    :param f_high: upper edge, Hz.
    :param extended: whether the path includes a greedy extension.
    """

    slots: np.ndarray
    states: np.ndarray
    score: float
    z_sum: float
    energy: float
    t_start: float
    t_end: float
    emitted_at: float
    latency: float
    f_low: np.ndarray
    f_high: np.ndarray
    extended: bool = False

    def row(self):
        """The candidate as one flat record."""
        return dict(t_start=self.t_start, t_end=self.t_end, duration=self.t_end - self.t_start,
                    score=self.score, z_sum=self.z_sum, energy=self.energy,
                    emitted_at=self.emitted_at, latency=self.latency,
                    f_start=float(np.sqrt(self.f_low[0] * self.f_high[0])),
                    f_end=float(np.sqrt(self.f_low[-1] * self.f_high[-1])),
                    n_slots=int(self.slots.size), extended=self.extended,
                    slots=self.slots.tolist(), states=self.states.tolist())


class StreamingRidge:
    """The local alignment, one slot at a time, from the past only.

    At slot k, ``H[k, s] = z[k, s] - b + max(0, max_s' H[k-1, s'] - pen[s', s])``
    with its back-pointer; a NaN row resets the scores. The back-pointers and
    rows of the last `max_slots` plus `xdrop_slots` slots are held in rings,
    so the memory is bounded. An episode begins when the best score rises from
    zero and ends when it drops `xdrop` below the episode's maximum or back to
    zero; its candidate, the path to the maximum, is emitted then, with the
    backward extension into the held past when `extend` is set.

    :type settings: RidgeSettings
    :param settings: penalty, drift, X-drop, threshold, max_slots.
    :type t0: float
    :param t0: GPS time of slot 0's start.
    :type extend: bool
    :param extend: extend each candidate backward, greedily, with the same
        X-drop.
    """

    def __init__(self, settings, t0, extend=True):
        if settings.drift is None or settings.xdrop is None:
            raise ValueError("the streaming ridge needs a drift and an X-drop; calibrate them "
                             "first (calibrate_drift)")
        self.settings, self.t0, self.extend = settings, float(t0), bool(extend)
        self.edges = settings.edges()
        self.pen = settings.penalty()
        ns = self.edges.size - 1
        self.depth = int(settings.max_slots) + int(settings.xdrop_slots) + 2
        self.back = np.full((self.depth, ns), -1, dtype=np.int32)
        self.rows = np.full((self.depth, ns), np.nan)
        self.cells = np.full((self.depth, ns), np.nan)
        self.H = np.zeros(ns)
        self.k = -1
        self.length = np.zeros(ns, dtype=int)
        self._episode = None

    def _slot(self, k):
        return k % self.depth

    def update(self, z, cells=None):
        """Take the next slot's z row (and its cells' energy).

        :type z: numpy.ndarray
        :param z: z of the slot's states; NaN anywhere resets the scores.
        :type cells: numpy.ndarray or None
        :param cells: the slot's cell energies, for the candidates' energy.
        :return: list[RidgeCandidate] -- the candidates whose episode ended
            at this slot.
        """
        s = self.settings
        self.k += 1
        i = self._slot(self.k)
        z = np.asarray(z, dtype=float)
        self.rows[i] = z
        self.cells[i] = np.nan if cells is None else cells
        out = []
        if not np.all(np.isfinite(z)):
            self.back[i] = -1
            self.H[:] = 0.0
            self.length[:] = 0
            out += self._close()
            return out
        cand = self.H[:, None] - self.pen
        arg = np.argmax(cand, axis=0)
        prev = cand[arg, np.arange(arg.size)]
        cont = (prev > 0) & (self.length[arg] < s.max_slots)
        self.H = z - s.drift + np.where(cont, prev, 0.0)
        self.back[i] = np.where(cont, arg, -1)
        self.length = np.where(cont, self.length[arg] + 1, 1)
        best = int(np.argmax(self.H))
        top = float(self.H[best])
        ep = self._episode
        if ep is None:
            if top > 0:
                self._episode = dict(max=top, k=self.k, s=best)
        elif top > ep["max"]:
            ep.update(max=top, k=self.k, s=best)
        if self._episode is not None and (top <= 0 or top < self._episode["max"] - s.xdrop
                                          or self.k - self._episode["k"] >= s.xdrop_slots + s.max_slots):
            out += self._close()
            if top > 0:
                self._episode = dict(max=top, k=self.k, s=best)
        return out

    def running(self):
        """The best path ending at the current slot.

        :return: RidgeCandidate or None -- None while no score is positive.
        """
        best = int(np.argmax(self.H))
        if self.H[best] <= 0:
            return None
        return self._candidate(self.k, best, float(self.H[best]))

    def flush(self):
        """End the open episode (end of the stream).

        :return: list[RidgeCandidate]
        """
        return self._close()

    def _close(self):
        ep, self._episode = self._episode, None
        if ep is None or ep["max"] < self.settings.threshold:
            return []
        return [self._candidate(ep["k"], ep["s"], ep["max"])]

    def _path(self, k, s):
        slots, states = [k], [s]
        while self.k - k < self.depth - 1:
            p = int(self.back[self._slot(k), s])
            if p < 0:
                break
            k, s = k - 1, p
            slots.append(k)
            states.append(s)
        return np.array(slots[::-1]), np.array(states[::-1])

    def _extend_back(self, k, s):
        """Greedy backward extension into the held past, X-drop."""
        st = self.settings
        rec, cum, best = [], 0.0, 0.0
        for _ in range(self.depth - 1):
            k -= 1
            if self.k - k >= self.depth or k < 0:
                break
            row = self.rows[self._slot(k)]
            if not np.all(np.isfinite(row)):
                break
            v = row - st.drift - self.pen[s]
            s = int(np.argmax(v))
            cum += float(v[s])
            rec.append((k, s, cum))
            best = max(best, cum)
            if cum < best - st.xdrop:
                break
        kb = int(np.argmax([0.0] + [r[2] for r in rec]))
        return rec[:kb]

    def _candidate(self, k, s, score):
        slots, states = self._path(k, s)
        extended = False
        if self.extend:
            ext = self._extend_back(int(slots[0]), int(states[0]))
            if ext:
                extended = True
                slots = np.concatenate([[r[0] for r in ext[::-1]], slots]).astype(int)
                states = np.concatenate([[r[1] for r in ext[::-1]], states]).astype(int)
        idx = np.array([self._slot(j) for j in slots])
        z_sum = float(np.sum(self.rows[idx, states]))
        cells = self.cells[idx, states]
        slot = self.settings.slot_s
        t_end = self.t0 + (slots[-1] + 1) * slot
        emitted = self.t0 + (self.k + 1) * slot
        return RidgeCandidate(slots=slots, states=states, score=float(score), z_sum=z_sum,
                              energy=float(np.sum(cells)) if np.all(np.isfinite(cells)) else float("nan"),
                              t_start=self.t0 + slots[0] * slot, t_end=t_end,
                              emitted_at=emitted, latency=emitted - t_end,
                              f_low=self.edges[states], f_high=self.edges[states + 1],
                              extended=extended)


def _extend_forward(Z, k, s, settings, pen):
    """Greedy forward extension over a whole map, X-drop."""
    rec, cum, best = [], 0.0, 0.0
    for _ in range(settings.max_slots):
        k += 1
        if k >= len(Z) or not np.all(np.isfinite(Z[k])):
            break
        v = Z[k] - settings.drift - pen[s]
        s = int(np.argmax(v))
        cum += float(v[s])
        rec.append((k, s, cum))
        best = max(best, cum)
        if cum < best - settings.xdrop:
            break
    kb = int(np.argmax([0.0] + [r[2] for r in rec]))
    return rec[:kb]


def offline_ridge(Z, t0, settings, X=None):
    """The ridge over a whole segment: the episodes of the local alignment,
    each extended greedily backward and forward with the same X-drop.

    :type Z: numpy.ndarray
    :param Z: z map, ``(slots, states)`` (one detector's, or `combine_min`
        of several).
    :type t0: float
    :param t0: GPS time of slot 0's start.
    :type settings: RidgeSettings
    :param settings: the ridge's settings; a drift or X-drop left to
        calibrate is calibrated on `Z` itself.
    :type X: numpy.ndarray or None
    :param X: the cells' energy, for the candidates' `energy`.
    :return: tuple -- (list of RidgeCandidate, by decreasing score; the
        settings used).
    """
    from dataclasses import replace

    Z = np.asarray(Z, dtype=float)
    if settings.drift is None or settings.xdrop is None:
        b, x = calibrate_drift(Z, settings)
        settings = replace(settings, drift=settings.drift if settings.drift is not None else b,
                           xdrop=settings.xdrop if settings.xdrop is not None else x)
    full = replace(settings, max_slots=max(settings.max_slots, len(Z)))
    ridge = StreamingRidge(full, t0, extend=True)
    found = []
    for k in range(len(Z)):
        found += ridge.update(Z[k], None if X is None else X[k])
    found += ridge.flush()
    pen = settings.penalty()
    edges = settings.edges()
    for c in found:
        ext = _extend_forward(Z, int(c.slots[-1]), int(c.states[-1]), settings, pen)
        if ext:
            c.slots = np.concatenate([c.slots, [r[0] for r in ext]]).astype(int)
            c.states = np.concatenate([c.states, [r[1] for r in ext]]).astype(int)
            c.extended = True
            c.z_sum = float(np.sum(Z[c.slots, c.states]))
            if X is not None:
                c.energy = float(np.sum(X[c.slots, c.states]))
            c.t_end = t0 + (c.slots[-1] + 1) * settings.slot_s
            c.f_low, c.f_high = edges[c.states], edges[c.states + 1]
        c.emitted_at, c.latency = float("nan"), float("nan")
    found.sort(key=lambda c: -c.score)
    return found, settings


def box_scores(Z, durations_s, settings, starts_every=_R["box_starts_every"]):
    """The best local-alignment score in every box of each duration, as ridge
    v2 standardises them for a false-alarm estimate.

    :type Z: numpy.ndarray
    :param Z: z map.
    :type durations_s: iterable of float
    :param durations_s: box durations, s.
    :type settings: RidgeSettings
    :param settings: penalty and drift (calibrated).
    :type starts_every: int
    :param starts_every: boxes start every duration / `starts_every`.
    :return: dict -- duration to (box starts, scores).
    """
    Z = np.nan_to_num(np.asarray(Z, dtype=float), nan=-np.inf)
    pen = settings.penalty()
    out = {}
    for D in durations_s:
        m = settings.slots(D)
        s = np.arange(0, len(Z) - m + 1, max(m // starts_every, 1))
        if s.size == 0:
            out[D] = (s, np.zeros(0))
            continue
        H = Z[s] - settings.drift
        best = H.max(1)
        for k in range(1, m):
            prev = (H[:, :, None] - pen[None]).max(1)
            H = Z[s + k] - settings.drift + np.maximum(prev, 0.0)
            best = np.maximum(best, H.max(1))
        out[D] = (s, best)
    return out


def standardise(boxes):
    """Per duration, the median and `MAD_TO_SIGMA` MAD of the box scores.

    :return: dict -- duration to (median, robust sigma).
    """
    out = {}
    for D, (_, v) in boxes.items():
        v = v[np.isfinite(v)]
        m = float(np.median(v)) if v.size else 0.0
        out[D] = (m, float(MAD_TO_SIGMA * np.median(np.abs(v - m))) if v.size else 1.0)
    return out


# ================================================================ energy along the ridge
def ridge_waveform(candidate, coefficients, sigmas, window_starts_gps, fs, t0, settings,
                   alpha=0.0):
    """The waveform under a path's cells and its integrated energy.

    In each window, the coefficients whose tile overlaps a cell of the path
    (its state widened by `waveform_states` on each side) are kept, in units
    of the window's scale, and the rest zeroed; the window is inverted
    (`inverse_local_cosine`, the re-chirp for a sweep, p4TSA otherwise) and
    its central stretch laid on the output. The energy is the sum of the
    squared waveform: the path's energy in noise units.

    :type candidate: RidgeCandidate
    :param candidate: the path.
    :type coefficients: numpy.ndarray
    :param coefficients: dense coefficients of the map's windows.
    :type sigmas: numpy.ndarray
    :param sigmas: each window's scale.
    :type window_starts_gps: numpy.ndarray
    :param window_starts_gps: each window's first sample, GPS.
    :type fs: float
    :param fs: sampling rate, Hz.
    :type t0: float
    :param t0: GPS time of slot 0's start.
    :type settings: RidgeSettings
    :param settings: the map's settings.
    :type alpha: float
    :param alpha: the sweep the coefficients were taken at.
    :return: tuple -- (GPS time of the first sample, waveform, energy).
    """
    from wdf.analysis.wavelets import LOCAL_COSINE, _local_cosine_segment

    C = np.atleast_2d(np.asarray(coefficients, dtype=float))
    n = C.shape[1]
    g = tile_geometry(n, fs, settings.basis, alpha)
    edges = settings.edges()
    ns = edges.size - 1
    slot = settings.slot_s
    margin = settings.waveform_states
    first_gps, last_gps = candidate.t_start, candidate.t_end
    out_start = first_gps
    size = int(np.ceil((last_gps - first_gps) * fs))
    y = np.zeros(size)
    lo_state = np.clip(candidate.states - margin, 0, ns - 1)
    hi_state = np.clip(candidate.states + margin, 0, ns - 1)
    half = 0.5 * settings.overlap / fs
    for j, ws in enumerate(np.asarray(window_starts_gps, dtype=float)):
        if ws + n / fs <= first_gps or ws >= last_gps:
            continue
        keep = np.zeros(n, bool)
        tl, th = ws + g["t_lo"], ws + g["t_hi"]
        for slot_i, a, b in zip(candidate.slots, lo_state, hi_state):
            sa, sb = t0 + slot_i * slot, t0 + (slot_i + 1) * slot
            keep |= ((tl < sb) & (th > sa) & (g["f_hi"] > edges[a]) & (g["f_lo"] < edges[b + 1]))
        if not keep.any():
            continue
        c = np.where(keep, C[j] / max(float(sigmas[j]), 1e-300), 0.0)
        if str(settings.basis).startswith(LOCAL_COSINE):
            from wdf.analysis.dechirp import inverse_dechirped_local_cosine
            w = inverse_dechirped_local_cosine(c, fs, alpha, _local_cosine_segment(settings.basis))
        else:
            from wdf.analysis.reconstruction import inverse_transform
            w = inverse_transform(c, settings.basis)
        t = ws + np.arange(n) / fs
        central = (t >= ws + half) & (t < ws + n / fs - half) & (t >= first_gps) & (t < last_gps)
        idx = np.round((t[central] - out_start) * fs).astype(int)
        ok = (idx >= 0) & (idx < size)
        y[idx[ok]] += w[central][ok]
    return out_start, y, float(np.sum(y ** 2))
