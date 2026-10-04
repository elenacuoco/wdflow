"""The online search: from the frames to the triggers and the ridge, causal or
at a fixed latency.

In its default `WhiteningMode`, "causal", `wdfUnitOnline` never uses
information from the future. Every output carries
the stream time at which it was emitted (`emitted_at`, the end of the data read
so far) and its latency, the time between the end of the data it describes and
that emission; altering the input after a time t changes no output emitted at
or before t.

The chain, block by block of `len` seconds:

1. **Band-pass and decimation, causal:** `BandPassDownSampling` with
   `BandPassPhase` "causal", its state carried from block to block.
2. **Whitening, causal:** the prediction-error filter of an autoregressive
   model of order `ARorder`, ``A(z) = 1 - sum a_k z^-k``, run forward with its
   state carried (`CausalPredictionError`; the lattice of `CausalWhitening` in
   direct form). The model is fitted (Burg, `Whitening.ParametersEstimate`) on
   `learn` seconds of the conditioned stream that end `ARFitGap` seconds before
   the time it starts whitening, after `ARFitContext` seconds of band-pass
   settling; it is fitted again every `ARRefitInterval` seconds from the past
   only, and a new filter takes over after a warm-up on the last `ARWarmup`
   samples (default `ARorder`), which sets its state exactly. The stretch
   is chosen from the past by a quality rule (`ARFitQuality`, on by
   default): of `ARFitCandidates` stretches `ARFitStep` seconds apart, the
   most recent whose whitened samples hold no census transient reaching
   `GlitchTrigger` and, for a refit, whose power in each octave from
   `GlitchCensusLow` through the current model stays within
   `ARFitLevelRatio` of white; a refit with none keeps the current model,
   the first fit takes the candidate of smallest census peak. Every model
   applied is written, with its switch time and fit stretch, to
   ``ARModels-<itf><suffix>.h5`` (`save_ar_models`; path `ARModelsFile`).
   With `ARModelsFrom` set to such a file the worker fits nothing: it applies
   that run's models at that run's switch times (the first model when its
   switch time is reached, each later one at its own), so two runs over the
   same stretch (a background and an injection run) whiten with one model
   history. Each model was fitted on data before its switch time, so the run
   stays causal, at the same latency.
3. **The search:** `wdf` at each window of the schedule, with `Bases`,
   `MinFrequency`, `ThresholdRule`, `BlockLength`, `BlockLambda` and
   `BlockRemainder` from the parameters. **No glitch subtraction and no
   gating:** the online search finds glitches too. Each trigger is labelled
   and kept: with `GlitchLabel` "census" (the default) "glitch" when its
   window meets a transient of the causal census of the whitened stream
   (`wdf.processes.causal_census.CausalCensus`, written as
   ``GlitchCensusOnline``), with "bases" when the basis it won is one of
   `GlitchLabelBases`; "candidate" otherwise. A label removes nothing.
4. **The ridge (`Ridge`):** the running wavegram of the window coefficients
   (`StreamingCells`, a ring of the recent windows' dense coefficients), a
   per-row z from the past only (`CausalBackground`), and the causal local
   alignment (`StreamingRidge`), whose drift and X-drop are calibrated on the
   past background unless given. It emits candidates with their path, their
   integrated energy (`ridge_waveform`) and their latency, and records the
   running best score of every slot.

Outputs go to ``<outdir>/<run>/<itf>/<channel>_<gpsStart>/``, every file
named with `SearchName` (default "online"): the triggers (Parquet), their
labels and latencies, the ridge candidates, the running best, the model refits,
and the parameters used.

**`WhiteningMode` "zerophase": the online search with the offline whitening
and a fixed, declared latency.** `WhiteningMode` "causal" (the default) is the
chain above. "zerophase" replaces step 2 by the offline whitening,
`TruncatedWhitening` of `ZeroPhaseDuration` seconds (`ZeroPhaseFilter`
"truncated", `HoldOutsideBand` as offline), built on the same autoregressive
model fitted on the past only, and allows `BandPassPhase` "zero" (its default
in this mode: the offline band-pass, `BandPassDownSampling.Process`) or
"causal". The search is then no longer causal but reads ahead by a fixed
latency ``Latency_s = BandPassLookahead_s + WhiteningLookahead_s``:

- `WhiteningLookahead_s`, the filter's half-support ``K / resampling`` with
  ``K = round(ZeroPhaseDuration * resampling / 2)`` (8 s at 16 s);
- `BandPassLookahead_s`, zero for the causal band-pass, and for the zero-phase
  one the read blocks it holds a block for until its settling `padlen` has
  been read, ``ceil(padlen / (len * sampling)) * len``.

The whitened stream at every read is complete up to ``now - Latency_s``: the
stream time `emitted_at` of every output is its `data_time` plus `Latency_s`,
and both are written. Altering the input after a time t changes no output whose
`data_time` is at or before ``t - Latency_s`` (tested). A model fitted when
the conditioned stream reaches S (on `learn` s ending ``S - ARFitGap``, as in
the causal mode) whitens the samples from S on, and the search's sigma follows
it from S. Glitches are labelled and kept, and the causal ridge runs on the
whitened stream, as in the causal mode.
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

import math

from wdf.analysis.io import parameters_file_name, search_suffix
from wdf.config.Parameters import (Parameters, check_bases, low_frequency_cut, reconstruction_mode,
                                   window_schedule)
from wdf.observers.observer import Observer

#: Basis names whose triggers are labelled "glitch" when none are given: the
#: shortest mothers, which win broadband transients and hardly any window of a
#: chirp.
DEFAULT_GLITCH_LABEL_BASES = ("Haar", "DaubC4")
#: Default `SearchName` of the online search.
DEFAULT_SEARCH_NAME = "online"
#: Default seconds of band-pass settling read before the fit stretch.
DEFAULT_FIT_CONTEXT_S = 20.0
#: The whitenings the online worker runs (`WhiteningMode`): "causal", the
#: prediction-error filter, no read-ahead; "zerophase", the offline zero-phase
#: whitening, with a fixed read-ahead `Latency_s`.
WHITENING_MODES = ("causal", "zerophase")
#: Default stretches of the past a fit may be taken on (`ARFitCandidates`).
DEFAULT_FIT_CANDIDATES = 3
#: Default spacing of those stretches, as a fraction of `learn` (`ARFitStep`).
DEFAULT_FIT_STEP_FRACTION = 0.5
#: Default largest octave power, through the current model, over white
#: (and smallest, its inverse) of a refit's stretch (`ARFitLevelRatio`).
DEFAULT_FIT_LEVEL_RATIO = 4.0
#: Default `SearchName` in the "zerophase" mode.
DEFAULT_ZEROPHASE_SEARCH_NAME = "online_zerophase"
#: Zero-phase filters the "zerophase" mode runs: those of fixed support, so
#: that the latency is fixed.
ONLINE_ZERO_PHASE_FILTERS = ("truncated",)


# ================================================================ data sources
class ArraySource:
    """A stream held in memory, read block by block.

    :type samples: numpy.ndarray
    :param samples: the strain.
    :type t0: float
    :param t0: GPS time of the first sample.
    :type sampling: float
    :param sampling: sampling rate, Hz.
    """

    def __init__(self, samples, t0, sampling):
        self.samples = np.asarray(samples, dtype=float)
        self.t0, self.sampling = float(t0), float(sampling)

    def read(self, start, seconds):
        """The samples of ``[start, start + seconds)``.

        :raises EOFError: if the stream does not hold them all.
        """
        a = int(round((float(start) - self.t0) * self.sampling))
        n = int(round(float(seconds) * self.sampling))
        if a < 0 or a + n > self.samples.size:
            raise EOFError(f"the stream holds no {seconds} s from {start}")
        return self.samples[a:a + n]


class FrameSource:
    """A channel of frame files, read in sequence through p4TSA.

    :type file: str
    :param file: the frame file or frame-file list `FrameIChannel` reads.
    :type channel: str
    :param channel: the channel.
    """

    def __init__(self, file, channel):
        self.file, self.channel = file, channel
        self._reader, self._next, self._length = None, None, None

    def read(self, start, seconds):
        """The samples of ``[start, start + seconds)``.

        :raises EOFError: if the frames do not return them.
        """
        from py4tsa.tsa import FrameIChannel
        from py4tsa.tsa import SeqView_double_t as SV

        from wdf.processes.BandPassDownSampling import SV_to_array

        if self._reader is None or self._next is None or abs(self._next - start) > 1e-6:
            self._reader = FrameIChannel(self.file, self.channel, float(seconds), float(start))
        elif self._length != seconds:
            self._reader.SetDataLength(float(seconds))
        self._length = seconds
        view = SV()
        try:
            self._reader.GetData(view)
        except Exception as error:
            raise EOFError(f"no frame data at {start}: {error}") from error
        if abs(view.GetStart() - start) > 0.5 * view.GetSampling():
            raise EOFError(f"asked for {start}, the frames gave {view.GetStart()}")
        self._next = start + seconds
        return SV_to_array(view) * view.GetScale()


# ================================================================ causal whitening
def fit_autoregressive(samples, start, rate, order):
    """Burg's autoregressive model of a stretch (`Whitening.ParametersEstimate`).

    :return: numpy.ndarray -- the scale in ``[0]`` and the prediction
        coefficients in ``[1:]``, as `Whitening` stores them.
    """
    from wdf.processes.Whitening import Whitening
    from wdf.structures.array2SeqView import array2SeqView

    view = array2SeqView(start, rate, len(samples))
    view.Fill(start, np.asarray(samples, dtype=float))
    model = Whitening(int(order))
    model.ParametersEstimate(view.SV)
    return np.array([model.ADE.GetAR(j) for j in range(int(order) + 1)])


class CausalPredictionError:
    """The autoregressive prediction-error filter, run forward with its state.

    The output is ``A(z) x`` with ``A = 1 - sum a_k z^-k``, the lattice
    filter's output in direct form, with standard deviation the model's scale
    ``ar[0]`` on the noise it was fitted to. Its state is the last `order`
    input samples, set exactly by a warm-up on at least that many.

    :type ar: numpy.ndarray
    :param ar: the model, scale in ``[0]``, coefficients in ``[1:]``.
    :type warmup: numpy.ndarray
    :param warmup: the samples just before the first one to whiten.
    """

    def __init__(self, ar, warmup):
        from wdf.processes.zero_phase_whitening import prediction_error_polynomial

        self.ar = np.asarray(ar, dtype=float)
        self.sigma = float(self.ar[0])
        self.taps = prediction_error_polynomial(self.ar)
        self.state = np.zeros(self.taps.size - 1)
        if len(warmup):
            self.process(warmup)

    def process(self, samples):
        """Whiten the next samples.

        :return: numpy.ndarray -- ``A(z) x``.
        """
        out, self.state = lfilter(self.taps, [1.0], np.asarray(samples, dtype=float), zi=self.state)
        return out


class LookaheadWhitening:
    """The offline zero-phase whitening of one model, applied from a time on.

    The taps are those `TruncatedWhitening` builds (`truncated_taps`), ``2 K +
    1`` symmetric with lag zero at ``K``; a whitened sample is their
    convolution with the conditioned stream ``K`` samples on each side of it,
    as the offline worker computes it, so it needs ``K`` samples of future.

    :type ar: numpy.ndarray
    :param ar: the model, scale in ``[0]``, coefficients in ``[1:]``.
    :type duration: float
    :param duration: `ZeroPhaseDuration`, s.
    :type rate: float
    :param rate: sampling rate of the conditioned stream, Hz.
    :type start: float
    :param start: GPS time of the first sample it whitens.
    :type held: dict
    :param held: ``band`` and ``blend`` with `HoldOutsideBand`, else empty.
    """

    def __init__(self, ar, duration, rate, start, held):
        from wdf.processes.zero_phase_whitening import TruncatedWhitening

        filt = TruncatedWhitening(np.asarray(ar, dtype=float), 1, 0, duration=float(duration),
                                  sampling=float(rate), **held)
        self.taps, self.support, self.sigma = filt.taps, int(filt.support), float(filt.sigma)
        self.start = float(start)

    def apply(self, stream, a, b):
        """The whitened samples ``[a, b)`` of `stream` (indices), which must
        hold ``K`` samples on each side."""
        from scipy.signal import fftconvolve

        k = self.support
        if a - k < 0 or b + k > stream.size:
            raise ValueError(f"whitening samples {a}-{b} needs {k} on each side of a stream "
                             f"of {stream.size}")
        return fftconvolve(stream[a - k:b + k], self.taps, mode="valid")


# ================================================================ model history
#: Datasets of a model-history file (`save_ar_models`).
AR_MODEL_FIELDS = ("switch_gps", "fit_start", "fit_end", "sigma", "ar")


def save_ar_models(path, models, attributes):
    """Write the time-stamped sequence of noise models a run applied.

    HDF5, one row per model in switch order: ``switch_gps`` (GPS from which
    the model whitens: the read time in the causal mode, the end of the
    conditioned stream in the "zerophase" mode), ``fit_start`` and
    ``fit_end`` (its fit stretch, GPS), ``sigma`` (its whitening's scale) and
    ``ar`` (``(models, ARorder + 1)``, the scale in column 0 and the
    prediction coefficients after it, as `fit_autoregressive` returns them).
    The run's ``ARorder``, ``WhiteningMode``, ``resampling``,
    ``ZeroPhaseDuration`` and ``itf`` are file attributes.

    :param path: the file.
    :param models: list of dicts with those fields.
    :param attributes: dict of the file attributes.
    :return: None
    """
    import h5py

    with h5py.File(path, "w") as h5:
        for name in AR_MODEL_FIELDS[:-1]:
            h5.create_dataset(name, data=np.array([m[name] for m in models], dtype=float))
        order = int(attributes["ARorder"])
        h5.create_dataset("ar", data=np.array([m["ar"] for m in models], dtype=float).reshape(
            len(models), order + 1))
        for key, value in attributes.items():
            h5.attrs[key] = value


def load_ar_models(path):
    """Read what `save_ar_models` wrote.

    :return: tuple -- (list of dicts in switch order, dict of attributes).
    """
    import h5py

    with h5py.File(path, "r") as h5:
        columns = {name: np.asarray(h5[name]) for name in AR_MODEL_FIELDS}
        attributes = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h5.attrs.items()}
    models = [{name: (columns[name][i] if name == "ar" else float(columns[name][i]))
               for name in AR_MODEL_FIELDS} for i in range(columns["switch_gps"].size)]
    return sorted(models, key=lambda m: m["switch_gps"]), attributes


# ================================================================ outputs
@dataclass
class OnlineResult:
    """What an online run emitted, each with its emission time.

    :param triggers: one dict per trigger: its record without the
        coefficients, ``window_end``, ``emitted_at``, ``latency``, ``label``
        and ``search``.
    :param candidates: one dict per ridge candidate (`RidgeCandidate.row`),
        with ``waveform_energy``.
    :param running: per slot, ``slot_end``, ``score`` and ``state`` of the
        running best path and ``emitted_at``.
    :param refits: one dict per model fit: ``emitted_at``, ``fit_start``,
        ``fit_end``, ``sigma``.
    :param models: the noise models applied, in switch order (the fields of
        `save_ar_models`).
    :param census: with `GlitchLabel` "census", one dict per transient of the
        causal census (`CausalCensus`) once closed: ``first``, ``last``,
        ``peak``, ``peak_time``, ``extent_first``, ``extent_last``,
        ``emitted_at``.
    :param parameters: the parameters used.
    :param tracks: with `Reconstruction` "track", one dict per track
        (`TRACK_EVENT_COLUMNS`), with ``seed_end`` (the end of its last
        seed's window), ``emitted_at``, ``latency`` (``emitted_at -
        seed_end``) and ``search``.
    :param track_waveforms: ``{cluster_id: (GPS of the first sample,
        waveform)}`` of those tracks.

    In the "zerophase" mode every row also holds ``data_time``,
    ``emitted_at - Latency_s``: the end of the whitened stream at emission.
    """

    triggers: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    running: dict = field(default_factory=dict)
    refits: list = field(default_factory=list)
    parameters: object = None
    tracks: list = field(default_factory=list)
    track_waveforms: dict = field(default_factory=dict)
    census: list = field(default_factory=list)
    models: list = field(default_factory=list)

    def before(self, t):
        """Everything emitted at or before GPS `t`, as plain lists (NaN as
        None), for comparing two runs.

        :return: dict
        """
        def plain(x):
            return None if isinstance(x, float) and np.isnan(x) else x

        def keep(rows):
            return [{k: plain(v) for k, v in r.items()} for r in rows if r["emitted_at"] <= t]

        mask = np.asarray(self.running.get("emitted_at", []), dtype=float) <= t
        running = {k: [plain(x) for x in np.asarray(v)[mask].tolist()]
                   for k, v in self.running.items()}
        def plain_track(row):
            return {k: (plain(v) if not isinstance(v, (list, np.ndarray)) else
                        [plain(float(x)) for x in v]) for k, v in row.items()}

        tracks = [plain_track(r) for r in self.tracks if r["emitted_at"] <= t]
        return dict(triggers=keep(self.triggers), candidates=keep(self.candidates),
                    running=running, refits=keep(self.refits), tracks=tracks,
                    census=keep(self.census))


class _TriggerLog(Observer):
    """Collects the triggers with their emission time, latency and label."""

    def __init__(self, worker, par):
        Observer.__init__(self)
        self.worker, self.par = worker, par
        self.duration = par.window / float(par.resampling)
        self.label_bases = set(getattr(par, "GlitchLabelBases", DEFAULT_GLITCH_LABEL_BASES))

    def update(self, event):
        row = {k: v for k, v in event.record().items() if k not in ("wt_index", "wt_value")}
        now = self.worker.now
        end = row["gps"] + self.duration
        census = self.worker._census
        if census is None:
            glitch = row["wave"] in self.label_bases
        else:
            glitch = census.overlaps(row["gps"], end)
        row.update(window_end=end, emitted_at=now, latency=now - end,
                   label="glitch" if glitch else "candidate", search=self.par.SearchName)
        if self.worker.latency is not None:
            row["data_time"] = now - self.worker.latency
        self.worker.result.triggers.append(row)


# ================================================================ the worker
class wdfUnitOnline:
    """The streaming search of one detector: causal, or with `WhiteningMode`
    "zerophase" the offline whitening at the fixed latency `Latency_s`.

    :type parameters: wdf.config.Parameters.Parameters
    :param parameters: the run configuration (`configs/online_default.toml`
        through `load_config`); copied, so the caller's object is not changed.
    """

    def __init__(self, parameters):
        self.par = Parameters()
        self.par.copy(parameters)
        par = self.par
        par.resampling = par.sampling / par.ResamplingFactor
        mode = str(getattr(par, "WhiteningMode", None) or "causal").lower()
        if mode not in WHITENING_MODES:
            raise ValueError(f"WhiteningMode {mode!r}: use one of {WHITENING_MODES}")
        self.mode = mode
        phase = str(getattr(par, "BandPassPhase", None)
                    or ("causal" if mode == "causal" else "zero")).lower()
        if mode == "causal" and phase != "causal":
            raise ValueError(f"the online worker runs the causal band-pass; BandPassPhase is {phase!r}")
        par.BandPassPhase = phase
        par.LowFrequencyCut = low_frequency_cut(par)
        default_name = DEFAULT_SEARCH_NAME if mode == "causal" else DEFAULT_ZEROPHASE_SEARCH_NAME
        par.SearchName = getattr(par, "SearchName", None) or default_name
        search_suffix(par)
        self.schedule = window_schedule(par)
        check_bases(par, self.schedule)
        par.Ncoeff = max(w for w, _ in self.schedule)
        par.Ridge = bool(getattr(par, "Ridge", True))
        par.GlitchLabelBases = list(getattr(par, "GlitchLabelBases", None) or DEFAULT_GLITCH_LABEL_BASES)
        from wdf.processes.causal_census import GLITCH_LABELS, CensusSettings

        label = str(getattr(par, "GlitchLabel", None) or GLITCH_LABELS[0]).lower()
        if label not in GLITCH_LABELS:
            raise ValueError(f"GlitchLabel {label!r}: use one of {GLITCH_LABELS}")
        par.GlitchLabel = label
        self.census_settings = CensusSettings.from_parameters(par) if label == "census" else None
        self._census = None
        par.ARFitGap = float(getattr(par, "ARFitGap", None) or 0.0)
        par.ARFitContext = float(getattr(par, "ARFitContext", None) or DEFAULT_FIT_CONTEXT_S)
        par.ARRefitInterval = float(getattr(par, "ARRefitInterval", None) or 0.0)
        warm = getattr(par, "ARWarmup", None)
        par.ARWarmup = int(par.ARorder if warm is None else warm)
        par.ARFitQuality = bool(getattr(par, "ARFitQuality", True))
        par.ARFitCandidates = int(getattr(par, "ARFitCandidates", None) or DEFAULT_FIT_CANDIDATES)
        step = getattr(par, "ARFitStep", None)
        par.ARFitStep = float(DEFAULT_FIT_STEP_FRACTION * par.learn if step is None else step)
        par.ARFitLevelRatio = float(getattr(par, "ARFitLevelRatio", None) or DEFAULT_FIT_LEVEL_RATIO)
        if par.ARFitCandidates < 1 or par.ARFitStep < 0 or par.ARFitLevelRatio <= 1:
            raise ValueError("ARFitCandidates must be at least 1, ARFitStep not negative and "
                             "ARFitLevelRatio above 1")
        from wdf.processes.causal_census import CensusSettings as _Census

        self._fit_census = _Census.from_parameters(par)
        self._model = None
        if par.ARWarmup < par.ARorder:
            raise ValueError(f"ARWarmup {par.ARWarmup} samples is shorter than the model's order "
                             f"{par.ARorder}: the filter's state would not be set by real data")
        self.now = None
        self.result = None
        #: The fixed read-ahead, s; None in the causal mode, which has none.
        self.latency = None
        if mode == "zerophase":
            self._declare_latency()
        #: Stage 2 the worker feeds: "wavegram" (nothing beyond the triggers)
        #: or "track" (`TrackObserver`).
        self.reconstruction = reconstruction_mode(par)
        self.track_settings = None
        if self.reconstruction == "track":
            self._declare_track_latency()

    def _declare_track_latency(self):
        """With `Reconstruction` "track": the track settings and the declared
        latency of a track, `TrackLatency_s`.

        A track is built once the map is complete `TrackHorizon` seconds past
        its last seed's window: the map's slot and the part of its last window
        past the slot (`TrackSettings.completion_s`) are read too, on top of
        the stream's own latency (`Latency_s`, 0 in the causal mode).

        :raises ValueError: without `Ridge`, or for a slice longer than the
            map's ring holds.
        """
        from wdf.analysis.track_reconstruction import TrackSettings

        par = self.par
        if not par.Ridge:
            raise ValueError("Reconstruction 'track' reads the ridge's map: set Ridge = true")
        par.Reconstruction = "track"
        rate = float(par.resampling)
        ts = TrackSettings.from_parameters(par, rate)
        rs = ts.ridge
        window_s = max(w for w, _ in self.schedule) / rate
        need = ts.lookback + window_s + ts.horizon + ts.completion_s(rate)
        held = rs.max_slots * rs.slot_s
        if need > held:
            raise ValueError(f"TrackLookback + window + TrackHorizon, {need:g} s, is longer than "
                             f"the map's ring, RidgeMaxSlots x RidgeSlot = {held:g} s")
        self.track_settings = ts
        par.TrackCompletion_s = ts.completion_s(rate)
        par.TrackLatency_s = (self.latency or 0.0) + ts.horizon + par.TrackCompletion_s
        logging.info(f"online track: declared latency {par.TrackLatency_s:g} s = stream "
                     f"{self.latency or 0.0:g} s + TrackHorizon {ts.horizon:g} s + map "
                     f"completion {par.TrackCompletion_s:g} s")

    def _declare_latency(self):
        """Set the "zerophase" mode's parameters and its latency, and log it.

        :raises ValueError: for a zero-phase filter whose support is not fixed,
            or a fit stretch shorter than the filter's past.
        """
        from wdf.processes.BandPassDownSampling import BAND_PASS_PHASES, BandPassDownSampling
        from wdf.processes.zero_phase_whitening import DEFAULT_TRUNCATION_S

        par = self.par
        par.WhiteningMode = self.mode
        if par.BandPassPhase not in BAND_PASS_PHASES:
            raise ValueError(f"BandPassPhase {par.BandPassPhase!r}: use one of {BAND_PASS_PHASES}")
        name = str(getattr(par, "ZeroPhaseFilter", None) or "truncated").lower()
        if name not in ONLINE_ZERO_PHASE_FILTERS:
            raise ValueError(f"WhiteningMode 'zerophase' runs a filter of fixed support, so of "
                             f"fixed latency: ZeroPhaseFilter {name!r}, use one of "
                             f"{ONLINE_ZERO_PHASE_FILTERS}")
        par.ZeroPhaseFilter = name
        par.ZeroPhaseDuration = float(getattr(par, "ZeroPhaseDuration", None) or DEFAULT_TRUNCATION_S)
        par.HoldOutsideBand = bool(getattr(par, "HoldOutsideBand", False))
        rate = float(par.resampling)
        support = int(round(0.5 * par.ZeroPhaseDuration * rate))
        par.ZeroPhaseLatency = support
        par.WhiteningLookahead_s = support / rate
        if par.learn + par.ARFitGap < par.WhiteningLookahead_s:
            raise ValueError(f"learn + ARFitGap, {par.learn + par.ARFitGap} s, is shorter than the "
                             f"whitening's past, {par.WhiteningLookahead_s} s: the first whitened "
                             f"sample would reach before the band-pass settled")
        if par.BandPassPhase == "zero":
            ds = BandPassDownSampling(par)
            block = int(round(float(par.len) * ds.sampling))
            par.BandPassSettling = int(ds.padlen)
            par.BandPassLookahead_s = math.ceil(ds.padlen / block) * float(par.len)
        else:
            par.BandPassLookahead_s = 0.0
        par.Latency_s = par.BandPassLookahead_s + par.WhiteningLookahead_s
        self.latency = par.Latency_s
        logging.info(f"online zerophase: declared latency {par.Latency_s:g} s = band-pass "
                     f"{par.BandPassLookahead_s:g} s ({par.BandPassPhase} phase) + whitening "
                     f"{par.WhiteningLookahead_s:g} s (TruncatedWhitening "
                     f"{par.ZeroPhaseDuration:g} s, half-support {support} samples)")

    # ------------------------------------------------------------ helpers
    def _view(self, samples, start):
        from wdf.structures.array2SeqView import array2SeqView

        view = array2SeqView(start, self.par.resampling, len(samples))
        view.Fill(start, samples)
        return view.SV

    def _searches(self, sigma):
        from wdf.observers.ParameterEstimationObserver import ParameterEstimation
        from wdf.observers.SingleEventPrintFileObserver import SingleEventPrintTriggers
        from wdf.processes.wdf import wdf
        from wdf.processes.wdfUnitDSWorker import search_names

        searches, writers = [], []
        for window, overlap, name in search_names(self.schedule, self.par.SearchName):
            par = Parameters()
            par.copy(self.par)
            par.window, par.overlap, par.Ncoeff, par.SearchName = window, overlap, window, name
            par.sigma = sigma
            search = wdf(par)
            estimation = ParameterEstimation(par)
            estimation.register(_TriggerLog(self, par))
            if self._tracker is not None:
                estimation.register(self._tracker)
            if self.dir is not None:
                writer = SingleEventPrintTriggers(par)
                estimation.register(writer)
                writers.append(writer)
                par.dump(par.dir + parameters_file_name(par))
            search.register(estimation)
            searches.append(search)
        return searches, writers

    def _fit(self, past, past_start, now):
        """A model fitted on the past only and its filter, from `now` on; None
        when a refit finds no stretch of the past fit to fit on.

        In the causal mode `now` is the read time and the filter is warmed up
        to it; in the "zerophase" mode it is the end of the conditioned stream,
        and the filter whitens the samples from it on.

        The stretch is chosen by `_choose_stretch`: the most recent of
        `ARFitCandidates` stretches of `learn` seconds, `ARFitStep` apart,
        ending `ARFitGap` before `now`, that passes the quality rule.
        """
        par, rate = self.par, float(self.par.resampling)
        chosen = (self._reused_model(now) if self._reused is not None
                  else self._choose_stretch(past, past_start, now))
        if chosen is None:
            return None
        fit_start, fit_end, ar, filt, quality = chosen
        self.result.models.append(dict(switch_gps=float(now), fit_start=float(fit_start),
                                       fit_end=float(fit_end), ar=np.asarray(ar, dtype=float)))
        if self.mode == "causal":
            warm = past[past.size - par.ARWarmup:]
            self.result.refits.append(dict(emitted_at=self.now, fit_start=fit_start,
                                           fit_end=fit_end, sigma=float(ar[0]), **quality))
            logging.info(f"online: AR {par.ARorder} fitted on {fit_start:.1f}-{fit_end:.1f}, "
                         f"sigma {ar[0]:.4g}")
            filt = CausalPredictionError(ar, warm)
        else:
            self.result.refits.append(dict(emitted_at=self.now, fit_start=fit_start,
                                           fit_end=fit_end, sigma=filt.sigma,
                                           ar_sigma=float(ar[0]), whitens_from=now,
                                           data_time=self.now - self.latency, **quality))
            logging.info(f"online zerophase: AR {par.ARorder} fitted on {fit_start:.1f}-"
                         f"{fit_end:.1f}, TruncatedWhitening {par.ZeroPhaseDuration:g} s from "
                         f"{now:.1f}, sigma {filt.sigma:.4g}")
        self.result.models[-1]["sigma"] = float(filt.sigma)
        self._model = filt
        return filt

    def _reused_model(self, now):
        """With `ARModelsFrom`: the model the other run switched to at `now`,
        or None when it switched to none then. No stretch is fitted or
        judged."""
        rate = float(self.par.resampling)
        for m in self._reused:
            if abs(m["switch_gps"] - now) <= 0.5 / rate:
                ar = np.asarray(m["ar"], dtype=float)
                filt = self._filter_of(ar, now) if self.mode == "zerophase" else None
                return m["fit_start"], m["fit_end"], ar, filt, dict(accepted=True, reused=True)
        return None

    def _next_switch(self, after):
        """When to try the next model: `ARRefitInterval` after `after`, or,
        with `ARModelsFrom`, the other run's next switch time."""
        if self._reused is not None:
            rate = float(self.par.resampling)
            later = [m["switch_gps"] for m in self._reused if m["switch_gps"] > after + 0.5 / rate]
            return min(later) if later else None
        return after + self.par.ARRefitInterval if self.par.ARRefitInterval > 0 else None

    def _load_reused(self):
        """`ARModelsFrom`: the other run's model history, checked against this
        run's model order, whitening and rate."""
        par = self.par
        path = str(getattr(par, "ARModelsFrom", None) or "")
        par.ARModelsFrom = path
        if not path:
            return None
        models, attributes = load_ar_models(path)
        if not models:
            raise ValueError(f"ARModelsFrom {path} holds no model")
        mine = dict(ARorder=int(par.ARorder), WhiteningMode=self.mode,
                    resampling=float(par.resampling))
        if self.mode == "zerophase":
            mine["ZeroPhaseDuration"] = float(par.ZeroPhaseDuration)
        for key, value in mine.items():
            theirs = attributes.get(key)
            if theirs is None or (theirs != value if isinstance(value, str)
                                  else abs(float(theirs) - float(value)) > 1e-9):
                raise ValueError(f"ARModelsFrom {path} was written with {key} {theirs}; this "
                                 f"run has {value}")
        par.ARModelsReused = len(models)
        logging.info(f"online: {len(models)} noise models reused from {path}; no fit is made")
        return models

    def _filter_of(self, ar, now):
        """The whitening filter of a model in this mode (no warm-up)."""
        if self.mode == "causal":
            return CausalPredictionError(ar, np.zeros(0))
        return LookaheadWhitening(ar, self.par.ZeroPhaseDuration, float(self.par.resampling), now,
                                  self._held)

    @staticmethod
    def _whitened_stretch(filt, x):
        """A stretch of the conditioned stream through a filter, in units of
        its scale, where the filter reached real data on every side."""
        from scipy.signal import fftconvolve

        if isinstance(filt, CausalPredictionError):
            return lfilter(filt.taps, [1.0], x)[filt.taps.size - 1:] / filt.sigma
        return fftconvolve(x, filt.taps, mode="valid") / filt.sigma

    def _stretch_quality(self, w, start, current):
        """The quality rule on a whitened stretch: its census (no transient
        reaching `GlitchTrigger`) and, against the current model, its power
        in each octave from `GlitchCensusLow` over that of white noise,
        within ``[1 / ARFitLevelRatio, ARFitLevelRatio]``."""
        from wdf.processes.census import band_stream, octave_bands
        from wdf.processes.glitch_subtraction import glitch_seeds

        par, rate, cs = self.par, float(self.par.resampling), self._fit_census
        seeds = glitch_seeds(w, start, rate, **cs.seeds()) if w.size else []
        peak = max((sd["peak"] for sd in seeds), default=0.0)
        ratios = []
        if current is not None and w.size:
            nyquist = 0.5 * rate
            for band in octave_bands(rate, cs.low):
                y = band_stream(w, rate, band, cs.order)
                ratios.append(float(np.mean(y ** 2)) / ((band[1] - band[0]) / nyquist))
        lo = min(ratios) if ratios else float("nan")
        hi = max(ratios) if ratios else float("nan")
        level_ok = not ratios or (lo >= 1.0 / par.ARFitLevelRatio and hi <= par.ARFitLevelRatio)
        return dict(census_peak=float(peak), census_transients=len(seeds), level_min=lo,
                    level_max=hi, passed=bool(not seeds and level_ok))

    def _first_stretch(self, past, past_start, now):
        """The first model's stretch: the `learn` seconds ending `ARFitGap`
        before `now`, judged by its own model's whitening. When it fails the
        rule the search waits `ARFitStep` seconds and judges the stretch then
        ending, up to `ARFitCandidates` stretches; then it takes the one of
        smallest census peak. None while waiting."""
        par, rate = self.par, float(self.par.resampling)
        tried = self._first_tried
        if tried and now < tried[-1][1] + par.ARFitGap + par.ARFitStep - 0.5 / rate:
            return None
        fit_end = now - par.ARFitGap
        fit_start = fit_end - par.learn
        a = int(round((fit_start - past_start) * rate))
        b = int(round((fit_end - past_start) * rate))
        if a < 0 or b > past.size:
            raise ValueError(f"the past held from {past_start} does not cover the fit stretch "
                             f"{fit_start}-{fit_end}")
        x = past[a:b]
        ar = fit_autoregressive(x, fit_start, rate, par.ARorder)
        filt = self._filter_of(ar, now) if self.mode == "zerophase" else None
        if not par.ARFitQuality:
            return fit_start, fit_end, ar, filt, dict(candidate=0, accepted=True)
        judge = filt or self._filter_of(ar, now)
        w = self._whitened_stretch(judge, x)
        offset = (x.size - w.size) / rate if self.mode == "causal" else 0.5 * (x.size - w.size) / rate
        q = self._stretch_quality(w, fit_start + offset, None)
        q.update(candidate=len(tried))
        tried.append((fit_start, fit_end, ar, filt, q))
        if q["passed"]:
            q["accepted"] = True
            return tried[-1]
        if len(tried) < par.ARFitCandidates:
            logging.warning(f"online: the fit stretch {fit_start:.1f}-{fit_end:.1f} holds a "
                            f"transient (census peak {q['census_peak']:.1f}); the search waits "
                            f"{par.ARFitStep:g} s for the next one")
            return None
        best = min(tried, key=lambda c: c[4]["census_peak"])
        best[4]["accepted"] = True
        logging.warning(f"online: no fit stretch passed the quality rule; the first model is "
                        f"fitted on {best[0]:.1f}-{best[1]:.1f}, census peak "
                        f"{best[4]['census_peak']:.1f}")
        if best is not tried[-1] and self.mode == "zerophase":
            best = best[:3] + (self._filter_of(best[2], now), best[4])
        return best

    def _choose_stretch(self, past, past_start, now):
        """The stretch of the past a model is fitted on, the model and its
        filter; None for a refit with no stretch passing the rule, which
        keeps the current model.

        With `ARFitQuality` (default True) each candidate is whitened, by the
        current model for a refit and by its own for the first fit, and must
        hold no census transient reaching `GlitchTrigger`; a refit's candidate
        must also keep each octave's power, through the current model, within
        `ARFitLevelRatio` of white. The first fit takes, when none passes, the
        candidate of smallest census peak.

        :return: tuple -- (fit_start, fit_end, model, filter or None, quality
            record) or None.
        """
        par, rate = self.par, float(self.par.resampling)
        current = self._model
        if current is None:
            return self._first_stretch(past, past_start, now)
        tried = []
        for j in range(par.ARFitCandidates):
            fit_end = now - par.ARFitGap - j * par.ARFitStep
            fit_start = fit_end - par.learn
            a = int(round((fit_start - past_start) * rate))
            b = int(round((fit_end - past_start) * rate))
            if a < 0 or b > past.size:
                if j == 0:
                    raise ValueError(f"the past held from {past_start} does not cover the fit "
                                     f"stretch {fit_start}-{fit_end}")
                break
            x = past[a:b]
            ar = fit_autoregressive(x, fit_start, rate, par.ARorder)
            filt = self._filter_of(ar, now) if self.mode == "zerophase" else None
            if not par.ARFitQuality:
                return fit_start, fit_end, ar, filt, dict(candidate=j, accepted=True)
            judge = current if current is not None else (filt or self._filter_of(ar, now))
            w = self._whitened_stretch(judge, x)
            offset = (x.size - w.size) / rate if self.mode == "causal" else 0.5 * (x.size - w.size) / rate
            q = self._stretch_quality(w, fit_start + offset, current)
            q.update(candidate=j)
            tried.append((fit_start, fit_end, ar, filt, q))
            if q["passed"]:
                q["accepted"] = True
                return fit_start, fit_end, ar, filt, q
        self.result.refits.append(dict(emitted_at=self.now, fit_start=float("nan"),
                                       fit_end=float("nan"), sigma=float("nan"), accepted=False,
                                       candidates=len(tried),
                                       census_peak=min(t[4]["census_peak"] for t in tried),
                                       **({"data_time": self.now - self.latency}
                                          if self.latency is not None else {})))
        logging.warning(f"online: no stretch of the past passed the quality rule at {now:.1f}; "
                        "the current model is kept")
        return None

    def _condition(self, ds, raw, t):
        """The conditioned samples a read releases and the time of the first:
        at once in the causal band-pass, one block late in the zero-phase one
        (None while its future is arriving)."""
        if ds.phase == "causal":
            return ds.process_array(raw, t)
        from wdf.processes.BandPassDownSampling import SV_to_array
        from wdf.structures.array2SeqView import array2SeqView

        view = array2SeqView(t, ds.sampling, len(raw))
        view.Fill(t, np.asarray(raw, dtype=float))
        out = ds.Process(view.SV)
        if out is None:
            return None, None
        return SV_to_array(out) * out.GetScale(), float(out.GetStart())

    # ------------------------------------------------------------ the run
    def segmentProcess(self, segment, source=None):
        """Run over one segment and write its outputs.

        :type segment: tuple
        :param segment: (gpsStart, gpsEnd): the analysis starts at gpsStart;
            the stream is read from ``gpsStart - ARFitContext - learn -
            ARFitGap``.
        :type source: ArraySource, FrameSource or None
        :param source: the strain; `FrameSource(file, channel)` when None.
        :return: OnlineResult
        """
        par = self.par
        gps_start, gps_end = float(segment[0]), float(segment[1])
        ID = f"{par.channel}_{int(gps_start)}"
        self.dir = f"{par.outdir}{par.run}/{par.itf}/{ID}/"
        os.makedirs(self.dir, exist_ok=True)
        par.dir, par.ID, par.gps, par.gpsStart = self.dir, ID, gps_start, gps_start
        if source is None:
            source = FrameSource(par.file, par.channel)
        result = self.process(source, gps_start, gps_end)
        self._write(result)
        return result

    def _start_search(self, origin, sigma):
        """The searches and the ridge's state, from `origin` on."""
        from wdf.analysis.ridge_tracking import CausalBackground, StreamingCells, StreamingRidge

        par = self.par
        par.AnalysisStart = origin
        par.sigma = sigma
        if self.census_settings is not None:
            from wdf.processes.causal_census import CausalCensus

            self._census = CausalCensus(self.census_settings, float(par.resampling))
        if self.reconstruction == "track":
            from wdf.observers.TrackObserver import TrackObserver

            self._tracker = TrackObserver(self.track_settings, float(par.resampling),
                                          getattr(par, "itf", None))
        self._searches_on, self._writers = self._searches(sigma)
        st = self._ridge_state
        if par.Ridge:
            rs = st["rs"]
            st["cells"] = StreamingCells(rs, origin, float(par.resampling))
            st["background"] = CausalBackground(rs)
            st["zpast"] = deque(maxlen=rs.slots(rs.background_s))
            if rs.drift is not None and rs.xdrop is not None:
                st["ridge"] = StreamingRidge(rs, st["cells"].origin)
            if self._tracker is not None:
                st["zrows"] = deque(maxlen=rs.max_slots)

    def _consume(self, w, w_start):
        """Search a whitened chunk and run the ridge over it."""
        from dataclasses import replace

        from wdf.analysis.ridge_tracking import StreamingRidge, calibrate_drift, ridge_waveform

        par, rate = self.par, float(self.par.resampling)
        if self._census is not None:
            for row in self._census.push(w, w_start):
                row["emitted_at"] = self.now
                if self.latency is not None:
                    row["data_time"] = self.now - self.latency
                self.result.census.append(row)
        view = self._view(w, w_start)
        for search in self._searches_on:
            search.SetData(view)
            search.Process()
        if not par.Ridge:
            return
        st, running = self._ridge_state, self._running
        rs, cells = st["rs"], st["cells"]
        for row in cells.push(w):
            k = len(running["slot_end"])
            z = st["background"].update(row)
            ridge = st["ridge"]
            if (ridge is not None and self._tracker is not None and self._track_calibrated is None
                    and np.all(np.isfinite(z))):
                # A drift given for the ridge leaves its background unread:
                # the track's calibration reads it.
                st["zpast"].append(z)
            if ridge is None and np.all(np.isfinite(z)):
                st["zpast"].append(z)
                if len(st["zpast"]) >= rs.slots(rs.background_min_s):
                    b, x = calibrate_drift(np.array(st["zpast"]), rs)
                    rs = replace(rs, drift=rs.drift if rs.drift is not None else b,
                                 xdrop=rs.xdrop if rs.xdrop is not None else x)
                    st["rs"] = rs
                    par.RidgeDriftUsed, par.RidgeXDropUsed = rs.drift, rs.xdrop
                    par.RidgeCalibratedAt = self.now
                    st["ridge"] = StreamingRidge(rs, cells.origin + (k + 1) * rs.slot_s)
                score, state = np.nan, -1
            elif ridge is not None:
                for c in ridge.update(z, row):
                    row_out = c.row()
                    starts, coeffs, sigmas = cells.held()
                    if starts.size and starts[0] <= c.t_start:
                        _, _, energy = ridge_waveform(c, coeffs, sigmas, starts, rate,
                                                      ridge.t0, rs)
                    else:
                        energy = float("nan")
                    row_out.update(emitted_at=self.now, latency=self.now - c.t_end,
                                   waveform_energy=energy, search=par.SearchName)
                    if self.latency is not None:
                        row_out["data_time"] = self.now - self.latency
                    self.result.candidates.append(row_out)
                best = int(np.argmax(ridge.H))
                score, state = float(ridge.H[best]), best
            else:
                score, state = np.nan, -1
            running["slot_end"].append(cells.origin + (k + 1) * rs.slot_s)
            running["score"].append(score)
            running["state"].append(state)
            running["emitted_at"].append(self.now)
            if self.latency is not None:
                running["data_time"].append(self.now - self.latency)
            if self._tracker is not None:
                st["zrows"].append((k, z, row))
        if self._tracker is not None:
            self._drain_tracks()

    def _drain_tracks(self):
        """The tracks whose seeds' slices of the map are complete."""
        from dataclasses import replace

        st, par = self._ridge_state, self.par
        cells = st["cells"]
        need = self.track_settings.ridge.calibration_steps + 2
        if (self._track_calibrated is None and st["ridge"] is not None
                and (len(st["zpast"]) >= need or (self.track_settings.ridge.drift is not None
                                                   and self.track_settings.ridge.xdrop is not None))):
            from wdf.analysis.track_reconstruction import calibrated as _calibrated

            # The track's drift and X-drop, with its own penalty, on the past
            # background the streaming ridge was calibrated on.
            self._track_calibrated = _calibrated(self.track_settings, np.array(st["zpast"]))
            tr = self._track_calibrated.ridge
            par.TrackDriftUsed, par.TrackXDropUsed = tr.drift, tr.xdrop
        calibrated = self._track_calibrated
        rows, waveforms = self._tracker.drain(cells.origin, cells.next_slot, list(st["zrows"]),
                                              cells.held(), calibrated)
        for row in rows:
            row.update(emitted_at=self.now, latency=self.now - row["seed_end"],
                       search=par.SearchName)
            if self.latency is not None:
                row["data_time"] = self.now - self.latency
            self.result.tracks.append(row)
        self.result.track_waveforms.update(waveforms)

    def process(self, source, gps_start, gps_end):
        """Run over ``[gps_start, gps_end)`` without writing, unless
        `segmentProcess` set an output folder.

        :return: OnlineResult
        """
        from wdf.analysis.ridge_tracking import RidgeSettings
        from wdf.processes.BandPassDownSampling import BandPassDownSampling

        par = self.par
        if not hasattr(self, "dir"):
            self.dir = None
        par.gps = getattr(par, "gps", None) or gps_start
        self.result = OnlineResult()
        self._model, self._first_tried = None, []
        self._reused = self._load_reused()
        rate = float(par.resampling)
        ds = BandPassDownSampling(par)
        par.BandPassFilter, par.BandPassPassband = ds.family, list(ds.passband())
        if getattr(ds, "lowpass_order", None) is not None:
            par.LowPassOrder = int(ds.lowpass_order)
            par.AliasAttenuationReached = ds.alias_attenuation_db()
        if par.ARFitContext * ds.sampling < ds.padlen:
            raise ValueError(f"ARFitContext {par.ARFitContext} s is shorter than the band-pass's "
                             f"settling, {ds.padlen / ds.sampling:.3f} s")
        self._held = {}
        if self.mode == "zerophase" and par.HoldOutsideBand:
            from wdf.processes.wdfUnitDSWorker import DEFAULT_PASSBAND_LEVEL_DB
            from wdf.processes.zero_phase_whitening import DEFAULT_BAND_BLEND_HZ

            par.PassbandLevel = float(getattr(par, "PassbandLevel", DEFAULT_PASSBAND_LEVEL_DB))
            par.BandEdgeBlend = float(getattr(par, "BandEdgeBlend", DEFAULT_BAND_BLEND_HZ))
            band = ds.passband(par.PassbandLevel)
            par.Passband = list(band)
            self._held = dict(band=band, blend=par.BandEdgeBlend)
        block = float(par.len)
        back = (par.ARFitCandidates - 1) * par.ARFitStep
        t_read = gps_start - par.ARFitContext - par.learn - par.ARFitGap
        keep_s = par.learn + par.ARFitGap + par.ARWarmup / rate + block + back
        if self.mode == "zerophase":
            keep_s = max(keep_s, 2.0 * par.WhiteningLookahead_s + block) + block
        self._ridge_state = dict(rs=RidgeSettings.from_parameters(par, rate) if par.Ridge else None,
                                 cells=None, background=None, zpast=None, ridge=None)
        self._searches_on, self._writers = [], []
        self._tracker = None
        self._track_calibrated = None
        self._running = dict(slot_end=[], score=[], state=[], emitted_at=[])
        if self.mode == "zerophase":
            self._running["data_time"] = []
        past, past_start = np.zeros(0), None
        whitener = None
        filters, released, in_search = [], None, None
        next_refit = None
        started = time.time()
        t = t_read
        while t < gps_end:
            try:
                raw = source.read(t, block)
            except EOFError as error:
                logging.warning(f"online: stopping at {t}: {error}")
                break
            y, y_start = self._condition(ds, raw, t)
            t += block
            self.now = t
            if y is None:
                continue
            past_start = y_start if past_start is None else past_start
            past = np.concatenate([past, y])
            excess = past.size - int(round(keep_s * rate))
            if excess > 0:
                past, past_start = past[excess:], past_start + excess / rate
            if self.mode == "causal":
                if whitener is None:
                    if self.now < gps_start:
                        continue
                    whitener = self._fit(past, past_start, self.now)
                    if whitener is None:
                        continue
                    self._start_search(self.now, whitener.sigma)
                    next_refit = self._next_switch(self.now)
                    continue
                w = whitener.process(y)
                self._consume(w, y_start)
                if next_refit is not None and self.now >= next_refit:
                    refit = self._fit(past, past_start, self.now)
                    if refit is not None:
                        whitener = refit
                        for search in self._searches_on:
                            search.parameters.sigma = whitener.sigma
                    next_refit = self._next_switch(next_refit)
                continue
            # ---- "zerophase": the conditioned stream ends at `clock`, the
            # whitened one at clock - WhiteningLookahead_s = now - Latency_s.
            clock = y_start + y.size / rate
            if abs((self.now - clock) - par.BandPassLookahead_s) > 0.5 / rate:
                raise RuntimeError(f"the conditioned stream ends at {clock}, {self.now - clock} s "
                                   f"behind the read, not BandPassLookahead_s "
                                   f"{par.BandPassLookahead_s}")
            if not filters:
                if clock < gps_start:
                    continue
                first = self._fit(past, past_start, clock)
                if first is None:
                    continue
                filters.append(first)
                released, in_search = clock, filters[0]
                self._start_search(clock, filters[0].sigma)
                next_refit = self._next_switch(clock)
                continue
            ready = clock - par.WhiteningLookahead_s
            while ready - released > 0.5 / rate:
                active = [f for f in filters if f.start <= released + 0.5 / rate]
                current = active[-1]
                later = [f.start for f in filters if f.start > released + 0.5 / rate]
                end = min([ready] + later)
                a = int(round((released - past_start) * rate))
                b = int(round((end - past_start) * rate))
                if current is not in_search:
                    for search in self._searches_on:
                        search.parameters.sigma = current.sigma
                    in_search = current
                self._consume(current.apply(past, a, b), released)
                released = past_start + b / rate
            # keep only the filters still needed
            while len(filters) > 1 and filters[1].start <= released + 0.5 / rate:
                filters.pop(0)
            if next_refit is not None and clock >= next_refit:
                refit = self._fit(past, past_start, clock)
                if refit is not None:
                    filters.append(refit)
                next_refit = self._next_switch(next_refit)
        for writer in self._writers:
            writer.close()
        par.RefitCount = sum(1 for r in self.result.refits if r.get("accepted", True))
        par.RefitRejected = len(self.result.refits) - par.RefitCount
        par.ElapsedSeconds = time.time() - started
        if par.Ridge:
            par.RidgeSettingsUsed = {k: (list(v) if isinstance(v, tuple) else v)
                                     for k, v in self._ridge_state["rs"].record().items()}
        if self._tracker is not None:
            par.TrackSeeds = len(self._tracker.records)
            par.TrackSeedsDropped = self._tracker.dropped
            par.TrackSeedsAbsorbed = self._tracker.absorbed
            par.TrackSeedsPending = len(self._tracker.queue)
            par.TrackCount = len(self.result.tracks)
        self.result.running = {k: np.asarray(v) for k, v in self._running.items()}
        self.result.parameters = par
        return self.result

    def _write(self, result):
        """The run's outputs beside its triggers."""
        import pandas as pd

        par, suffix = self.par, search_suffix(self.par)
        pd.DataFrame(result.triggers).to_csv(f"{self.dir}TriggerLatency{suffix}.csv", index=False)
        pd.DataFrame(result.candidates).to_csv(f"{self.dir}RidgeCandidates{suffix}.csv", index=False)
        pd.DataFrame(result.refits).to_csv(f"{self.dir}ARRefits{suffix}.csv", index=False)
        par.ARModelsFile = f"{self.dir}ARModels-{par.itf}{suffix}.h5"
        attributes = dict(ARorder=int(par.ARorder), WhiteningMode=self.mode,
                          resampling=float(par.resampling), itf=str(par.itf))
        if self.mode == "zerophase":
            attributes["ZeroPhaseDuration"] = float(par.ZeroPhaseDuration)
        save_ar_models(par.ARModelsFile, result.models, attributes)
        if self._census is not None:
            pd.DataFrame(result.census).to_csv(f"{self.dir}GlitchCensusOnline{suffix}.csv",
                                               index=False)
        np.savez(f"{self.dir}RidgeRunning{suffix}.npz", **result.running)
        if self.reconstruction == "track":
            from wdf.analysis.track_reconstruction import TRACK_EVENT_COLUMNS, save_tracks

            columns = TRACK_EVENT_COLUMNS + ["seed_end", "emitted_at", "latency", "search"] + (
                ["data_time"] if self.latency is not None else [])
            save_tracks(pd.DataFrame(result.tracks, columns=columns), result.track_waveforms,
                        f"{self.dir}TrackEvents{suffix}.csv", f"{self.dir}TrackWaveforms{suffix}.npz")
        par.dump(f"{self.dir}parametersUsed-Online{suffix}.json")
