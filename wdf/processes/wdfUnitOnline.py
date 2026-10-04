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
   samples (default `ARorder`), which sets its state exactly.
3. **The search:** `wdf` at each window of the schedule, with `Bases`,
   `MinFrequency`, `ThresholdRule`, `BlockLength`, `BlockLambda` and
   `BlockRemainder` from the parameters. **No glitch subtraction and no
   gating:** the online search finds glitches too. Each trigger is labelled
   ("glitch" when the basis it won is one of `GlitchLabelBases`, else
   "candidate") and kept.
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
from wdf.config.Parameters import Parameters, low_frequency_cut, window_schedule
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
    :param parameters: the parameters used.

    In the "zerophase" mode every row also holds ``data_time``,
    ``emitted_at - Latency_s``: the end of the whitened stream at emission.
    """

    triggers: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    running: dict = field(default_factory=dict)
    refits: list = field(default_factory=list)
    parameters: object = None

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
        return dict(triggers=keep(self.triggers), candidates=keep(self.candidates),
                    running=running, refits=keep(self.refits))


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
        row.update(window_end=row["gps"] + self.duration, emitted_at=now,
                   latency=now - (row["gps"] + self.duration),
                   label="glitch" if row["wave"] in self.label_bases else "candidate",
                   search=self.par.SearchName)
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
        par.Ncoeff = max(w for w, _ in self.schedule)
        par.Ridge = bool(getattr(par, "Ridge", True))
        par.GlitchLabelBases = list(getattr(par, "GlitchLabelBases", None) or DEFAULT_GLITCH_LABEL_BASES)
        par.ARFitGap = float(getattr(par, "ARFitGap", None) or 0.0)
        par.ARFitContext = float(getattr(par, "ARFitContext", None) or DEFAULT_FIT_CONTEXT_S)
        par.ARRefitInterval = float(getattr(par, "ARRefitInterval", None) or 0.0)
        warm = getattr(par, "ARWarmup", None)
        par.ARWarmup = int(par.ARorder if warm is None else warm)
        if par.ARWarmup < par.ARorder:
            raise ValueError(f"ARWarmup {par.ARWarmup} samples is shorter than the model's order "
                             f"{par.ARorder}: the filter's state would not be set by real data")
        self.now = None
        self.result = None
        #: The fixed read-ahead, s; None in the causal mode, which has none.
        self.latency = None
        if mode == "zerophase":
            self._declare_latency()

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
            if self.dir is not None:
                writer = SingleEventPrintTriggers(par)
                estimation.register(writer)
                writers.append(writer)
                par.dump(par.dir + parameters_file_name(par))
            search.register(estimation)
            searches.append(search)
        return searches, writers

    def _fit(self, past, past_start, now):
        """A model fitted on the past only and its filter, from `now` on.

        In the causal mode `now` is the read time and the filter is warmed up
        to it; in the "zerophase" mode it is the end of the conditioned stream,
        and the filter whitens the samples from it on.
        """
        par, rate = self.par, float(self.par.resampling)
        fit_end = now - par.ARFitGap
        fit_start = fit_end - par.learn
        a = int(round((fit_start - past_start) * rate))
        b = int(round((fit_end - past_start) * rate))
        if a < 0 or b > past.size:
            raise ValueError(f"the past held from {past_start} does not cover the fit stretch "
                             f"{fit_start}-{fit_end}")
        ar = fit_autoregressive(past[a:b], fit_start, rate, par.ARorder)
        if self.mode == "causal":
            warm = past[past.size - par.ARWarmup:]
            self.result.refits.append(dict(emitted_at=self.now, fit_start=fit_start,
                                           fit_end=fit_end, sigma=float(ar[0])))
            logging.info(f"online: AR {par.ARorder} fitted on {fit_start:.1f}-{fit_end:.1f}, "
                         f"sigma {ar[0]:.4g}")
            return CausalPredictionError(ar, warm)
        filt = LookaheadWhitening(ar, par.ZeroPhaseDuration, rate, now, self._held)
        self.result.refits.append(dict(emitted_at=self.now, fit_start=fit_start, fit_end=fit_end,
                                       sigma=filt.sigma, ar_sigma=float(ar[0]), whitens_from=now,
                                       data_time=self.now - self.latency))
        logging.info(f"online zerophase: AR {par.ARorder} fitted on {fit_start:.1f}-{fit_end:.1f}, "
                     f"TruncatedWhitening {par.ZeroPhaseDuration:g} s from {now:.1f}, "
                     f"sigma {filt.sigma:.4g}")
        return filt

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
        self._searches_on, self._writers = self._searches(sigma)
        st = self._ridge_state
        if par.Ridge:
            rs = st["rs"]
            st["cells"] = StreamingCells(rs, origin, float(par.resampling))
            st["background"] = CausalBackground(rs)
            st["zpast"] = deque(maxlen=rs.slots(rs.background_s))
            if rs.drift is not None and rs.xdrop is not None:
                st["ridge"] = StreamingRidge(rs, st["cells"].origin)

    def _consume(self, w, w_start):
        """Search a whitened chunk and run the ridge over it."""
        from dataclasses import replace

        from wdf.analysis.ridge_tracking import StreamingRidge, calibrate_drift, ridge_waveform

        par, rate = self.par, float(self.par.resampling)
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
        t_read = gps_start - par.ARFitContext - par.learn - par.ARFitGap
        keep_s = par.learn + par.ARFitGap + par.ARWarmup / rate + block
        if self.mode == "zerophase":
            keep_s = max(keep_s, 2.0 * par.WhiteningLookahead_s + block) + block
        self._ridge_state = dict(rs=RidgeSettings.from_parameters(par, rate) if par.Ridge else None,
                                 cells=None, background=None, zpast=None, ridge=None)
        self._searches_on, self._writers = [], []
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
                    self._start_search(self.now, whitener.sigma)
                    next_refit = self.now + par.ARRefitInterval if par.ARRefitInterval > 0 else None
                    continue
                w = whitener.process(y)
                self._consume(w, y_start)
                if next_refit is not None and self.now >= next_refit:
                    whitener = self._fit(past, past_start, self.now)
                    for search in self._searches_on:
                        search.parameters.sigma = whitener.sigma
                    next_refit += par.ARRefitInterval
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
                filters.append(self._fit(past, past_start, clock))
                released, in_search = clock, filters[0]
                self._start_search(clock, filters[0].sigma)
                next_refit = clock + par.ARRefitInterval if par.ARRefitInterval > 0 else None
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
                filters.append(self._fit(past, past_start, clock))
                next_refit += par.ARRefitInterval
        for writer in self._writers:
            writer.close()
        par.RefitCount = len(self.result.refits)
        par.ElapsedSeconds = time.time() - started
        if par.Ridge:
            par.RidgeSettingsUsed = {k: (list(v) if isinstance(v, tuple) else v)
                                     for k, v in self._ridge_state["rs"].record().items()}
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
        np.savez(f"{self.dir}RidgeRunning{suffix}.npz", **result.running)
        par.dump(f"{self.dir}parametersUsed-Online{suffix}.json")
