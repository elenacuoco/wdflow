"""The online search: causal from the frames to the triggers and the ridge.

`wdfUnitOnline` never uses information from the future. Every output carries
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
        self.worker.result.triggers.append(row)


# ================================================================ the worker
class wdfUnitOnline:
    """The causal search of one detector's stream.

    :type parameters: wdf.config.Parameters.Parameters
    :param parameters: the run configuration (`configs/online_default.toml`
        through `load_config`); copied, so the caller's object is not changed.
    """

    def __init__(self, parameters):
        self.par = Parameters()
        self.par.copy(parameters)
        par = self.par
        par.resampling = par.sampling / par.ResamplingFactor
        phase = str(getattr(par, "BandPassPhase", None) or "causal").lower()
        if phase != "causal":
            raise ValueError(f"the online worker runs the causal band-pass; BandPassPhase is {phase!r}")
        par.BandPassPhase = phase
        par.LowFrequencyCut = low_frequency_cut(par)
        par.SearchName = getattr(par, "SearchName", None) or DEFAULT_SEARCH_NAME
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
        """A model fitted on the past only and its filter, warmed up to `now`."""
        par, rate = self.par, float(self.par.resampling)
        fit_end = now - par.ARFitGap
        fit_start = fit_end - par.learn
        a = int(round((fit_start - past_start) * rate))
        b = int(round((fit_end - past_start) * rate))
        if a < 0 or b > past.size:
            raise ValueError(f"the past held from {past_start} does not cover the fit stretch "
                             f"{fit_start}-{fit_end}")
        ar = fit_autoregressive(past[a:b], fit_start, rate, par.ARorder)
        warm = past[past.size - par.ARWarmup:]
        self.result.refits.append(dict(emitted_at=now, fit_start=fit_start, fit_end=fit_end,
                                       sigma=float(ar[0])))
        logging.info(f"online: AR {par.ARorder} fitted on {fit_start:.1f}-{fit_end:.1f}, "
                     f"sigma {ar[0]:.4g}")
        return CausalPredictionError(ar, warm)

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

    def process(self, source, gps_start, gps_end):
        """Run over ``[gps_start, gps_end)`` without writing, unless
        `segmentProcess` set an output folder.

        :return: OnlineResult
        """
        from wdf.analysis.ridge_tracking import (CausalBackground, RidgeSettings, StreamingCells,
                                                 StreamingRidge, calibrate_drift, ridge_waveform)
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
        block = float(par.len)
        t_read = gps_start - par.ARFitContext - par.learn - par.ARFitGap
        keep_s = par.learn + par.ARFitGap + par.ARWarmup / rate + block
        ridge_on = par.Ridge
        if ridge_on:
            rs = RidgeSettings.from_parameters(par, rate)
        past, past_start = np.zeros(0), None
        whitener, searches, writers = None, [], []
        cells = background = ridge = None
        next_refit = None
        running = dict(slot_end=[], score=[], state=[], emitted_at=[])
        zpast = None
        started = time.time()
        t = t_read
        while t < gps_end:
            try:
                raw = source.read(t, block)
            except EOFError as error:
                logging.warning(f"online: stopping at {t}: {error}")
                break
            y, y_start = ds.process_array(raw, t)
            t += block
            self.now = t
            past_start = y_start if past_start is None else past_start
            past = np.concatenate([past, y])
            excess = past.size - int(round(keep_s * rate))
            if excess > 0:
                past, past_start = past[excess:], past_start + excess / rate
            if whitener is None:
                if self.now < gps_start:
                    continue
                whitener = self._fit(past, past_start, self.now)
                par.AnalysisStart = self.now
                par.sigma = whitener.sigma
                searches, writers = self._searches(whitener.sigma)
                next_refit = self.now + par.ARRefitInterval if par.ARRefitInterval > 0 else None
                if ridge_on:
                    cells = StreamingCells(rs, self.now, rate)
                    background = CausalBackground(rs)
                    zpast = deque(maxlen=rs.slots(rs.background_s))
                    if rs.drift is not None and rs.xdrop is not None:
                        ridge = StreamingRidge(rs, cells.origin)
                continue
            w = whitener.process(y)
            view = self._view(w, y_start)
            for search in searches:
                search.SetData(view)
                search.Process()
            if ridge_on:
                for row in cells.push(w):
                    k = len(running["slot_end"])
                    z = background.update(row)
                    if ridge is None and np.all(np.isfinite(z)):
                        zpast.append(z)
                        if len(zpast) >= rs.slots(rs.background_min_s):
                            b, x = calibrate_drift(np.array(zpast), rs)
                            from dataclasses import replace
                            rs = replace(rs, drift=rs.drift if rs.drift is not None else b,
                                         xdrop=rs.xdrop if rs.xdrop is not None else x)
                            par.RidgeDriftUsed, par.RidgeXDropUsed = rs.drift, rs.xdrop
                            par.RidgeCalibratedAt = self.now
                            ridge = StreamingRidge(rs, cells.origin + (k + 1) * rs.slot_s)
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
                            self.result.candidates.append(row_out)
                        best = int(np.argmax(ridge.H))
                        score, state = float(ridge.H[best]), best
                    else:
                        score, state = np.nan, -1
                    running["slot_end"].append(cells.origin + (k + 1) * rs.slot_s)
                    running["score"].append(score)
                    running["state"].append(state)
                    running["emitted_at"].append(self.now)
            if next_refit is not None and self.now >= next_refit:
                whitener = self._fit(past, past_start, self.now)
                for search in searches:
                    search.parameters.sigma = whitener.sigma
                next_refit += par.ARRefitInterval
        for writer in writers:
            writer.close()
        par.RefitCount = len(self.result.refits)
        par.ElapsedSeconds = time.time() - started
        if ridge_on:
            par.RidgeSettingsUsed = {k: (list(v) if isinstance(v, tuple) else v)
                                     for k, v in rs.record().items()}
        self.result.running = {k: np.asarray(v) for k, v in running.items()}
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
