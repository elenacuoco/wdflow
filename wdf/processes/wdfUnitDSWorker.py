"""One segment, from frames to trigger files, with the conditioning downsampled.

The unit of work a run is divided into. It reads a segment block by block,
conditions it once --- band-pass, downsample, whiten --- and searches the
conditioned stream at the configured analysis window, writing the triggers of
that segment.

The conditioning is shared and the search reads its output, so the expensive
part of the chain is done once per block however the search is configured.
"""
__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"
__status__ = "Development"
import time

import numpy as np

from py4tsa.tsa import *
from py4tsa.tsa import WaveletThreshold
from py4tsa.tsa import SeqView_double_t as SV


from wdf.observers.ParameterEstimationObserver import ParameterEstimation 
from wdf.observers.SingleEventPrintFileObserver import SingleEventPrintTriggers

from wdf.processes.BandPassDownSampling import (BandPassDownSampling,
                                                discard_edges,
                                                read_conditioned)
from wdf.config.Parameters import (Parameters, check_bases, low_frequency_cut, threshold_rule,
                                   window_schedule)
from wdf.processes.wdf import wdf
from wdf.analysis.io import parameters_file_name
from wdf.processes.Whitening import CausalWhitening, Whitening
from wdf.processes.zero_phase_whitening import (
    DEFAULT_BAND_BLEND_HZ,
    DEFAULT_RESPONSE_FLOOR,
    DEFAULT_SQRT_ORDER,
    DEFAULT_TRUNCATION_S,
    MagnitudeWhitening,
    TruncatedWhitening,
    ZeroPhaseWhitening,
)

DEFAULT_AR_ESTIMATION_OFFSET_S = 50.0
#: Seconds of real data read on each side of the fit stretch and dropped after
#: the band-pass, so that the model is fitted on samples the filter reached
#: settled. It must be at least the band-pass's settling length, which is
#: checked; the default is several settling lengths of the default filter.
DEFAULT_AR_FIT_CONTEXT_S = 20.0
#: The whitening filters the worker can run: "root", the default, is
#: `ZeroPhaseWhitening`, the fitted square root run forward and backward;
#: "magnitude" is `MagnitudeWhitening`, the response ``|A|`` itself; "causal" is
#: `CausalWhitening`, the fitted lattice filter run forward only, at zero
#: latency and with ``A``'s phase; "truncated", experimental, is
#: `TruncatedWhitening`, the model's inverse spectrum truncated
#: to `Parameters.ZeroPhaseDuration` seconds, gwpy's whitening with the Burg
#: model's ASD.
ZERO_PHASE_FILTERS = ("root", "magnitude", "causal", "truncated")
DEFAULT_ZERO_PHASE_FILTER = "root"
#: Level of the band-pass's ``|H|^2`` that defines the passband the whitening
#: whitens (`BandPassDownSampling.passband`), dB.
DEFAULT_PASSBAND_LEVEL_DB = -3.0
import logging
import os






def search_names(schedule, name=None):
    """The schedule's searches with the name each one's files carry.

    The searches of a schedule that share a window length would write the same
    files; each of them gets the run's `SearchName` (or "s") followed by its
    position in the schedule. A window searched once keeps the run's name.

    :type schedule: list[tuple[int, int]]
    :param schedule: (window, overlap) pairs, as `window_schedule` returns them.
    :type name: str or None
    :param name: the run's `SearchName`.
    :return: list[tuple[int, int, str or None]] -- (window, overlap, name).
    """
    windows = [w for w, _ in schedule]
    out = []
    for i, (window, overlap) in enumerate(schedule):
        own = name
        if windows.count(window) > 1:
            own = f"{name or 's'}{i}"
        out.append((window, overlap, own))
    return out


class wdfUnitDSWorker(object):
    def __init__(self, parameters):
        """
        :type parameters: class Parameters object
        :param parameters: run configuration (channel, sampling, window, downsampling,
            AR whitening order, learn length, output paths, ...); copied onto a fresh
            `Parameters` instance so per-worker mutations (e.g. `Ncoeff`, `resampling`,
            `sigma`, set during `segmentProcess`) don't leak back into the caller's object.
        """
        self.par = Parameters()
        self.par.copy(parameters)
        self.schedule = window_schedule(parameters)
        check_bases(parameters, self.schedule)
        self.par.Ncoeff = max(window for window, _ in self.schedule)
        self.par.channel = parameters.channel
        self.learn = parameters.learn
        self.par.resampling=parameters.sampling/parameters.ResamplingFactor
        self.par.len=parameters.len
           
    def _learn_stretch(self, gpsStart, gpsEnd):
        """The conditioned stretch a noise model is fitted on.

        `AREstimationOffset` seconds into the segment, `learn` seconds long,
        conditioned by the estimation front end -- the same stretch whichever
        way the filter is then fitted, so that the two are comparable.

        The band-pass run over a stretch in one shot starts and ends on an
        assumed boundary, and its settling there is a transient of the filter
        at the band edges, where the conditioned data are weakest. A Burg fit
        estimates mean power, so it absorbs that transient into the model and
        over-estimates the spectrum near the band edge; the whitening built
        from the model then leaves that band below white. The stretch is
        therefore read with `ARFitContext` seconds of real data on each side,
        conditioned whole, and those sides are dropped: every sample fitted is
        one the filter reached with real data behind it and ahead of it.

        :type gpsStart: float
        :param gpsStart: start of the segment.
        :type gpsEnd: float
        :param gpsEnd: end of the segment.
        :return: py4tsa.tsa.SeqView_double_t -- the conditioned stretch,
            `learn` seconds, labelled with the time of its first sample.
        :raises ValueError: if `ARFitContext` is shorter than the band-pass's
            settling length, or the frames do not return the whole stretch
            with its context.
        """
        offset = getattr(self.par, "AREstimationOffset",
                         DEFAULT_AR_ESTIMATION_OFFSET_S)
        if gpsEnd - gpsStart >= self.learn + offset:
            gpsE = gpsStart + offset
        else:
            gpsE = gpsEnd - self.learn

        front = BandPassDownSampling(self.par, estimation=True)
        context = float(getattr(self.par, "ARFitContext", DEFAULT_AR_FIT_CONTEXT_S))
        if context * front.sampling < front.padlen:
            raise ValueError(
                f"ARFitContext is {context} s, shorter than the band-pass's "
                f"settling length of {front.padlen / front.sampling:.3f} s: the "
                f"fitted stretch would still hold the filter's edge transient")
        self.par.ARFitContext = context

        first, length = gpsE - context, self.learn + 2.0 * context
        stream = FrameIChannel(self.par.file, self.par.channel, length, first)
        raw = SV()
        stream.GetData(raw)
        # The context is only context if it is real data at the times asked
        # for; a reader that returns a shifted or short stretch would put the
        # edge transient back inside the fitted samples.
        expected = round(length / raw.GetSampling())
        if (abs(raw.GetStart() - first) > 0.5 * raw.GetSampling()
                or raw.GetSize() != expected):
            raise ValueError(
                f"asked for {length} s from GPS {first} for the fit stretch and "
                f"its context, got {raw.GetSize()} samples from GPS "
                f"{raw.GetStart()}")
        return discard_edges(front.Process(raw), context)

    def _conditioning(self, gpsStart, gpsEnd, dir_chunk, ID):
        """The segment's noise model, whitening filter and conditioning front end.

        Fits (or loads) the autoregressive model, builds the whitening filter
        `ZeroPhaseFilter` names and the band-pass, and sets the warm-up, recording
        every value in the run parameters.

        :type gpsStart: float
        :param gpsStart: start of the segment.
        :type gpsEnd: float
        :param gpsEnd: end of the segment.
        :type dir_chunk: str
        :param dir_chunk: the segment's output folder.
        :type ID: str
        :param ID: the segment's name.
        :return: tuple -- (whitening filter, `BandPassDownSampling`).
        """
        # self.parameter for whitening and its estimation self.parameters
        whiten = Whitening(self.par.ARorder)
        # .h5 (not .txt): Whitening.ParametersSave/Load now use HDF5
        # (wdf.processes.ar_lv_io), not p4TSA's old XML Save/Load.
        self.par.ARfile = dir_chunk + "ARcoeff-AR%s-fs%s-%s.h5" % (
            self.par.ARorder,
            self.par.resampling,
            self.par.channel,
        )
        self.par.LVfile = dir_chunk + "LVcoeff-AR%s-fs%s-%s.h5" % (
            self.par.ARorder,
            self.par.resampling,
            self.par.channel,
        )

        if os.path.isfile(self.par.ARfile) and os.path.isfile(self.par.LVfile):
            logging.info("Load AR parameters")
            whiten.ParametersLoad(self.par.ARfile, self.par.LVfile)
             
        else:
            logging.info("Start AR parameter estimation")
            Learn_DS = self._learn_stretch(gpsStart, gpsEnd)
            whiten.ParametersEstimate(Learn_DS)
            self._after_fit(Learn_DS, whiten)
            whiten.ParametersSave(self.par.ARfile, self.par.LVfile)
            del Learn_DS
            
        # sigma for the noise
        self.par.sigma = whiten.GetSigma()
        logging.info("Estimated sigma= %s" % self.par.sigma)

        # Which filter whitens the stream. "root" is the square-root model
        # run forward and backward, which approximates |A| at a finite
        # order; "magnitude" has the response |A| itself, so its whitened
        # spectrum is the causal whitening's; "causal" is the lattice filter
        # A(z) run forward only, at zero latency and with A's phase.
        zero_phase = str(getattr(self.par, "ZeroPhaseFilter",
                                 DEFAULT_ZERO_PHASE_FILTER)).lower()
        if zero_phase not in ZERO_PHASE_FILTERS:
            raise ValueError(f"ZeroPhaseFilter is {zero_phase!r}; expected one "
                             f"of {ZERO_PHASE_FILTERS}")
        self.par.ZeroPhaseFilter = zero_phase
        floor = float(getattr(self.par, "ZeroPhaseResponseFloor",
                              DEFAULT_RESPONSE_FLOOR))
        self.par.ZeroPhaseResponseFloor = floor
        if zero_phase == "truncated":
            duration = float(getattr(self.par, "ZeroPhaseDuration",
                                     DEFAULT_TRUNCATION_S))
            self.par.ZeroPhaseDuration = duration

        # Coefficients of the square-root model the "root" filter runs in
        # both directions (see wdf.processes.zero_phase_whitening).
        # Unset means the model's own order: a lower one costs accuracy
        # twice over, since the response is the square of the filter's
        # magnitude.
        sqrt_order = getattr(self.par, "SqrtWhiteningOrder", None)
        sqrt_order = (max(DEFAULT_SQRT_ORDER, self.par.ARorder)
                      if sqrt_order is None else int(sqrt_order))
        self.par.SqrtWhiteningOrder = sqrt_order
        ar = np.array([whiten.ADE.GetAR(j)
                       for j in range(self.par.ARorder + 1)])

        # Which fit the whitening filter comes from. "burg" is the
        # historical path and the default: the autoregressive model above,
        # then its square root. "spectrum" fits the same filter straight to
        # the measured spectrum of the same stretch, which is one fit
        # instead of two and weighs its error in decibels across the band
        # rather than in absolute power -- Burg has no incentive to fit an
        # octave 60 dB below the one that carries the power, and on O4b
        # strain it leaves the whitened spectrum a factor 2.9 low below
        # 32 Hz. The autoregressive model is estimated and saved either
        # way, so a run can be read back and compared against the other.
        model = str(getattr(self.par, "WhiteningModel", "burg")).lower()
        if model not in ("burg", "spectrum"):
            raise ValueError(
                f"WhiteningModel is {model!r}; expected 'burg' or 'spectrum'")
        
        # update the self.parameters to be saved in local json file
        self.par.ID = ID
        self.par.dir = dir_chunk
        self.par.gps = gpsStart
        self.par.gpsStart = gpsStart
        

        ######################
        # self.parameter for sequence of data and the resampling
    
        ds = BandPassDownSampling(self.par)
        # What the conditioning was, recorded with the rest of the run's
        # parameters: its family decides where the band edges are.
        self.par.BandPassFilter = ds.family
        self.par.BandPassPassband = list(ds.passband())
        
        #Perform operation to intialite the detection loop    
        #gpsStart = gpsStart - self.par.preWhite            
        data = SV()
        data_ds = SV()
        dataw = SV()
        Noutdata = int(self.par.resampling)

        # The band the whitening whitens. The model is fitted on band-passed
        # data, so over the full band |A| also inverts the band-pass's stop
        # band -- a gain up to 1e6 on nothing -- which the whitening must
        # not undo: the stop band belongs to the band-pass alone. Held, the
        # target is |A| inside the band-pass's own passband and its edge
        # values outside it. False keeps the full band, and is the default
        # until the search is calibrated on a stream white in band only.
        hold = bool(getattr(self.par, "HoldOutsideBand", False))
        self.par.HoldOutsideBand = hold
        if hold:
            level = float(getattr(self.par, "PassbandLevel",
                                  DEFAULT_PASSBAND_LEVEL_DB))
            blend = float(getattr(self.par, "BandEdgeBlend",
                                  DEFAULT_BAND_BLEND_HZ))
            band = ds.passband(level)
            self.par.PassbandLevel, self.par.BandEdgeBlend = level, blend
            self.par.Passband = list(band)
            held = dict(band=band, sampling=self.par.resampling, blend=blend)
            logging.info(f"Whitening held outside the passband "
                         f"{band[0]:.3f}-{band[1]:.3f} Hz ({level:g} dB), "
                         f"blend {blend:g} Hz")
        else:
            band, blend = (low_frequency_cut(self.par), 0.5 * self.par.resampling), 0.0
            held = {}
        if zero_phase == "causal" and (model == "spectrum" or hold):
            raise ValueError("ZeroPhaseFilter 'causal' runs the fitted "
                             "autoregressive model over the full band; it "
                             "needs WhiteningModel 'burg' and HoldOutsideBand "
                             "False")
        if model == "spectrum":
            learn = self._learn_stretch(gpsStart, gpsEnd)
            samples = np.array([learn.GetY(0, i) for i in range(learn.GetSize())])
            del learn
            if zero_phase == "magnitude":
                whitening = MagnitudeWhitening.from_spectrum(
                    samples, self.par.resampling, Noutdata, 0, band=band,
                    blend=blend)
            elif zero_phase == "truncated":
                # A Welch estimate on segments of the filter's length and
                # a Hann truncation: gwpy's TimeSeries.whiten.
                whitening = MagnitudeWhitening.from_spectrum(
                    samples, self.par.resampling, Noutdata, 0, band=band,
                    blend=blend, taper=1.0,
                    nperseg=2 * int(round(0.5 * duration * self.par.resampling)))
            else:
                whitening = ZeroPhaseWhitening.from_spectrum(
                    samples, self.par.resampling, Noutdata, 0,
                    order=sqrt_order, band=band, blend=blend)
        elif zero_phase == "magnitude":
            whitening = MagnitudeWhitening(ar, Noutdata, 0, floor=floor, **held)
        elif zero_phase == "causal":
            whitening = CausalWhitening(whiten, Noutdata, 0)
        elif zero_phase == "truncated":
            whitening = TruncatedWhitening(
                ar, Noutdata, 0, duration=duration,
                **dict(held, sampling=self.par.resampling))
        else:
            whitening = ZeroPhaseWhitening(ar, Noutdata, 0, order=sqrt_order,
                                           **held)
        if model == "spectrum" or hold or zero_phase == "truncated":
            # The scale of the stream the search is given is this filter's
            # level in band (in_band_scale, whitened_level): the stream
            # over it is at unit density in band. Only the "cuoco" rule
            # thresholds on it; "block" and "dohonojohnston" read the
            # scale of each window from its own coefficients.
            self.par.sigma = whitening.sigma
        self.par.sigmaWhitened = whitening.sigma
        self.par.ZeroPhaseLatency = int(whitening.latency)
        logging.info(f"Whitening model: {model}, whitening filter: {zero_phase}")
        root = f", square-root order {sqrt_order}" if zero_phase == "root" else ""
        logging.info(f"Whitening latency {whitening.latency} samples "
                     f"({whitening.latency / self.par.resampling:.3f} s){root}")

        # The warm-up whitens data that is then discarded, and it has to
        # last until the filter's past is real data: a zero-phase output
        # sample depends on `latency` samples before it, zeros until the
        # stream has supplied them; a causal one on the model's `ARorder`
        # samples before it, which have to be conditioned samples the
        # band-pass reached settled. `preWhite` is the floor; a longer past
        # lengthens the warm-up rather than emitting a start-up transient.
        if zero_phase == "causal":
            past_s = (self.par.ARorder / self.par.resampling
                      + ds.padlen / ds.sampling)
        else:
            past_s = whitening.latency / self.par.resampling
        pre_white = max(int(self.par.preWhite), int(np.ceil(past_s)))
        if pre_white > int(self.par.preWhite):
            logging.info(f"Warm-up lengthened from {self.par.preWhite} to "
                         f"{pre_white} s to cover the whitening's past")
        self.par.preWhite = pre_white
        return whitening, ds

    def _read_block(self, streaming, block, downsampling):
        """The next conditioned block the whitening reads.

        :return: py4tsa.tsa.SeqView_double_t -- as `read_conditioned`; a
            subclass may clean it.
        """
        return read_conditioned(streaming, block, downsampling)

    def _after_fit(self, learn, whiten):
        """Called once the noise model is fitted on `learn`, before it is saved.

        :type learn: py4tsa.tsa.SeqView_double_t
        :param learn: the conditioned fit stretch.
        :type whiten: Whitening
        :param whiten: the fitted model, which a subclass may refit.
        :return: None
        """

    def _prepare_segment(self, gpsStart, gpsEnd, whitening, dir_chunk):
        """Called once the whitening is built, before the stream is read.

        :return: None
        """

    def _on_whitened(self, dataw):
        """Called with every whitened block the searches read.

        :type dataw: py4tsa.tsa.SeqView_double_t
        :param dataw: the whitened block.
        :return: None
        """

    def _observers(self, par):
        """Observers registered, beside the trigger writer, on each search's
        parameter estimation; none here, a subclass may add its own.

        :param par: the search's parameters.
        :return: list of `Observer`
        """
        return []

    def _finish_segment(self, dir_chunk):
        """Called once the segment's stream is exhausted, before the trigger
        files are closed.

        :return: None
        """

    #: Keys of a search's own parameters, which the end of the segment does not
    #: overwrite in its parametersUsed file.
    SEARCH_KEYS = ("window", "overlap", "Ncoeff", "SearchName", "waveletThreshold", "filename")

    def _record_end(self, searched, gpsEnd, ds, search_pars):
        """What the segment's run used and searched, written again into every
        search's parametersUsed file once the segment is done.

        `gpsEndSearched` is the end of the last whitened block the searches
        read and `UnsearchedTail_s` what is left of the segment after it
        (`gpsEndSegment`; `gpsEnd` is the last conditioned block start the
        loop accepts). The tail is what the chain reads ahead, summed in
        `ReadAheadParts_s`: ``blocks``, two blocks of `len`, since reading
        stops before a block would pass the frames' end, which the reader
        would zero-pad; ``bandpass``, the band-pass's settling `padlen`;
        ``whitening``, the whitening's look-ahead `WhiteningExtraSize`, kept
        in the margin of the conditioned blocks; and ``whitening_lag``, one
        block plus `WhiteningExtraSize`, by which the whitened output trails
        the conditioned input, since the whitening emits a block once that
        much of its future is buffered. About ``3 len + 2 WhiteningExtraSize
        + padlen``: `len` (I/O batching, which changes no trigger) is the term
        to shorten. Values set during
        the run (the ridge's drift and X-drop, the low-pass order, the track
        and glitch counts) are added.
        """
        par = self.par
        par.gpsEndSearched = float(searched) if searched is not None else float("nan")
        par.UnsearchedTail_s = float(gpsEnd - par.gpsEndSearched)
        if getattr(ds, "lowpass_order", None) is not None:
            par.LowPassOrder = int(ds.lowpass_order)
            par.AliasAttenuationReached = ds.alias_attenuation_db()
        if hasattr(ds, "passband"):
            par.BandPassPassband = list(ds.passband())
        logging.info(f"searched up to {par.gpsEndSearched:.1f}, {par.UnsearchedTail_s:.1f} s "
                     f"before the segment's end {gpsEnd:.1f}")
        for sp in search_pars:
            for key, value in vars(par).items():
                if key not in self.SEARCH_KEYS:
                    setattr(sp, key, value)
            sp.dump(sp.dir + parameters_file_name(sp))

    def segmentProcess(self, segment, wavThresh=None):
        """Runs the full offline WDF pipeline over one contiguous GPS segment:
        estimate (or load cached) AR-whitening parameters from a `learn`-second
        warm-up read, then stream the rest of the segment through
        downsampling -> zero-phase whitening -> WDF trigger search, writing triggers to
        `<outdir>/<run>/<itf>/<channel>_<gpsStart>/` as they're found.

        :type segment: tuple[float, float]
        :param segment: (gpsStart, gpsEnd) bounds of the segment to analyze.
        :type wavThresh: py4tsa.tsa.WaveletThreshold.WaveletThresholding
        :param wavThresh: the rule for the coefficients of a window, passed to WDF's C++
            engine; None (the default) reads `Parameters.ThresholdRule`
            (`wdf.config.Parameters.threshold_rule`, "block" when unset), and an explicit
            value overrides it. `block` judges contiguous coefficients of one level
            together, so that a signal spread over neighbouring coefficients, each below
            the universal threshold, survives as a block; `dohonojohnston` is that universal
            threshold on each coefficient alone. The rule is recorded beside the triggers
            as `waveletThreshold`.
        :return: None -- triggers are written to disk (Parquet, or CSV for older runs),
            not returned; a `ProcessEnded.check` marker file in the segment's output
            directory means a prior run already completed it and this call is a no-op.

        AR parameters are estimated from `Parameters.learn` seconds of data taken
        `Parameters.AREstimationOffset` seconds after the segment start (default
        `DEFAULT_AR_ESTIMATION_OFFSET_S`), read with `Parameters.ARFitContext`
        seconds of real data on each side that the band-pass settles over and
        that are then dropped (default `DEFAULT_AR_FIT_CONTEXT_S`); the frames
        must hold that context. With `Parameters.HoldOutsideBand` (default
        False) the whitening whitens only the passband of the band-pass, read off
        its design at `Parameters.PassbandLevel` dB (default -3) and recorded
        as `Passband`: the model's ``|A|`` inside, held at its edge values
        outside with a raised cosine of `Parameters.BandEdgeBlend` Hz (default
        1), so that the stop band exists in the band-pass alone; the search
        then thresholds on the whitened stream's own scale. False, the
        default, is ``|A|`` over the full band: held, the stream is white in
        the passband only, and the search reads the noise scale of each
        window as the median absolute coefficient over every level
        (`WaveletThreshold`, p4TSA), which is right on a stream white to
        Nyquist only. On the pure-noise fixture, whose passband is 36-537 Hz
        at 4096 Hz, three quarters of the coefficients are empty, the scale
        read is 13 times low, and the search returns twice the triggers at
        fifteen times the energy.
        The whitening is the filter `Parameters.ZeroPhaseFilter` names
        (`ZERO_PHASE_FILTERS`, default `DEFAULT_ZERO_PHASE_FILTER`): the square
        root run both ways at order `Parameters.SqrtWhiteningOrder`, the
        response ``|A|`` kept down to `Parameters.ZeroPhaseResponseFloor` of its
        peak, the causal lattice filter, or, "truncated" (experimental), the
        response taken to `Parameters.ZeroPhaseDuration` seconds (default
        `DEFAULT_TRUNCATION_S`) and read half of it ahead. The warm-up
        `Parameters.preWhite` is lengthened when it is shorter than the
        filter's past: its latency for the zero-phase filters, the model's
        order and the band-pass's settling for the causal one. What was used is
        recorded in the run parameters. The offset skips the beginning of a
        segment, where noise following lock acquisition can still be settling and
        would bias the noise model; set it to 0 for data known to be in science
        mode throughout. When the segment is too short to hold both the offset and
        the estimation window, the window is taken from the segment end instead.
        """
        gpsStart, gpsEnd = segment[0],segment[1]
        if wavThresh is None:
            wavThresh = threshold_rule(self.par)
        self.par.ThresholdRule = wavThresh.name
        self.par.LowFrequencyCut = low_frequency_cut(self.par)
        logging.info(
            "Analyzing segment: %s-%s for channel %s downsampled at %dHz"
            % (gpsStart, gpsEnd, self.par.channel, self.par.resampling)
        )
        start_time = time.time()
        ID = "".join([str(self.par.channel),"_",str(int(gpsStart))])
        dir_chunk = "".join([self.par.outdir,self.par.run, "/", self.par.itf,"/",ID,'/'])
        # create the output dir
        if not os.path.exists(dir_chunk):
            os.makedirs(dir_chunk)
        if not os.path.isfile(dir_chunk + "ProcessEnded.check"):
            whitening, ds = self._conditioning(gpsStart, gpsEnd, dir_chunk, ID)
            self._prepare_segment(gpsStart, gpsEnd, whitening, dir_chunk)
            data = SV()
            data_ds = SV()
            dataw = SV()
            for i in range(100):
                try:
                    streaming = FrameIChannel(self.par.file, self.par.channel, 1.0, gpsStart)
                    streaming.GetData(data)
                    break  # If no exceptions are thrown, exit the while loop
                except:
                    gpsStart=gpsStart+1.0
                    print("No frame, moving to the next one. New gpsStart is", gpsStart)
                continue  # If an exception is thrown, continue with the next iteration of the while loop
            ###---preheating---###
            streaming = FrameIChannel(self.par.file, self.par.channel, 1.0, gpsStart)
            # reading data, downsampling and whitening
            for i in range(self.par.preWhite):
                data_ds = self._read_block(streaming, data, ds)
                whitening.Process(data_ds,dataw)
               
                
            # Fixed, len-independent lookahead window for whitening: real
            # *future* data buffered beyond the output block. The "root"
            # filter's backward pass settles its lattice state over it (see
            # DoubleWhitening::GetData in p4TSA); the "magnitude" filter reads
            # its support of it. It is a FIXED size, decoupled from par.len
            # (an I/O batching/perf knob), mirroring BandPassDownSampling's
            # own padlen convention. Default: 20 seconds of resampled-rate
            # data or the filter's latency, whichever is longer, so that the
            # future a sample depends on is always read rather than assumed.
            # Set parameters.WhiteningExtraSize explicitly to override; the
            # filter refuses a positive value below its latency. 0 makes the
            # lookahead scale with par.len instead (legacy behavior).
            extra_size = int(getattr(self.par, "WhiteningExtraSize",
                                     max(20 * self.par.resampling, whitening.latency)))
            self.par.WhiteningExtraSize = extra_size

            # The chain reads ahead of what it emits, and the segment has to end
            # far enough from the frame's end to supply that. Three terms, each
            # a real buffer rather than an estimate:
            #
            #   par.len       the whitening holds a whole output block, since
            #                 DoubleWhitening::GetData needs mOutputSize +
            #                 ExtraSize buffered before it produces anything
            #   par.len       the loop reads one block past the last it uses,
            #                 because the read that ends the loop still happens
            #   padlen        the conditioning filter's backward pass settles
            #                 over this much data following the block it emits
            #   ExtraSize     the whitening's own backward lookahead
            #
            # The two read blocks were already there as a bare `2 * par.len`,
            # and that was right: what it did not cover was the conditioning
            # filter's own lookahead, which is why the reader could still run
            # off the end of the frame. Spelling the terms out costs about two
            # seconds of observation time and makes the margin follow the
            # filter instead of a constant that has to be remembered.
            read_ahead_s = (2 * self.par.len
                            + ds.padlen / self.par.sampling
                            + extra_size / self.par.resampling)
            self.par.gpsEnd = gpsEnd - read_ahead_s
            self.par.gpsEndSegment = gpsEnd
            self.par.ReadAheadParts_s = dict(blocks=2 * self.par.len,
                                             bandpass=ds.padlen / self.par.sampling,
                                             whitening=extra_size / self.par.resampling,
                                             whitening_lag=self.par.len
                                             + extra_size / self.par.resampling)

            #Set new size for the function in the loop
            streaming.SetDataLength(self.par.len)

            self.par.NoutData= int(self.par.resampling*self.par.len)
            if extra_size > 0:
                # Prime the whitening buffer before the detection loop starts.
                # DoubleWhitening::GetData needs mOutputSize + ExtraSize samples
                # buffered before it can produce anything, and each call removes
                # only mOutputSize, so the surplus is pre-loaded exactly once
                # here. whitening.Input() is SetData-only, so it neither needs
                # nor consumes an output chunk.
                #
                # This runs after SetDataLength so that the conditioning front
                # end has already flushed the short warm-up blocks still held in
                # its lookahead queue. Priming first would leave those queued: the
                # loop would then feed the whitening a one-second block while it
                # expected par.len seconds, and it would starve on the second
                # pass. Counted in samples delivered rather than in reads, since
                # a read and a delivered block are neither the same event nor
                # the same size.
                needed = extra_size + int(self.par.resampling * self.par.len)
                buffered = 0
                while buffered < needed:
                    data_ds = self._read_block(streaming, data, ds)
                    buffered += data_ds.GetSize()
                    whitening.Input(data_ds)

            whitening.SetOutputSize(self.par.NoutData, extra_size)


            

            # One search per analysis window length, all reading the same
            # whitened stream: the conditioning is the expensive part and is
            # done once, while each window length has its own stride, its own
            # coefficient grid and its own trigger file.
            searches, writers, search_pars = [], [], []
            for window, overlap, name in search_names(self.schedule,
                                                      getattr(self.par, "SearchName", None)):
                par = Parameters()
                par.copy(self.par)
                par.window, par.overlap, par.Ncoeff = window, overlap, window
                par.SearchName = name
                par.waveletThreshold = wavThresh.name
                search = wdf(par, wavThresh)
                savetrigger = SingleEventPrintTriggers(par)
                parameterestimation = ParameterEstimation(par)
                parameterestimation.register(savetrigger)
                for observer in self._observers(par):
                    parameterestimation.register(observer)
                search.register(parameterestimation)
                par.dump(par.dir + parameters_file_name(par))
                searches.append(search)
                writers.append(savetrigger)
                search_pars.append(par)
            # Start detection loop
            logging.info("Starting detection loop")
            data = SV()
            data_ds = SV()
            dataw = SV()
            # Tested on the block that comes out of conditioning, not on the
            # reader: the two are not at the same time, and testing the reader
            # would end the loop while conditioned data was still queued.
            data_ds = self._read_block(streaming, data, ds)
            searched = None
            while data_ds.GetStart() <= self.par.gpsEnd:
                whitening.Process(data_ds, dataw)
                for search in searches:
                    search.SetData(dataw)
                    search.Process()
                self._on_whitened(dataw)
                searched = dataw.GetStart() + dataw.GetSize() * dataw.GetSampling()
                if data.GetStart() + 2 * self.par.len > gpsEnd:
                    logging.warning(
                        "Stopping at %.1f: the next read would pass the end of "
                        "the segment at %.1f", data_ds.GetStart(), gpsEnd)
                    break
                data_ds = self._read_block(streaming, data, ds)

            # Reading stops a whole priming ahead of what the whitening has
            # emitted, so the filters still hold analysable data when the last
            # read is refused. Draining it costs the segment nothing; leaving it
            # costs a span set by the filters rather than by the segment.
            while (whitening.DataNeeded() <= 0
                   and dataw.GetStart() <= self.par.gpsEnd):
                emitted = dataw.GetStart()
                whitening.Output(dataw)
                if dataw.GetStart() <= emitted:
                    break
                for search in searches:
                    search.SetData(dataw)
                    search.Process()
                self._on_whitened(dataw)
                searched = dataw.GetStart() + dataw.GetSize() * dataw.GetSampling()

            self._finish_segment(dir_chunk)
            for savetrigger in writers:
                savetrigger.close()
            self._record_end(searched, gpsEnd, ds, search_pars)

            elapsed_time = time.time() - start_time
            timeslice = gpsEnd - gpsStart
            logging.info(
                "analyzed %s seconds in %s seconds" % (timeslice, elapsed_time)
            )
            fileEnd = self.par.dir + "ProcessEnded.check"
            open(fileEnd, "a").close()
        else:
            logging.info("Segment already processed")
