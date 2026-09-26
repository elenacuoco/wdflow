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
import hashlib
import json
import time

import numpy as np

from py4tsa.tsa import *
from py4tsa.tsa import WaveletThreshold
from py4tsa.tsa import SeqView_double_t as SV


from wdf.observers.ParameterEstimationObserver import ParameterEstimation 
from wdf.observers.SingleEventPrintFileObserver import SingleEventPrintTriggers

from wdf.processes.BandPassDownSampling import (BandPassDownSampling,
                                                read_conditioned)
from wdf.config.Parameters import Parameters, window_schedule
from wdf.processes.wdf import wdf
from wdf.processes.Whitening import Whitening
from wdf.processes.zero_phase_whitening import (
    DEFAULT_SQRT_ORDER,
    ZeroPhaseWhitening,
)

DEFAULT_AR_ESTIMATION_OFFSET_S = 50.0 
import logging
import os






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
        self.par.Ncoeff = max(window for window, _ in self.schedule)
        self.par.channel = parameters.channel
        self.learn = parameters.learn
        self.par.resampling=parameters.sampling/parameters.ResamplingFactor
        self.par.len=parameters.len
           
    def _fit_start(self, gpsStart, gpsEnd, context_s=0.0):
        """Where in the segment the noise model is fitted.

        `AREstimationOffset` seconds into the segment or, when the segment is
        too short to hold both the offset and `learn` seconds, its last `learn`
        seconds; then moved inward just as far as it takes for `context_s`
        seconds on each side of the stretch to lie inside the segment too.

        :type gpsStart: float
        :param gpsStart: start of the segment.
        :type gpsEnd: float
        :param gpsEnd: end of the segment.
        :type context_s: float
        :param context_s: seconds of real data needed on each side of the
            stretch; none for a stretch that is read and not filtered.
        :return: float -- GPS start of the `learn` seconds the model is fitted
            on.
        :raises ValueError: if the segment is shorter than the stretch and its
            two contexts.
        """
        offset = getattr(self.par, "AREstimationOffset",
                         DEFAULT_AR_ESTIMATION_OFFSET_S)
        if gpsEnd - gpsStart >= self.learn + offset:
            start = gpsStart + offset
        else:
            start = gpsEnd - self.learn
        start = min(max(start, gpsStart + context_s),
                    gpsEnd - self.learn - context_s)
        if start < gpsStart + context_s:
            raise ValueError(
                f"the segment {gpsStart}-{gpsEnd} is shorter than the {self.learn} s "
                f"the noise model is fitted on and the {context_s} s of real data "
                f"its conditioning needs on each side")
        return start

    def _fit_plan(self, gpsStart, gpsEnd):
        """The conditioning the noise model is fitted under, and where.

        :type gpsStart: float
        :param gpsStart: start of the segment.
        :type gpsEnd: float
        :param gpsEnd: end of the segment.
        :return: tuple -- `(front end, context, start)`: the estimation front
            end, the whole seconds of real data it needs on each side of a
            stretch, and the GPS start of the stretch.
        :raises ValueError: if the segment cannot hold the stretch and its
            context.
        """
        ds = BandPassDownSampling(self.par, estimation=True)
        context_s = float(np.ceil(ds.padlen / self.par.sampling))
        return ds, context_s, self._fit_start(gpsStart, gpsEnd, context_s)

    def _model_key(self, gpsStart, gpsEnd):
        """What the noise model depends on, and a name derived from it.

        The model is a function of the samples it is fitted on and of its
        order, and the samples are a function of the channel, the rates, the
        conditioning filter, the numerical type they are handed over in and the
        stretch they are taken from. The name digests all of them, so a model
        fitted under any other conditioning has another name and is never found
        in its place; the description is stored in the model's file.

        :type gpsStart: float
        :param gpsStart: start of the segment.
        :type gpsEnd: float
        :param gpsEnd: end of the segment.
        :return: tuple -- `(name, description)`: twelve hexadecimal characters,
            and the JSON they are the digest of.
        """
        ds, context_s, start = self._fit_plan(gpsStart, gpsEnd)
        sections = np.ascontiguousarray(ds.sos, dtype=np.float64)
        description = json.dumps(dict(
            channel=str(self.par.channel),
            sampling=float(self.par.sampling),
            resampling_factor=int(self.par.ResamplingFactor),
            band=[float(ds.low_freq_hp), float(ds.cutoff_frequency)],
            filter_order=int(ds.order),
            stopband_attenuation_db=float(ds.stopband_attenuation_db),
            sections=hashlib.sha256(sections.tobytes()).hexdigest(),
            samples="float64",
            start=float(start),
            learn=float(self.learn),
            context_s=context_s,
            ar_order=int(self.par.ARorder)), sort_keys=True)
        return hashlib.sha256(description.encode()).hexdigest()[:12], description

    def _learn_stretch(self, gpsStart, gpsEnd):
        """The conditioned stretch a noise model is fitted on.

        `learn` seconds starting at `_fit_start`, the same stretch whichever
        way the filter is then fitted, so that the two are comparable. It is
        read together with the conditioning's settling of real data on each
        side and conditioned by `BandPassDownSampling.condition_stretch`, so
        its edges are filtered as the stream is filtered there. Filtered alone,
        they would carry the filter's start over a settling at each end, and
        the model would be fitted on that as though it were the noise.

        :type gpsStart: float
        :param gpsStart: start of the segment.
        :type gpsEnd: float
        :param gpsEnd: end of the segment.
        :return: py4tsa.tsa.SeqView_double_t -- the conditioned stretch.
        :raises ValueError: if the segment cannot hold the stretch and its
            context.

        Side effects: sets `AREstimationStart`, the GPS start of the stretch,
        on the worker's parameters.
        """
        ds, context_s, start = self._fit_plan(gpsStart, gpsEnd)
        self.par.AREstimationStart = start

        stream = FrameIChannel(self.par.file, self.par.channel,
                               self.learn + 2 * context_s, start - context_s)
        raw = SV()
        stream.GetData(raw)
        return ds.condition_stretch(raw, int(round(context_s * self.par.sampling)))

    def _noise_model(self, gpsStart, gpsEnd, dir_chunk):
        """Fit the segment's noise model, or load it, and say how to whiten with it.

        The autoregressive model is fitted on the stretch `_learn_stretch`
        conditions and saved beside the segment's triggers; a model already
        saved there is loaded instead. `WhiteningModel` then selects where the
        whitening filter's coefficients come from, "burg" (the autoregressive
        model and its square root) or "spectrum" (the measured spectrum of the
        same stretch).

        What is returned builds the whitening rather than being it: the filter
        carries state, and each pass over the segment needs its own, identical,
        filter.

        :type gpsStart: float
        :param gpsStart: start of the segment.
        :type gpsEnd: float
        :param gpsEnd: end of the segment.
        :type dir_chunk: str
        :param dir_chunk: the segment's output directory, where the model is
            saved.
        :return: callable -- takes no argument and returns a fresh
            `ZeroPhaseWhitening`.
        :raises ValueError: if `WhiteningModel` is neither "burg" nor
            "spectrum".

        Side effects: sets `ARkey`, `ARfile`, `LVfile`, `AREstimationStart`,
        `sigma`, `sigmaWhitened` and `SqrtWhiteningOrder` on the worker's
        parameters, and writes the model files when it fits them.
        """
        whiten = Whitening(self.par.ARorder)
        # HDF5 (wdf.processes.ar_lv_io), named by everything the model depends
        # on, so that a model saved under another conditioning is not loaded.
        key, conditioning = self._model_key(gpsStart, gpsEnd)
        self.par.ARkey = key
        self.par.ARfile = dir_chunk + "ARcoeff-AR%s-fs%s-%s-%s.h5" % (
            self.par.ARorder,
            self.par.resampling,
            self.par.channel,
            key,
        )
        self.par.LVfile = dir_chunk + "LVcoeff-AR%s-fs%s-%s-%s.h5" % (
            self.par.ARorder,
            self.par.resampling,
            self.par.channel,
            key,
        )

        learn = None
        if os.path.isfile(self.par.ARfile) and os.path.isfile(self.par.LVfile):
            logging.info("Load AR parameters")
            whiten.ParametersLoad(self.par.ARfile, self.par.LVfile)
            self.par.AREstimationStart = json.loads(conditioning)["start"]
        else:
            logging.info("Start AR parameter estimation")
            learn = self._learn_stretch(gpsStart, gpsEnd)
            whiten.ParametersEstimate(learn)
            whiten.ParametersSave(self.par.ARfile, self.par.LVfile, conditioning)

        # sigma for the noise
        self.par.sigma = whiten.GetSigma()
        logging.info("Estimated sigma= %s" % self.par.sigma)

        # Coefficients of the square-root model the zero-phase whitening runs
        # in both directions (see wdf.processes.zero_phase_whitening). Unset
        # means the model's own order: a lower one costs accuracy twice over,
        # since the response is the square of the filter's magnitude.
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

        output_size = int(self.par.resampling)
        if model == "spectrum":
            if learn is None:
                learn = self._learn_stretch(gpsStart, gpsEnd)
            samples = np.array([learn.GetY(0, i) for i in range(learn.GetSize())])
            band = (self.par.LowFrequencyCut, 0.5 * self.par.resampling)

            def build():
                return ZeroPhaseWhitening.from_spectrum(
                    samples, self.par.resampling, output_size, 0,
                    order=sqrt_order, band=band)
        else:
            def build():
                return ZeroPhaseWhitening(ar, output_size, 0, order=sqrt_order)

        whitening = build()
        if model == "spectrum":
            # The scale the search thresholds on is the scale of the stream it
            # is given, and that stream is this filter's output.
            self.par.sigma = whitening.sigma
        self.par.sigmaWhitened = whitening.sigma
        logging.info("Whitening model: %s" % model)
        logging.info("Zero-phase whitening, square-root order %s, "
                     "latency %s samples" % (sqrt_order, whitening.latency))
        return build

    def _prime(self, gpsStart, gpsEnd, ds, whitening):
        """Open the segment and fill the chain up to its first searched block.

        The first stretch of the segment is read, conditioned and whitened
        without being searched, so that both filters have settled, and the
        whitening's lookahead is then loaded ahead of the first block it emits.

        :type gpsStart: float
        :param gpsStart: start of the segment.
        :type gpsEnd: float
        :param gpsEnd: end of the segment.
        :type ds: BandPassDownSampling
        :param ds: a conditioning front end that has read nothing yet.
        :type whitening: ZeroPhaseWhitening
        :param whitening: a whitening that has filtered nothing yet.
        :return: py4tsa.tsa.FrameIChannel -- the reader, positioned after what
            has been read, delivering `len` seconds per read.

        Side effects: sets `WhiteningExtraSize`, `gpsEnd` and `NoutData` on the
        worker's parameters.
        """
        data = SV()
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
        # The first searched sample has to be settled in both filters. The
        # conditioning starts at the segment's first sample with no past and
        # settles over `padlen` samples; the whitening's forward pass is FIR of
        # its order, so it forgets its own start after that many samples of
        # settled input. The warm-up is therefore at least the sum of the two,
        # in reads of one second, whatever `preWhite` asks for, and the value
        # used is what is recorded.
        warm_up_s = (ds.padlen / self.par.sampling
                     + whitening.latency / self.par.resampling)
        self.par.preWhite = max(int(self.par.preWhite), int(np.ceil(warm_up_s)))
        # reading data, downsampling and whitening
        for i in range(self.par.preWhite):
            data_ds = read_conditioned(streaming, data, ds)
            whitening.Process(data_ds,dataw)

        # Fixed, len-independent lookahead window for whitening.
        # DoubleWhitening's backward pass needs a buffer of real *future*
        # data to settle its lattice-filter state before it can produce a
        # good backward-pass estimate for the current output chunk (see
        # DoubleWhitening::GetData in p4TSA). That lookahead ("ExtraSize")
        # is a FIXED size, decoupled from par.len (an I/O batching/perf
        # knob), mirroring BandPassDownSampling's own padlen convention.
        # Set parameters.WhiteningExtraSize explicitly to override it, or
        # to 0 to make the lookahead scale with par.len instead (legacy
        # behaviour).
        # The default is the filter's own order, which is exactly what the
        # backward pass reads ahead: the filter is the prediction error of
        # the model and is therefore FIR, so after `order` steps the
        # initialisation is forgotten identically rather than
        # asymptotically, and a longer lookahead buys nothing. It costs,
        # though, because the pass is re-run over the lookahead for every
        # output block: measured on O4b with an order of 3000, whitening
        # 60 s took 35 s with a lookahead of 20 s and blocks of 1 s, and
        # 2.3 s with a lookahead of `order` and blocks of 4 s.
        extra_size = int(getattr(self.par, "WhiteningExtraSize",
                                 self.par.SqrtWhiteningOrder))
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
                data_ds = read_conditioned(streaming, data, ds)
                buffered += data_ds.GetSize()
                whitening.Input(data_ds)

        whitening.SetOutputSize(self.par.NoutData, extra_size)
        return streaming

    def _whitened(self, streaming, gpsEnd, ds, whitening):
        """The segment's whitened stream, one block at a time.

        Continues from where `_prime` left the chain and yields every block
        the search reads, in order, up to the last one the frames can supply
        with its lookahead.

        :type streaming: py4tsa.tsa.FrameIChannel
        :param streaming: the reader `_prime` returned.
        :type gpsEnd: float
        :param gpsEnd: end of the segment.
        :type ds: BandPassDownSampling
        :param ds: the conditioning front end `_prime` was given.
        :type whitening: ZeroPhaseWhitening
        :param whitening: the whitening `_prime` was given.
        :return: generator of py4tsa.tsa.SeqView_double_t -- the whitened
            blocks. The same view is refilled for each block, so a consumer
            that keeps a block copies it before asking for the next.
        """
        data = SV()
        dataw = SV()
        # Tested on the block that comes out of conditioning, not on the
        # reader: the two are not at the same time, and testing the reader
        # would end the loop while conditioned data was still queued.
        data_ds = read_conditioned(streaming, data, ds)
        while data_ds.GetStart() <= self.par.gpsEnd:
            whitening.Process(data_ds, dataw)
            yield dataw
            if data.GetStart() + 2 * self.par.len > gpsEnd:
                logging.warning(
                    "Stopping at %.1f: the next read would pass the end of "
                    "the segment at %.1f", data_ds.GetStart(), gpsEnd)
                break
            data_ds = read_conditioned(streaming, data, ds)

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
            yield dataw

    def segmentProcess(self, segment, wavThresh=WaveletThreshold.block):
        """Runs the full offline WDF pipeline over one contiguous GPS segment:
        estimate (or load cached) AR-whitening parameters from a `learn`-second
        warm-up read, then stream the rest of the segment through
        downsampling -> zero-phase whitening -> WDF trigger search, writing triggers to
        `<outdir>/<run>/<itf>/<channel>_<gpsStart>/` as they're found.

        :type segment: tuple[float, float]
        :param segment: (gpsStart, gpsEnd) bounds of the segment to analyze.
        :type wavThresh: py4tsa.tsa.WaveletThreshold.WaveletThresholding
        :param wavThresh: the rule for the coefficients of a window, passed to WDF's C++
            engine. The default `block` judges contiguous coefficients of one level
            together, so that a signal spread over neighbouring coefficients, each below
            the universal threshold, survives as a block; `dohonojohnston` is that universal
            threshold on each coefficient alone. The rule is recorded beside the triggers
            as `waveletThreshold`.
        :return: None -- triggers are written to disk (Parquet, or CSV for older runs),
            not returned; a `ProcessEnded.check` marker file in the segment's output
            directory means a prior run already completed it and this call is a no-op.

        AR parameters are estimated from `Parameters.learn` seconds of data taken
        `Parameters.AREstimationOffset` seconds after the segment start (default
        `DEFAULT_AR_ESTIMATION_OFFSET_S`). The offset skips the beginning of a
        segment, where noise following lock acquisition can still be settling and
        would bias the noise model; set it to 0 for data known to be in science
        mode throughout. When the segment is too short to hold both the offset and
        the estimation window, the window is taken from the segment end instead.
        Either way the window is conditioned with the settling of real data on
        each side, and is moved inward as far as that requires (`_fit_start`).
        """
        gpsStart, gpsEnd = segment[0],segment[1]
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
            build_whitening = self._noise_model(gpsStart, gpsEnd, dir_chunk)

            # update the self.parameters to be saved in local json file
            self.par.ID = ID
            self.par.dir = dir_chunk
            self.par.gps = gpsStart
            self.par.gpsStart = gpsStart

            ds = BandPassDownSampling(self.par)
            whitening = build_whitening()
            streaming = self._prime(gpsStart, gpsEnd, ds, whitening)

            # One search per analysis window length, all reading the same
            # whitened stream: the conditioning is the expensive part and is
            # done once, while each window length has its own stride, its own
            # coefficient grid and its own trigger file.
            searches, writers = [], []
            for window, overlap in self.schedule:
                par = Parameters()
                par.copy(self.par)
                par.window, par.overlap, par.Ncoeff = window, overlap, window
                par.waveletThreshold = wavThresh.name
                search = wdf(par, wavThresh)
                savetrigger = SingleEventPrintTriggers(par)
                parameterestimation = ParameterEstimation(par)
                parameterestimation.register(savetrigger)
                search.register(parameterestimation)
                par.dump("%sparametersUsed-Win%s.json" % (par.dir, window))
                searches.append(search)
                writers.append(savetrigger)
            # Start detection loop
            logging.info("Starting detection loop")
            for dataw in self._whitened(streaming, gpsEnd, ds, whitening):
                for search in searches:
                    search.SetData(dataw)
                    search.Process()

            for savetrigger in writers:
                savetrigger.close()

            elapsed_time = time.time() - start_time
            timeslice = gpsEnd - gpsStart
            logging.info(
                "analyzed %s seconds in %s seconds" % (timeslice, elapsed_time)
            )
            fileEnd = self.par.dir + "ProcessEnded.check"
            open(fileEnd, "a").close()
        else:
            logging.info("Segment already processed")
