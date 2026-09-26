"""One segment, from frames to trigger files, with the conditioning downsampled.

The unit of work a run is divided into, for one detector. It finds the
segment's lines and fits its noise model, then reads the segment block by block
and conditions it --- notch, band-pass, downsample, whiten --- twice: once in
full, to find the transients to gate and to check that the stream is fit to be
searched, and once to search it, gated, at the configured analysis windows,
writing the triggers of that segment. The two passes run the same front end and
the same filter, so the stream checked is the stream searched; the first holds
the whitened segment in memory, the second streams it.

The search pass is shared by every analysis window, so the conditioning is done
once per block of that pass however the search is configured.
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
                                                SV_to_array,
                                                read_conditioned)
from wdf.processes.gating import gate_weights, merged, octave_bands, transients
from wdf.processes.lines import median_spectrum, spectral_lines
from wdf.processes import validation
from wdf.processes.validation import ConditioningRejected, bns_range
from wdf.config.Parameters import Parameters, window_schedule
from wdf.processes.wdf import wdf
from wdf.processes.Whitening import Whitening
from wdf.processes.zero_phase_whitening import (
    DEFAULT_SQRT_ORDER,
    ZeroPhaseWhitening,
)

DEFAULT_AR_ESTIMATION_OFFSET_S = 50.0 
#: Height above the local floor, as a ratio of amplitude spectral densities,
#: from which a line is notched when the configuration names none.
DEFAULT_LINE_THRESHOLD = 5.0
#: Height, in robust standard deviations of the whitened stream in any octave
#: the search reads, from which a transient is gated rather than searched.
DEFAULT_GATE_THRESHOLD = 50.0
#: Seconds over which a gate takes the whitened stream to zero on each side.
DEFAULT_GATE_TAPER_S = 0.25
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
        # Lines named in the configuration are notched in every segment; when
        # it names none, each segment's are found on its own fit stretch.
        self.configured_lines = getattr(parameters, "LineNotches", None)

    def _raw_stretch(self, start, seconds):
        """The strain of a stretch as the frames hold it, unconditioned.

        :type start: float
        :param start: GPS start of the stretch.
        :type seconds: float
        :param seconds: its length.
        :return: numpy.ndarray -- the samples, at the frames' rate.
        """
        stream = FrameIChannel(self.par.file, self.par.channel, seconds, start)
        view = SV()
        stream.GetData(view)
        return SV_to_array(view)

    def _fit_spectrum(self, gpsStart, gpsEnd):
        """The spectrum of the strain on the stretch the offset names for the fit.

        The median of the periodograms of `learn` seconds of unconditioned
        strain, at the frames' rate (`wdf.processes.lines.median_spectrum`).

        :type gpsStart: float
        :param gpsStart: start of the segment.
        :type gpsEnd: float
        :param gpsEnd: end of the segment.
        :return: tuple -- `(frequency, psd)`, Hz and strain squared per Hz.
        """
        start = self._fit_start(gpsStart, gpsEnd)
        return median_spectrum(self._raw_stretch(start, self.learn),
                               self.par.sampling)

    def _segment_lines(self, gpsStart, gpsEnd):
        """The lines a segment is notched at.

        Those the configuration names, when it names any. Otherwise every line
        standing `LineThreshold` times above the local floor of the fit
        stretch's spectrum, anywhere its content can reach the analysed stream
        (`BandPassDownSampling.line_band`). The stretch is the one the offset
        names, read without conditioning; the model is fitted on the same
        stretch, moved inward by at most the conditioning's settling when the
        segment's edges require it. A `LineThreshold` of zero or None notches
        nothing.

        :type gpsStart: float
        :param gpsStart: start of the segment.
        :type gpsEnd: float
        :param gpsEnd: end of the segment.
        :return: list -- `[frequency, bandwidth, height]` per line, as
            `wdf.processes.lines.spectral_lines` returns them.
        """
        if self.configured_lines is not None:
            return np.asarray(self.configured_lines, dtype=float).reshape(-1, 3).tolist()
        threshold = getattr(self.par, "LineThreshold", DEFAULT_LINE_THRESHOLD)
        if not threshold:
            return []
        self.par.LineNotches = None
        low, high = BandPassDownSampling(self.par).line_band
        frequency, psd = self._fit_spectrum(gpsStart, gpsEnd)
        return spectral_lines(frequency, psd, low, high,
                              threshold=float(threshold)).tolist()
           
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

    def _whitened_segment(self, gpsStart, gpsEnd, build_whitening):
        """The whitened stream the search reads, as one array, before searching it.

        The segment is conditioned and whitened exactly as the search pass
        conditions and whitens it -- the same front end, a fresh filter from the
        same model, the same warm-up and the same blocks -- and the blocks are
        joined in time order.

        :type gpsStart: float
        :param gpsStart: start of the segment.
        :type gpsEnd: float
        :param gpsEnd: end of the segment.
        :type build_whitening: callable
        :param build_whitening: what `_noise_model` returned.
        :return: tuple -- `(start, samples, front end)`: the GPS time of the
            first sample, the samples at the analysed rate, and the conditioning
            front end that produced them.
        :raises RuntimeError: if the segment yields no block, or if the blocks
            do not follow one another without a gap, since a stream placed from
            its first sample would then be misplaced after the gap.
        """
        ds = BandPassDownSampling(self.par)
        whitening = build_whitening()
        streaming = self._prime(gpsStart, gpsEnd, ds, whitening)
        starts, blocks = [], []
        for dataw in self._whitened(streaming, gpsEnd, ds, whitening):
            starts.append(dataw.GetStart())
            blocks.append(SV_to_array(dataw))
        if not blocks:
            raise RuntimeError(
                f"the segment {gpsStart}-{gpsEnd} holds no block to search once "
                f"the warm-up and the read-ahead are taken from it")
        lengths = np.array([len(b) for b in blocks], dtype=float)
        expected = starts[0] + np.concatenate([[0.0], np.cumsum(lengths[:-1])]) / self.par.resampling
        if np.max(np.abs(np.array(starts) - expected)) > 0.5 / self.par.resampling:
            raise RuntimeError("the whitened blocks of the segment do not follow one another")
        return starts[0], np.concatenate(blocks), ds

    def _gates(self, start, found):
        """The stretches of the whitened stream the search is not given.

        Those the configuration declares (`Gates`, GPS `[start, stop]` pairs),
        and every transient of the census whose height reaches `GateThreshold`
        robust standard deviations, broadband or in an octave the search reads;
        a threshold of zero or None gates only what is declared.

        :type start: float
        :param start: GPS time of the stream's first sample.
        :type found: wdf.processes.gating.Transients or None
        :param found: the census of the whitened stream; None when none was
            taken.
        :return: numpy.ndarray -- shape `(n, 2)`, GPS start and stop of each
            zeroed stretch, sorted and disjoint.
        """
        declared = getattr(self.par, "Gates", None)
        gates = np.asarray([] if declared is None else declared, dtype=float).reshape(-1, 2)
        threshold = getattr(self.par, "GateThreshold", DEFAULT_GATE_THRESHOLD)
        if threshold and found is not None:
            rate = float(self.par.resampling)
            loud = found.peak >= float(threshold)
            gates = np.vstack([gates, np.column_stack([start + found.start[loud] / rate,
                                                       start + found.stop[loud] / rate])])
        return merged(gates)

    def _examine(self, gpsStart, gpsEnd, build_whitening, check):
        """Whiten the whole segment once, find its gates and, if asked, check it.

        The census of the whitened stream (`wdf.processes.gating.transients`,
        over the octaves from the detector's search low frequency) gives the
        gates. The check (`wdf.processes.validation.validate`) is read on the
        stream as the search will read it: gated, and divided by the scale the
        search divides by, with the detector's binary neutron star range from
        the spectrum of its fit stretch.

        :type gpsStart: float
        :param gpsStart: start of the segment.
        :type gpsEnd: float
        :param gpsEnd: end of the segment.
        :type build_whitening: callable
        :param build_whitening: what `_noise_model` returned.
        :type check: bool
        :param check: whether to check the stream as well as gate it.
        :return: tuple -- `(gates, report)`: as `_gates` returns them, and the
            `ValidationReport`, or None when not checked.
        """
        start, whitened, ds = self._whitened_segment(gpsStart, gpsEnd, build_whitening)
        range_mpc = bns_range(*self._fit_spectrum(gpsStart, gpsEnd)) if check else None
        return self._examine_stream(start, whitened, ds, check, range_mpc)

    def _examine_stream(self, start, whitened, ds, check, range_mpc=float("nan")):
        """Find the gates of a whitened stream and, if asked, check it.

        The census, the gates and the check of `_examine`, on a stream already
        in hand: the census's robust scale, the gates and every criterion are
        read on this stream alone.

        :type start: float
        :param start: GPS time of the stream's first sample.
        :type whitened: numpy.ndarray
        :param whitened: the whitened stream, ungated, at the analysed rate.
        :type ds: BandPassDownSampling
        :param ds: the conditioning front end that produced it.
        :type check: bool
        :param check: whether to check the stream as well as gate it.
        :type range_mpc: float
        :param range_mpc: the detector's binary neutron star range, reported.
        :return: tuple -- `(gates, report)`, as `_examine` returns them.
        """
        rate = float(self.par.resampling)
        bands = octave_bands(rate, ds.search_low_frequency)
        found = transients(whitened, rate, bands)
        gates = self._gates(start, found)
        if not check:
            return gates, None
        taper = float(getattr(self.par, "GateTaper", DEFAULT_GATE_TAPER_S))
        times = start + np.arange(whitened.size) / rate
        gated = whitened * gate_weights(times, gates, taper) / float(self.par.sigma)
        window = max(window for window, _ in self.schedule) / rate
        report = validation.validate(self.par.itf, gated, rate, start, bands,
                                     (ds.cutoff_frequency, 0.5 * rate), found, gates,
                                     taper, window, range_mpc=range_mpc)
        logging.info("Conditioning check:\n%s" % report.table())
        return gates, report

    def _segment_directory(self, gpsStart):
        """Where a segment's model, check and triggers are written.

        :type gpsStart: float
        :param gpsStart: start of the segment.
        :return: tuple -- `(ID, directory)`, the directory with a trailing
            separator, created if missing.
        """
        ID = "".join([str(self.par.channel), "_", str(int(gpsStart))])
        dir_chunk = "".join([self.par.outdir, self.par.run, "/", self.par.itf, "/", ID, '/'])
        if not os.path.exists(dir_chunk):
            os.makedirs(dir_chunk)
        return ID, dir_chunk

    def validate(self, segment):
        """Condition and whiten a segment as the search would, and check it.

        Everything `segmentProcess` does before its first search: the lines,
        the noise model, the whitened stream, its gates and its check; nothing
        is searched. The report is written beside the segment's triggers as
        `conditioning-check.json`.

        :type segment: tuple[float, float]
        :param segment: `(gpsStart, gpsEnd)` of the segment.
        :return: wdf.processes.validation.ValidationReport
        """
        gpsStart, gpsEnd = segment[0], segment[1]
        _, dir_chunk = self._segment_directory(gpsStart)
        self.par.LineNotches = self._segment_lines(gpsStart, gpsEnd)
        build_whitening = self._noise_model(gpsStart, gpsEnd, dir_chunk)
        _, report = self._examine(gpsStart, gpsEnd, build_whitening, True)
        self._record(report, dir_chunk)
        return report

    def validate_starts(self, segment, starts, stop_at_first=True):
        """Check a segment as though it began at each of several later instants.

        A segment searched from a later start keeps its own fit stretch, lines
        and model, so the stream its search reads is the whole segment's stream
        from that start's warm-up on: both filters settle within the warm-up and
        forget where they were started. The segment is therefore whitened once,
        and each start is checked on the tail of that stream a segment
        beginning there would search, from the start plus the warm-up to the
        segment's end, with the census, the gates and every criterion read on
        that tail alone -- the check `validate` makes of the shorter segment,
        without whitening it again. Nothing is searched and nothing is written.

        :type segment: tuple[float, float]
        :param segment: `(gpsStart, gpsEnd)` of the segment; its fit stretch is
            where `AREstimationOffset` puts it, and a later start keeps it there.
        :type starts: sequence of float
        :param starts: GPS times the segment is checked from, in the order they
            are tried; each must leave the fit stretch, with its settling,
            inside the segment it begins.
        :type stop_at_first: bool
        :param stop_at_first: stop at the first start whose tail passes.
        :return: list -- one `(start, ValidationReport)` per start tried, in
            order; the report is None where the tail holds nothing to check.
        :raises ValueError: if a start would leave the fit stretch outside the
            segment it begins.
        """
        gpsStart, gpsEnd = float(segment[0]), float(segment[1])
        # The lines are read on the stretch the offset names and the model is
        # fitted on it with its settling of real data on each side; a later
        # start keeps both only if it leaves the two inside the segment it
        # begins, the second with its settling.
        _, context_s, model_start = self._fit_plan(gpsStart, gpsEnd)
        latest = min(self._fit_start(gpsStart, gpsEnd), model_start - context_s)
        for later in starts:
            if not gpsStart <= float(later) <= latest:
                raise ValueError(f"a start at {later} leaves the fit stretch at "
                                 f"{model_start} outside the segment it begins")
        _, dir_chunk = self._segment_directory(gpsStart)
        self.par.LineNotches = self._segment_lines(gpsStart, gpsEnd)
        build_whitening = self._noise_model(gpsStart, gpsEnd, dir_chunk)
        start, whitened, ds = self._whitened_segment(gpsStart, gpsEnd, build_whitening)
        rate = float(self.par.resampling)
        range_mpc = bns_range(*self._fit_spectrum(gpsStart, gpsEnd))
        # `_whitened_segment` primed the chain, which set the warm-up every
        # segment of this detector takes before its first searched block.
        warm_up = float(self.par.preWhite)
        times = start + np.arange(whitened.size) / rate
        out = []
        for later in starts:
            tail = times >= float(later) + warm_up - 0.5 / rate
            if not tail.any():
                out.append((float(later), None))
                continue
            first = int(np.argmax(tail))
            _, report = self._examine_stream(times[first], whitened[first:], ds, True,
                                             range_mpc)
            out.append((float(later), report))
            if stop_at_first and report.passed:
                break
        return out

    @staticmethod
    def _record(report, dir_chunk):
        """Write a check's report beside the segment's triggers."""
        with open(dir_chunk + "conditioning-check.json", "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=1)

    def whitened_stretch(self, segment, start, stop, gates=None):
        """The whitened stream of a segment between two instants, as the search reads it.

        The segment's lines and noise model are found and fitted as
        `segmentProcess` finds and fits them --- a model already saved beside
        the segment is loaded --- and the strain is conditioned and whitened
        from a whole number of seconds before `start`, far enough for both
        filters to have settled there. The band-pass is applied with real data
        on both sides of every block and the whitening is a finite filter run
        forward and then backward over a look-ahead of its own order, so once
        they have settled each sample is what the pass over the whole segment
        gives at that instant, whatever block it falls in, to the rounding the
        whitening's arithmetic leaves at a block join; starting on a whole
        second keeps the decimation on the segment's phase. The stream is then
        gated as the search gates it and divided by the scale the search
        divides it by, which is the stream `validate` checks.

        :type segment: tuple[float, float]
        :param segment: `(gpsStart, gpsEnd)` of the segment.
        :type start: float
        :param start: GPS time of the first sample wanted.
        :type stop: float
        :param stop: GPS time of the last sample wanted.
        :type gates: numpy.ndarray or None
        :param gates: shape `(n, 2)`, GPS start and stop of each gated stretch
            of the segment, as its check reports them
            (`ValidationReport.gates`); None gates nothing.
        :return: tuple -- `(t0, samples)`: the GPS time of the first sample and
            the samples at the analysed rate, on the search's noise scale.
        :raises ValueError: if the stretch asked for is not inside what the
            segment's search reads, once its warm-up and its read-ahead are
            taken from it.
        :raises RuntimeError: if the whitened blocks do not follow one another
            without a gap.

        Side effects: as `validate`, the lines and the model are set on the
        worker's parameters and a model fitted here is saved beside the
        segment's triggers.
        """
        gpsStart, gpsEnd = float(segment[0]), float(segment[1])
        start, stop = float(start), float(stop)
        if not gpsStart <= start < stop <= gpsEnd:
            raise ValueError(f"{start}-{stop} is not inside the segment {gpsStart}-{gpsEnd}")
        _, dir_chunk = self._segment_directory(gpsStart)
        self.par.LineNotches = self._segment_lines(gpsStart, gpsEnd)
        build_whitening = self._noise_model(gpsStart, gpsEnd, dir_chunk)
        ds = BandPassDownSampling(self.par)
        whitening = build_whitening()
        rate = float(self.par.resampling)
        # The warm-up `_prime` will take, so that its first emitted sample
        # falls at or before `start`; a second more for the reader's rounding.
        warm_up = max(int(self.par.preWhite),
                      int(np.ceil(ds.padlen / self.par.sampling + whitening.latency / rate)))
        first = gpsStart + max(0.0, np.floor(start - warm_up - 1.0 - gpsStart))
        streaming = self._prime(first, gpsEnd, ds, whitening)
        starts, blocks = [], []
        for dataw in self._whitened(streaming, gpsEnd, ds, whitening):
            if dataw.GetStart() > stop:
                break
            starts.append(dataw.GetStart())
            blocks.append(SV_to_array(dataw))
        if not blocks:
            raise ValueError(f"the segment {gpsStart}-{gpsEnd} emits no block before {stop}")
        lengths = np.array([len(b) for b in blocks], dtype=float)
        expected = starts[0] + np.concatenate([[0.0], np.cumsum(lengths[:-1])]) / rate
        if np.max(np.abs(np.array(starts) - expected)) > 0.5 / rate:
            raise RuntimeError("the whitened blocks of the stretch do not follow one another")
        samples = np.concatenate(blocks)
        times = starts[0] + np.arange(samples.size) / rate
        keep = (times >= start - 0.5 / rate) & (times <= stop + 0.5 / rate)
        if not keep.any() or times[keep][0] > start + 0.5 / rate \
                or times[keep][-1] < stop - 1.5 / rate:
            raise ValueError(
                f"the search of {gpsStart}-{gpsEnd} reads {times[0]:.3f}-{times[-1]:.3f}, "
                f"which does not hold {start}-{stop}")
        taper = float(getattr(self.par, "GateTaper", DEFAULT_GATE_TAPER_S))
        declared = np.zeros((0, 2)) if gates is None else np.asarray(gates, dtype=float)
        weights = gate_weights(times[keep], declared.reshape(-1, 2), taper)
        return float(times[keep][0]), samples[keep] * weights / float(self.par.sigma)

    def _apply_gates(self, view, gates):
        """Multiply a whitened block by the gates' weights, in place.

        Only the samples a gate or its taper reaches are rewritten, so a block
        no gate touches is left exactly as it is.

        :type view: py4tsa.tsa.SeqView_double_t
        :param view: the whitened block.
        :type gates: numpy.ndarray
        :param gates: as `_gates` returns them.
        :return: None
        """
        if gates.shape[0] == 0:
            return
        taper = float(getattr(self.par, "GateTaper", DEFAULT_GATE_TAPER_S))
        times = view.GetStart() + np.arange(view.GetSize()) / float(self.par.resampling)
        weights = gate_weights(times, gates, taper)
        for i in np.flatnonzero(weights < 1.0):
            view.FillPoint(0, int(i), view.GetY(0, int(i)) * float(weights[i]))

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

        The strain is notched at the segment's lines before the band-pass
        (`_segment_lines`: those `LineNotches` names, or those standing
        `LineThreshold` times above the floor of the estimation window). Before
        any of the segment is searched it is conditioned and whitened once in
        full, and every transient of that whitened stream reaching
        `GateThreshold` robust standard deviations, broadband or in an octave
        the search reads, is gated together with the stretches `Gates`
        declares (`_gates`): the search pass zeroes them, with a raised-cosine
        taper of `GateTaper` seconds on each side. The lines and the gates used
        are recorded with the run's parameters, as `LineNotches` and
        `GatesApplied`.

        The same whitened stream, gated, is checked before the search starts
        (`wdf.processes.validation`): in every octave the search reads it must
        be white, Gaussian and stationary, and transients and gates must cover
        little of it. The report is written beside the triggers as
        `conditioning-check.json`; `ValidateConditioning = False` skips the
        check.

        :raises wdf.processes.validation.ConditioningRejected: if the check
            fails, before anything is searched; its message names the
            detector, the band and the criterion.
        """
        gpsStart, gpsEnd = segment[0],segment[1]
        logging.info(
            "Analyzing segment: %s-%s for channel %s downsampled at %dHz"
            % (gpsStart, gpsEnd, self.par.channel, self.par.resampling)
        )
        start_time = time.time()
        ID, dir_chunk = self._segment_directory(gpsStart)
        if not os.path.isfile(dir_chunk + "ProcessEnded.check"):
            self.par.LineNotches = self._segment_lines(gpsStart, gpsEnd)
            logging.info("Notching %d lines" % len(self.par.LineNotches))
            build_whitening = self._noise_model(gpsStart, gpsEnd, dir_chunk)

            # update the self.parameters to be saved in local json file
            self.par.ID = ID
            self.par.dir = dir_chunk
            self.par.gps = gpsStart
            self.par.gpsStart = gpsStart

            # The gates are found, and the stream is checked, on the whole
            # whitened segment before any of it is searched: a transient's
            # extent, the noise it is measured against and the stationarity of
            # the stretch are only known from the stream around them. A stream
            # that fails the check is not searched.
            check = bool(getattr(self.par, "ValidateConditioning", True))
            if check or getattr(self.par, "GateThreshold", DEFAULT_GATE_THRESHOLD):
                gates, report = self._examine(gpsStart, gpsEnd, build_whitening, check)
            else:
                gates, report = self._gates(0.0, None), None
            self.par.GatesApplied = gates.tolist()
            logging.info("Gating %d stretches of the whitened stream" % len(gates))
            if report is not None:
                self._record(report, dir_chunk)
                if not report.passed:
                    raise ConditioningRejected(report)

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
                self._apply_gates(dataw, gates)
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
