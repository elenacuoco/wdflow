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

from pytsa.tsa import *
from pytsa.tsa import WaveletThreshold
from pytsa.tsa import SeqView_double_t as SV


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
           
    def segmentProcess(self, segment, wavThresh=WaveletThreshold.dohonojohnston):
        """Runs the full offline WDF pipeline over one contiguous GPS segment:
        estimate (or load cached) AR-whitening parameters from a `learn`-second
        warm-up read, then stream the rest of the segment through
        downsampling -> zero-phase whitening -> WDF trigger search, writing triggers to
        `<outdir>/<run>/<itf>/<channel>_<gpsStart>/` as they're found.

        :type segment: tuple[float, float]
        :param segment: (gpsStart, gpsEnd) bounds of the segment to analyze.
        :type wavThresh: pytsa.tsa.WaveletThreshold.WaveletThresholding
        :param wavThresh: wavelet-coefficient thresholding rule passed to WDF's C++
            engine (default `dohonojohnston`, the Donoho-Johnstone universal threshold).
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
                offset = getattr(self.par, "AREstimationOffset",
                                 DEFAULT_AR_ESTIMATION_OFFSET_S)
                if gpsEnd - gpsStart >= self.learn + offset:
                    gpsE = gpsStart + offset
                else:
                    gpsE = gpsEnd - self.learn
                
                strLearn = FrameIChannel(
                    self.par.file, self.par.channel, self.learn, gpsE) 
                Learn = SV()
                Learn_DS = SV()
                ds = BandPassDownSampling(self.par,estimation=True)
                strLearn.GetData(Learn)
                Learn_DS=ds.Process(Learn)
                whiten.ParametersEstimate(Learn_DS)
                whiten.ParametersSave(self.par.ARfile, self.par.LVfile)
                
                del Learn, ds, strLearn, Learn_DS
                
            # sigma for the noise
            self.par.sigma = whiten.GetSigma()
            logging.info("Estimated sigma= %s" % self.par.sigma)

            # Coefficients of the square-root model the zero-phase whitening
            # runs in both directions (see wdf.processes.zero_phase_whitening).
            sqrt_order = int(getattr(self.par, "SqrtWhiteningOrder",
                                     DEFAULT_SQRT_ORDER))
            self.par.SqrtWhiteningOrder = sqrt_order
            ar = np.array([whiten.ADE.GetAR(j)
                           for j in range(self.par.ARorder + 1)])
            
            # update the self.parameters to be saved in local json file
            self.par.ID = ID
            self.par.dir = dir_chunk
            self.par.gps = gpsStart
            self.par.gpsStart = gpsStart
            

            ######################
            # self.parameter for sequence of data and the resampling
        
            ds = BandPassDownSampling(self.par)        
            
            #Perform operation to intialite the detection loop    
            #gpsStart = gpsStart - self.par.preWhite            
            data = SV()
            data_ds = SV()
            dataw = SV()
            Noutdata = int(self.par.resampling)
            whitening = ZeroPhaseWhitening(ar, Noutdata, 0, order=sqrt_order)
            self.par.sigmaWhitened = whitening.sigma
            logging.info("Zero-phase whitening, square-root order %s, "
                         "latency %s samples" % (sqrt_order, whitening.latency))
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
                data_ds = read_conditioned(streaming, data, ds)
                whitening.Process(data_ds,dataw)
               
                
            # Fixed, len-independent lookahead window for whitening.
            # DoubleWhitening's backward pass needs a buffer of real *future*
            # data to settle its lattice-filter state before it can produce a
            # good backward-pass estimate for the current output chunk (see
            # DoubleWhitening::GetData in p4TSA). That lookahead ("ExtraSize")
            # is a FIXED size, decoupled from par.len (an I/O batching/perf
            # knob), mirroring BandPassDownSampling's own padlen convention.
            # Default: 20 seconds of resampled-rate data, large enough for
            # AR orders up to a few thousand to settle. Set
            # parameters.WhiteningExtraSize explicitly to override, or to 0
            # to make the lookahead scale with par.len instead (legacy
            # behavior).
            extra_size = int(getattr(self.par, "WhiteningExtraSize", 20 * self.par.resampling))
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


            

            # One search per analysis window length, all reading the same
            # whitened stream: the conditioning is the expensive part and is
            # done once, while each window length has its own stride, its own
            # coefficient grid and its own trigger file.
            searches, writers = [], []
            for window, overlap in self.schedule:
                par = Parameters()
                par.copy(self.par)
                par.window, par.overlap, par.Ncoeff = window, overlap, window
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
            data = SV()
            data_ds = SV()
            dataw = SV()
            # Tested on the block that comes out of conditioning, not on the
            # reader: the two are not at the same time, and testing the reader
            # would end the loop while conditioned data was still queued.
            data_ds = read_conditioned(streaming, data, ds)
            while data_ds.GetStart() <= self.par.gpsEnd:
                whitening.Process(data_ds, dataw)
                for search in searches:
                    search.SetData(dataw)
                    search.Process()
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
