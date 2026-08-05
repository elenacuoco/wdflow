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

from wdf.processes.BandPassDownSampling import BandPassDownSampling
from wdf.config.Parameters import Parameters 
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
    def __init__(self, parameters, fullPrint=1):
        """
        :type parameters: class Parameters object
        :param parameters: run configuration (channel, sampling, window, downsampling,
            AR whitening order, learn length, output paths, ...); copied onto a fresh
            `Parameters` instance so per-worker mutations (e.g. `Ncoeff`, `resampling`,
            `sigma`, set during `segmentProcess`) don't leak back into the caller's object.
        :type fullPrint: int
        :param fullPrint: verbosity passed through to `ParameterEstimation`/trigger
            printing -- e.g. whether whitened-domain reconstruction columns (`rw*`) are
            included in the output triggers.
        """
        self.par = Parameters()
        self.par.copy(parameters)
        self.par.Ncoeff = parameters.window
        self.fullPrint = fullPrint
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
            
            self.par.gpsEnd = gpsEnd - 2 * self.par.len

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
                streaming.GetData(data)
                data_ds=ds.Process(data) 
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

            if extra_size > 0:
                # Prime the whitening buffer with `extra_size` samples of real
                # future data *before* the main loop starts. DoubleWhitening::
                # GetData requires mOutputSize + ExtraSize samples already
                # buffered, and each call only removes mOutputSize samples --
                # so the ExtraSize surplus must be pre-loaded exactly once,
                # here, using extra reads at the still-active warm-up 1.0s
                # dLength (SetDataLength(self.par.len) hasn't been called
                # yet). whitening.Input() is SetData-only (no GetData/Output), so it
                # does not require or consume an output chunk. extra_size is
                # in resampled-rate samples; each 1.0s native read yields
                # self.par.resampling resampled samples.
                extra_native_seconds = -(-extra_size // int(self.par.resampling))  # ceil division, no numpy needed
                for _ in range(extra_native_seconds):
                    streaming.GetData(data)
                    data_ds = ds.Process(data)
                    whitening.Input(data_ds)

            #Set new size for the function in the loop
            streaming.SetDataLength(self.par.len)

            self.par.NoutData= int(self.par.resampling*self.par.len)
            whitening.SetOutputSize(self.par.NoutData, extra_size)


            

            # WDF process
            WDF = wdf(self.par, wavThresh)
            
            # register obesevers to WDF process
            # put 0 to save only metaself.parameters, 1 for wavelet coefficients and 2
            # for waveform estimation, 3 for full event print
            savetrigger = SingleEventPrintTriggers(self.par, self.fullPrint)
            parameterestimation = ParameterEstimation(self.par)
            parameterestimation.register(savetrigger)
            WDF.register(parameterestimation)
            filejson = "parametersUsed.json"
            self.par.dump(self.par.dir + filejson)
            # Start detection loop
            logging.info("Starting detection loop")
            data = SV()
            data_ds = SV()
            dataw = SV()
            while data.GetStart() <=self.par.gpsEnd:
                streaming.GetData(data)
                data_ds=ds.Process(data)
                whitening.Process(data_ds, dataw)
                WDF.SetData(dataw)
                WDF.Process()

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
