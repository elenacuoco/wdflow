__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"
__status__ = "Development"
import time

from pytsa.tsa import *
from pytsa.tsa import WaveletThreshold
from pytsa.tsa import SeqView_double_t as SV


from wdf.observers.ParameterEstimationObserver import ParameterEstimation 
from wdf.observers.SingleEventPrintFileObserver import SingleEventPrintTriggers
from wdf.processes.BandPassDownSampling import BandPassDownSampling
from wdf.config.Parameters import Parameters 
from wdf.processes.wdf import wdf
from wdf.processes.Whitening import Whitening 
from wdf.processes.DWhitening import DWhitening
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
        downsampling -> double-whitening -> WDF trigger search, writing triggers to
        `<outdir>/<run>/<itf>/<channel>_<gpsStart>/` as they're found.

        :type segment: tuple[float, float]
        :param segment: (gpsStart, gpsEnd) bounds of the segment to analyze.
        :type wavThresh: pytsa.tsa.WaveletThreshold.WaveletThresholding
        :param wavThresh: wavelet-coefficient thresholding rule passed to WDF's C++
            engine (default `dohonojohnston`, the Donoho-Johnstone universal threshold).
        :return: None -- triggers are written to disk (Parquet, or CSV for older runs),
            not returned; a `ProcessEnded.check` marker file in the segment's output
            directory means a prior run already completed it and this call is a no-op.
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
            self.par.ARfile = dir_chunk + "ARcoeff-AR%s-fs%s-%s.txt" % (
                self.par.ARorder,
                self.par.resampling,
                self.par.channel,
            )
            self.par.LVfile = dir_chunk + "LVcoeff-AR%s-fs%s-%s.txt" % (
                self.par.ARorder,
                self.par.resampling,
                self.par.channel,
            )

            if os.path.isfile(self.par.ARfile) and os.path.isfile(self.par.LVfile):
                logging.info("Load AR parameters")
                whiten.ParametersLoad(self.par.ARfile, self.par.LVfile)
                 
            else:
                logging.info("Start AR parameter estimation")
                ######## read data for AR estimation###############
                # self.parameter for sequence of data.
                # Add a 100.0 seconds delay to not include too much after lock noise in
                # the estimation, not needed if working in DataScience segments
                #
                if gpsEnd - gpsStart >= self.learn + 100.0:
                    gpsE = gpsStart + 100.0
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
            
            # update the self.parameters to be saved in local json file
            self.par.ID = ID
            self.par.dir = dir_chunk
            self.par.gps = gpsStart
            self.par.gpsStart = gpsStart
            # Safety margin: the main loop below checks the *previous* read's
            # start before issuing the next one, so its last read can extend
            # up to par.len past the checked bound. FrameIChannel does not
            # always raise cleanly when asked to read past the last available
            # frame data, so this margin keeps every read within legitimately
            # available data.
            self.par.gpsEnd = gpsEnd-self.par.len

            ######################
            # self.parameter for sequence of data and the resampling
        
            ds = BandPassDownSampling(self.par)        
            
            #Perform operation to intialite the detection loop    
            #gpsStart = gpsStart - self.par.preWhite            
            data = SV()
            data_ds = SV()
            dataw = SV()
            Noutdata = int(self.par.resampling)
            DW=DWhitening(whiten.LV, Noutdata,0)
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
                DW.Process(data_ds,dataw)
               
                
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
                # yet). DW.Input() is SetData-only (no GetData/Output), so it
                # does not require or consume an output chunk. extra_size is
                # in resampled-rate samples; each 1.0s native read yields
                # self.par.resampling resampled samples.
                extra_native_seconds = -(-extra_size // int(self.par.resampling))  # ceil division, no numpy needed
                for _ in range(extra_native_seconds):
                    streaming.GetData(data)
                    data_ds = ds.Process(data)
                    DW.Input(data_ds)

            #Set new size for the function in the loop
            streaming.SetDataLength(self.par.len)

            self.par.NoutData= int(self.par.resampling*self.par.len)
            DW.SetOutputSize(self.par.NoutData, extra_size)


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
                DW.Process(data_ds, dataw)
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
