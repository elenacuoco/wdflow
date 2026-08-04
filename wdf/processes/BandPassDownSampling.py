__author__ = "Francesco Di Renzo, Elena Cuoco"
__project__ = "wdf"


import logging
from wdf.structures.array2SeqView import *
import numpy as np
from scipy.signal import sosfilt, sosfiltfilt, butter

def SV_to_array(seqView):
    """Copies a pytsa SeqView's single channel into a plain numpy array.

    :type seqView: pytsa.tsa.SeqView_double_t
    :param seqView: sequence view to read from (channel 0 only).
    :return: numpy.ndarray -- 1-D array of length `seqView.GetSize()`.
    """
    y = np.zeros(seqView.GetSize())
    for i in range(seqView.GetSize()):
        y[i] = seqView.GetY(0, i)
    return y


class BandPassDownSampling(object):
    """
    The downsampling class base on scipy and numpy library. First implemente a band pass sos filter , and late decimate the data
    """

    def __init__(self, Parameters, order=None, low_freq_hp=None, padlen=None,estimation=False):
        """
        The constructor

        :type Parameters: dict
        :param Parameters: The dictionary containing list of parameters
        :type order: int
        :order : the filter order; if None (the default), taken from
            `Parameters.FilterOrder`, falling back to 5 if that is unset. Pass a
            number to override the configured value.
        :type padlen:int
        :padlen:  the lenght of workspace for backward filter. It must be <= the lenght of the input data but more that 1 sampling frame to cut th transient effect
        """
        try:
            self.sampling = int(Parameters.sampling)
        except ValueError:
            logging.error("sampling not defined")
        try:
            self.resampling = int(Parameters.resampling)
        except ValueError:
            logging.error("Resampling  not defined")
        try:
            self.ResamplingFactor = int(Parameters.ResamplingFactor)
        except ValueError:
            logging.error("Resampling factor not defined")

        self.nyquist_frequency = 0.5 * self.sampling
        self.cutoff_frequency = 0.98 * (self.nyquist_frequency / self.ResamplingFactor)
         
        self.low_freq_hp = getattr(Parameters, "LowFrequencyCut", low_freq_hp)
        
        if order is None:
            order = getattr(Parameters, "FilterOrder", None)
        self.order = 5 if order is None else int(order)

         # Apply a low-pass filter to the data to prevent aliasing
        self.sos = butter(self.order,[self.low_freq_hp, self.cutoff_frequency], fs=self.sampling, btype='bandpass', output='sos')
        self.estimation=estimation
        

       # Get the steady state of the filter's step response.

        self.z1forw = np.zeros((self.sos.shape[0], 2), dtype=np.float32)
        self.first_call = True

        if padlen is None:
           self.padlen = int(self.sampling )
        else:
            self.padlen = padlen

        self.prefix = np.zeros(self.padlen)

    def Process(self, data):
        """
        The method for the downsampling the data.

        On the first call after `estimation=True` construction, applies a
        zero-phase (forward-backward, `sosfiltfilt`) band-pass + decimate in one
        shot and clears the estimation flag. On every subsequent call, runs a
        streaming forward-then-backward `sosfilt` pass instead (carrying filter
        state and a `padlen`-sized prefix/lookahead buffer across calls), so
        consecutive chunks stay continuous without needing the whole segment in
        memory at once.

        :type data: pytsa.tsa.SeqView_double_t
        :param data: input data chunk at the original sampling rate.
        :return: pytsa.tsa.SeqView_double_t -- band-passed, decimated data at
            `self.resampling` Hz.
        """
        ##
        DSdata = data.GetSize()
        # dimension of decimated data
        Noutdata = int(DSdata / self.ResamplingFactor)
        # decimate signal array
        y_ds = np.zeros(Noutdata)
         # signal array
        y = SV_to_array(data)

        if self.estimation==True:  
            y_ds=sosfiltfilt(self.sos,y)[::self.ResamplingFactor]
            data_ds = array2SeqView(data.GetStart(), self.resampling, Noutdata)
            data_ds.Fill(data.GetStart(), array=y_ds)
            data_ds = data_ds.SV 
            self.estimation=False        
        else:
            
            ## implementation of forward and backward filter Francesco

            ext = np.concatenate([self.prefix, y])
            s1 = ext[:DSdata]
            s2 = ext[DSdata:DSdata + self.padlen]
            self.prefix = ext[-self.padlen:]
            # Forward
            s1f, self.z1forw = sosfilt(self.sos, s1, zi=self.z1forw)
            s2f, z2f = sosfilt(self.sos, s2, zi=self.z1forw)
            # Backward
            s2b, z2b = sosfilt(self.sos, s2f[::-1], zi=z2f)
            s1b, z1b = sosfilt(self.sos, s1f[::-1], zi=z2b)

            y_ds = s1b[::-self.ResamplingFactor]
            startTime = data.GetStart() - self.padlen / self.sampling

            data_ds = array2SeqView(startTime, self.resampling, Noutdata)
            data_ds.Fill(startTime, array=y_ds)
            data_ds = data_ds.SV

        return data_ds
