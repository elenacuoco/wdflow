__author__ = "Francesco Di Renzo, Elena Cuoco"
__project__ = "wdf"


import logging
from wdf.structures.array2SeqView import *
import numpy as np
from scipy.signal import sosfilt, sosfiltfilt, butter

def SV_to_array(seqView):
    y = np.zeros(seqView.GetSize())
    for i in range(seqView.GetSize()):
        y[i] = seqView.GetY(0, i)
    return y


class BandPassDownSampling(object):
    """
    The downsampling class base on scipy and numpy library. First implemente a band pass sos filter , and late decimate the data
    """

    def __init__(self, Parameters, order=5, low_freq_hp=4., padlen=None,estimation=False):
        """
        The constructor

        :type Parameters: dict
        :param Parameters: The dictionary containing list of parameters
        :type order: int
        :order : the filter order
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
        self.low_freq_hp=low_freq_hp 
        # Set the default filter order if not specified
        if order is None:
            self.order = 4
        else:
            self.order = order
            
         

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
        The method for the downsampling the data
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
