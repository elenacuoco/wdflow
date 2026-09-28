"""Handing a NumPy array to the C++ core.

The `p4TSA` primitives read and write `SeqView`, a view carrying its samples
together with the time it starts at and the interval between them. This module
wraps a NumPy array in one, which is how every array assembled in Python enters
the transform, the whitening or the frame writer.

The epoch and the sampling interval travel with the samples deliberately. A
series that has passed through the conditioning chain does not begin where it
was asked to begin, and a consumer that assumes it does places everything
downstream at the wrong time; carrying the start in the view itself is what lets
the caller read the time the data actually has.
"""
__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"
__status__ = "Development"

from py4tsa.tsa import SeqView_double_t as SV
import logging
import numpy as np


class array2SeqView(object):
    """
    This class converts and array into Sequence View data that can be used later on by p4TSA methods
    """

    def __init__(self, start, sampling, N):
        """
        This class converts and array into Sequence View data that can be used later on by p4TSA methods

        :type start: float
        :param start: Start gps time for the data

        :type sampling: float
        :param sampling: Sampling rate of the data

        :type N: int
        :param N: Length of the vector stored in the Sequence View data
        """
        try:
            self.start = float(start)
        except ValueError:
            logging.info("starting time not defined")
        try:
            self.sampling = float(sampling)
        except ValueError:
            logging.info("sampling not defined")
        try:
            self.N = N
        except ValueError:
            logging.info("lenght not defined")
        self.SV = SV(self.start, 1.0 / self.sampling, self.N)

    def Fill(self, start, array):
        """
        Filles the Sequence View with the data from array

        :type start: float
        :param start: Start gps time

        :type array: numpy array
        :param array: Array of data to be converted to the Sequence View

        :return: Sequence View data
        """
        self.SV.SetStart(start)

        # The view is `SeqView<double>` and keeps every sample in double
        # precision, so the samples are handed over as they are. Narrowing them
        # to single precision on the way in adds a rounding noise 2^-24 below
        # each sample's own size, and that noise is white: in a band the
        # conditioning has attenuated by more than the single-precision range,
        # such as the stop band of the band-pass, it becomes the whole content
        # of the band, and the whitening lifts it back up as though it were the
        # detector's noise. The array is converted to plain floats once, since
        # converting each sample on its own builds a NumPy scalar per point,
        # which for a block of coefficients costs more than the transform it
        # feeds.
        values = np.asarray(array, dtype=np.float64).tolist()
        fill_point = self.SV.FillPoint
        for i, value in enumerate(values):
            fill_point(0, i, value)
        return self.SV

    def SetStart(self, N):
        """
        Alternative methods to set the start GPS time

        :type N: int
        :param N: Length of the vector stored in the Sequence View data
        """

        self.SV.SetStart(np.float(self.N) / self.sampling)

    def GetStart(self):
        """
        Returns start GPS time
        """
        return self.SV.GetStart()

    def GetSize(self):
        """
        Returns start GPS time
        """
        return self.SV.GetSize() 
        
    def SetSize(self,N):
        """
        Returns start GPS time
        """
        return self.SV.SetSize(N)    
       
