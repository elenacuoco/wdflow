__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@ego-gw.it"
__status__ = "Development"

from pytsa.tsa import SeqView_double_t as SV
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
         
        for i in range(self.N):
            self.SV.FillPoint(0, i, np.float32(array[i]))
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
       
