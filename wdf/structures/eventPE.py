__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = ["http://www.giantflyingsaucer.com/"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@ego-gw.it"
__status__ = "Development"

from collections import OrderedDict
 


class OrderedMeta(type):
    @classmethod
    def __prepare__(metacls, name, bases):
        return OrderedDict()

    def __new__(cls, name, bases, clsdict):
        c = type.__new__(cls, name, bases, clsdict)
        c._orderedKeys = clsdict.keys()
        return c


class eventPE(object):

    """
    This class stands for the encapsulation of the trigger data into one object
    """

    def __init__(self, gps, gpsPeak, duration, EnWDF, snrMean, snrPeak, freqMin, freqMean, freqMax, freqPeak, wave, coeff, Icoeff):
        """
        This class stands for the encapsulation of the trigger data into one object

        :type gps: float
        :param gps: GPS time of the trigger denoting the first gps of analyzing window
        
        :type gpsPeak: float
        :param gps: GPS time of the trigger denoting the moment it appeared at maximum SNR

        :type EnWDF: float
        :param EnWDF: The Signal to Noise Ratio of the trigger statistics of WDF

        :type snrMean: float
        :param snrMean: The estimated mean Signal to Noise Ratio of the trigger

        :type snrPeak: float
        :param snrPeak: The estimated Signal to Noise Ratio of the trigger at its peak

        :type freqMin: float
        :param freqMin: The minimum frequency of the trigger

        :type freqMax: float
        :param freqMax: The maximum frequency of the trigger
        
        :type freqMean: float
        :param freqMean: The mean frequency of the trigger
        
        :type freqPeak: float
        :param freqPeak: The frequency at the peak of the trigger

        :type duration: float
        :param duration: The time duration of the trigger

        :type wave: str
        :param wave: The type of the wavelet

        :type coeff: list
        :param coeff: The list containing wavelet coefficients of the trigger

        :type Icoeff: list
        :param Icoeff: The list containing raw wavelet coefficients of the trigger
        """
        self.gps = gps
        self.gpsPeak = gpsPeak
        self.EnWDF = EnWDF
        self.snrMean = snrMean
        self.snrPeak = snrPeak
        self.freqMean = freqMean
        self.freqMin = freqMin
        self.freqMax = freqMax
        self.freqPeak = freqPeak
        self.duration = duration
        self.wave = wave
        self.Ncoeff = len(coeff)
        for i in range(len(coeff)):
            setattr(self, "wt" + str(i), coeff[i])
        for i in range(len(Icoeff)):
            setattr(self, "rw" + str(i), Icoeff[i])
 
    def update(self, gps, gpsPeak, duration, EnWDF, snrMean, snrPeak, freqMin, freqMean, freqMax, freqPeak, wave, coeff, Icoeff):
        """
        This method updates the eventPE object with new parameters
        
        :type gps: float
        :param gps: GPS time of the trigger denoting the first gps of analyzing window
        
        :type gpsPeak: float
        :param gps: GPS time of the trigger denoting the moment it appeared at maximum SNR

        :type EnWDF: float
        :param EnWDF: The Signal to Noise Ratio of the trigger statistics of WDF

        :type snrMean: float
        :param snrMean: The estimated mean Signal to Noise Ratio of the trigger
        
        :type snrPeak: float
        :param snrPeak: The estimated Signal to Noise Ratio of the trigger at its peak

        :type freqMin: float
        :param freqMin: The minimum frequency of the trigger

        :type freqMax: float
        :param freqMax: The maximum frequency of the trigger
        
        :type freqMean: float
        :param freqMean: The mean frequency of the trigger
        
        :type freqPeak: float
        :param freqPeak: The frequency at the peak of the trigger

        :type duration: float
        :param duration: The time duration of the trigger

        :type wave: str
        :param wave: The type of the wavelet

        :type coeff: list
        :param coeff: The list containing wavelet coefficients of the trigger

        :type Icoeff: list
        :param Icoeff: The list containing raw wavelet coefficients of the trigger
        """
        self.gps = gps
        self.gpsPeak = gpsPeak
        self.EnWDF = EnWDF
        self.snrMean = snrMean
        self.snrPeak = snrPeak
        self.freqMean = freqMean
        self.freqMin = freqMin
        self.freqMax = freqMax
        self.freqPeak = freqPeak
        self.duration = duration
        self.wave = wave
        for i in range(len(coeff)):
            setattr(self, "wt" + str(i), coeff[i])
        for i in range(len(Icoeff)):
            setattr(self, "rw" + str(i), Icoeff[i])

    def evCopy(self, ev):
        """
        This method copies the parameter of the ev, eventPE object

        :type ev: eventPE
        :param ev: The eventPE object to copy parameters from
        
        :type gps: float
        :param gps: GPS time of the trigger denoting the first gps of analyzing window
        
        
        :type gpsPeak: float
        :param gps: GPS time of the trigger denoting the moment it appeared at maximum SNR

        :type EnWDF: float
        :param EnWDF: The Signal to Noise Ratio of the trigger statistics of WDF

        :type snrMean: float
        :param snrMean: The estimated mean Signal to Noise Ratio of the trigger
        
        :type snrPeak: float
        :param snrPeak: The estimated Signal to Noise Ratio of the trigger at its peak

        :type freqMin: float
        :param freqMin: The minimum frequency of the trigger

        :type freqMax: float
        :param freqMax: The maximum frequency of the trigger
        
        :type freqMean: float
        :param freqMean: The mean frequency of the trigger
        
        :type freqPeak: float
        :param freqPeak: The frequency at the peak of the trigger

        :type duration: float
        :param duration: The time duration of the trigger

        :type wave: str
        :param wave: The type of the wavelet

        :type coeff: list
        :param coeff: The list containing wavelet coefficients of the trigger

        :type Icoeff: list
        :param Icoeff: The list containing raw wavelet coefficients of the trigger
        """
        
        self.gps = ev.gps
        self.gpsPeak = ev.gpsPeak
        self.EnWDF = ev.EnWDF
        self.snrMean = ev.snrMean
        self.snrPeak = ev.snrPeak
        self.freqMean = ev.freqMean
        self.freqMin = ev.freqMin
        self.freqMax = ev.freqMax
        self.freqPeak = ev.freqPeak
        self.duration = ev.duration
        self.wave = ev.wave

        for i in range(self.Ncoeff):
            setattr(self, "wt" + str(i), setattr(ev, "wt" + str(i)))
            setattr(self, "rw" + str(i), setattr(ev, "rw" + str(i)))
