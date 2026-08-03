"""Whitening

.. moduleauthor:: Elena Cuoco <elena.cuoco@unibo.it>

"""
__author__ = "Elena Cuoco"
__project__ = "pytsa"

from pytsa.tsa import DoubleWhitening
from pytsa.tsa import SeqView_double_t as SV
from pytsa.tsa import LatticeView 

class DWhitening(object):
    
    def __init__(self, LV, OutputSize, ExtraSize):
        """
        :type LV: pytsa.tsa.LatticeView-like
        :param LV: AR lattice-filter coefficients (from `Whitening.ParametersEstimate`/
            `ParametersLoad`) used to build this instance's own `LatticeView`.
        :type OutputSize: int
        :param OutputSize: number of whitened output samples produced per `Process` call.
        :type ExtraSize: int
        :param ExtraSize: extra lookahead/lookbehind buffer size the double-whitening
            (forward+backward AR pass) lattice filter needs to settle before its output
            is valid.
        """
        self.LV = LatticeView(LV)
        self.DW = DoubleWhitening(self.LV, OutputSize, ExtraSize)
        self.DW.init(self.LV)
    
    def ParametersLoad(self, LVfile):
        """
        This method loads the calculated AR and LV parameter from the file

       

        :type LVfile: basestring
        :param LVfile: file for Lattice View parameters

        :return: Autoregressive and Lattice View
        """
       
        self.LV.Load(LVfile, "txt")
        ## not clear, but absolutly neeeded for initialitiate Dwhitening class
        self.LV.Load(LVfile)
       
        return   self.LV  

    def Process(self, data, dataw):
        """Synchronous double-whitening: blocks until `OutputSize` whitened
        samples are available, writing them into `dataw`.

        :type data: pytsa.tsa.SeqView_double_t
        :param data: input (downsampled, band-passed) data chunk to feed the filter.
        :type dataw: pytsa.tsa.SeqView_double_t
        :param dataw: output sequence view, filled in place with the whitened samples.
        :return: None
        """
        self.DW(data,dataw)

        return
    def ProcessAsync(self, data, dataw):
        """Non-blocking variant of `Process`: feeds `data` in, then attempts to
        read whatever output is already available without waiting for a full
        `OutputSize` block -- prints a message and leaves `dataw` untouched if
        the filter isn't ready yet, rather than raising.

        :type data: pytsa.tsa.SeqView_double_t
        :param data: input data chunk to feed the filter.
        :type dataw: pytsa.tsa.SeqView_double_t
        :param dataw: output sequence view, filled in place if output is ready.
        :return: None
        """
        self.DW.Input(data)
        try:
            self.DW.Output(dataw)
        except BaseException:
            print("no output data available")

        return

    def Input(self,data):
        """Feeds one data chunk into the lattice filter without reading output.

        :type data: pytsa.tsa.SeqView_double_t
        :param data: input data chunk.
        :return: None
        """
        self.DW.Input(data)
        return

    def Output(self,data):
        """Reads whatever whitened output is currently available, if any.

        :type data: pytsa.tsa.SeqView_double_t
        :param data: output sequence view, filled in place; left untouched (with a
            printed message) if no output is available yet.
        :return: None
        """
        try:
            self.DW.Output(data)
        except BaseException:
            print("no output data available")
        return

    def Init(self,LV) :
        """Re-initializes the underlying `DoubleWhitening` filter state with a
        (possibly updated) set of lattice-view coefficients.

        :type LV: pytsa.tsa.LatticeView-like
        :param LV: AR lattice-filter coefficients.
        :return: None
        """
        self.DW.init(LV)


    def SetOutputSize(self, Nout, Extrasize):
        """Changes the output block size and lookahead/lookbehind margin without
        rebuilding the filter from scratch.

        :type Nout: int
        :param Nout: number of whitened output samples produced per `Process` call.
        :type Extrasize: int
        :param Extrasize: extra lookahead/lookbehind buffer size (see `__init__`).
        :return: None
        """
        self.DW.SetOutputSize(Nout, Extrasize)

    def GetDataNeeded(self):
        """
        :return: int -- number of additional input samples the filter needs
            before its next `Output` call can produce a full block.
        """
        return self.DW.GetDataNeeded()
     
 