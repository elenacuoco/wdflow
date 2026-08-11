"""Whitening by an adaptive autoregressive model of the noise.

The noise model is estimated from the data as they arrive and updated with
them, so a change in the spectrum is followed rather than being fixed at the
start of a run. What the search sees is the residual: a series whose spectrum is
flat where the model is right, on which one threshold means the same thing at
every frequency.

The filter estimated here is causal. `zero_phase_whitening` builds the
square-root filter that removes the same colour without moving the transient in
time, which is what the reconstruction needs.
"""
__author__ = "Elena Cuoco"
__project__ = "pytsa"

from pytsa.tsa import ArBurgEstimator,LatticeView,LatticeFilter
from wdf.processes.ar_lv_io import save_ar_burg, load_ar_burg, save_lattice_view, load_lattice_view


class Whitening(object):

    """
    This class is responsible for the communiction with whitening functions from pytsa
    """

    def __init__(self, ARorder):
        """
        This class is responsible for the communiction with whitening functions from pytsa

        :type ARorder: int
        :param ARorder: The order for AutoRegressive filter
        """
        self.ARorder = ARorder
        self.ADE = ArBurgEstimator(self.ARorder)
        self.LV = LatticeView(self.ARorder)
        self.LF = LatticeFilter(self.LV)

    def ParametersEstimate(self, data):
        """
        This method estimates parameters of data by calling proper methods from pytsa

        :type data: pytsa.SeqViewDouble
        :param data: The Sequence View object containing the data to be processed
        """
        self.ADE(data)
        self.ADE.GetLatticeView(self.LV)
        self.LF.init(self.LV)

    def GetSigma(self):
        """
        This method returns the sigma parameter of the Whitening process

        :return: The sigma parameter of the whitened data
        """
        return self.ADE.GetAR(0)

    def Process(self, data, dataw):
        """
        This method whitens the data by calling proper function from pytsa

        :param data: pytsa.SeqViewDouble
        :param dataw: pytsa.SeqViewDouble
        """
        self.LF(data, dataw)
        return 

    def ParametersSave(self, ARfile, LVfile):
        """
        This method saves the calculated AR and LV parameter to the file
        (HDF5 -- see wdf.processes.ar_lv_io -- not p4TSA's old XML Save/Load).

        :type ARfile: basestring
        :param ARfile: file for AutoRegressive parameters

        :type LVfile: basestring
        :param LVfile: file for Lattice View parameters

        """
        save_ar_burg(ARfile, self.ADE)
        save_lattice_view(LVfile, self.LV)
        return

    def ParametersLoad(self, ARfile, LVfile):
        """
        This method loads the calculated AR and LV parameter from the file
        (HDF5 -- see wdf.processes.ar_lv_io -- not p4TSA's old XML Save/Load).

        :type ARfile: basestring
        :param ARfile: file for AutoRegressive parameters

        :type LVfile: basestring
        :param LVfile: file for Lattice View parameters

        :return: Autoregressive and Lattice View
        """
        load_ar_burg(ARfile, self.ADE)
        load_lattice_view(LVfile, self.LV)
        self.ADE.GetLatticeView(self.LV)
        ## not clear, but absolutly neeeded for initialitiate Dwhitening class
        load_lattice_view(LVfile, self.LV)
        self.LF.init(self.LV)
        return


        

         

    def GetLV(self):
        """
        This method returns LV object

        :return: LV object
        """

        return self.LV
