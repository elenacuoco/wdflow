__author__ = "Elena Cuoco"
__project__ = "wdf"

from pytsa.tsa import ARMAFilter, ArBurgEstimator
from wdf.processes.ar_lv_io import load_ar_burg


class Coloring(object):
    """Inverse-whitening (recoloring): applies the AR filter estimated for
    whitening in reverse, turning a whitened time series back into strain
    units.
    """

    def __init__(self, ARorder):
        """
        :type ARorder: int
        :param ARorder: The order of the AutoRegressive filter used for whitening
        """
        self.ARorder = ARorder
        self.ADE = ArBurgEstimator(self.ARorder)

    def ParametersLoad(self, ARfile):
        """Loads the AR coefficients estimated for whitening from `ARfile`
        and builds the corresponding recoloring ARMA filter.

        :type ARfile: basestring
        :param ARfile: file with the AutoRegressive coefficients (HDF5 -- see
            wdf.processes.ar_lv_io -- not p4TSA's old XML Save/Load)
        """
        load_ar_burg(ARfile, self.ADE)
        arorder = self.ARorder + 1
        self.ARMAflt = ARMAFilter(arorder, 1, self.ADE.GetAR(0))
        self.ARMAflt.SetARFilter(0, 1.0)

        for i in range(1, arorder):
            self.ARMAflt.SetARFilter(i, self.ADE.GetAR(i))

        self.ARMAflt.SetMAFilter(0, self.ADE.GetAR(0))

    def Process(self, dataw, datac):
        """Recolors one chunk of whitened data.

        :param dataw: pytsa.SeqViewDouble, whitened input
        :param datac: pytsa.SeqViewDouble, recolored output
        """
        self.ARMAflt(dataw, datac)
