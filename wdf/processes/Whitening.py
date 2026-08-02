__author__ = "Elena Cuoco"
__project__ = "pytsa"

from pytsa.tsa import ArBurgEstimator,LatticeView,LatticeFilter


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

        :type ARfile: basestring
        :param ARfile: file for AutoRegressive parameters

        :type LVfile: basestring
        :param LVfile: file for Lattice View parameters

        """
        self.ADE.Save(ARfile, "txt")
        self.LV.Save(LVfile, "txt")
        return 

    def ParametersLoad(self, ARfile, LVfile):
        """
        This method loads the calculated AR and LV parameter from the file

        :type ARfile: basestring
        :param ARfile: file for AutoRegressive parameters

        :type LVfile: basestring
        :param LVfile: file for Lattice View parameters

        :return: Autoregressive and Lattice View
        """
        self.ADE.Load(ARfile, "txt")
        self.LV.Load(LVfile, "txt")
        self.ADE.GetLatticeView(self.LV)
        ## not clear, but absolutly neeeded for initialitiate Dwhitening class
        self.LV.Load(LVfile)
        self.LF.init(self.LV)
        return 


        

         

    def GetLV(self):
        """
        This method returns LV object

        :return: LV object
        """

        return self.LV
