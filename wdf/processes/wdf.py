"""The search itself: transform a block, threshold it, and emit what survived.

Each analysis block is transformed in every candidate basis, the basis that
represents it most compactly wins, and the coefficients that stand above the
noise are kept. What leaves this stage is those coefficients and the noise scale
they were measured against, which is the whole of what a trigger is.

The stage is an observable: it does not know what happens to a trigger, and the
parameter estimation, the writing and any diagnostics attach as observers.
"""
__author__ = "Elena Cuoco"
__project__ = "wdf"

from py4tsa.tsa import WaveletThreshold, WDF2Classify, EventFullFeatured

from wdf.observers.observable import Observable
from wdf.config import Parameters


def use_bases(classify, bases):
    """Make these bases the candidates of the per-window competition.

    A name is a mother for its pyramidal transform (``"DaubC8"``), or a mother
    followed by ``P`` and a depth for that uniform level of its wavelet-packet
    tree (``"Coif1P6"``, `wdf.analysis.wavelets.packet_depth`).

    :type classify: py4tsa.tsa.WDF2Classify
    :param classify: the search whose candidates are replaced.
    :type bases: str or list[str]
    :param bases: the candidate names, as a list or one comma-separated string.
    :raises RuntimeError: if the installed py4tsa cannot take a list of bases.
    :raises ValueError: if no name is given.
    """
    names = [bases] if isinstance(bases, str) else list(bases)
    names = [n.strip() for name in names for n in str(name).split(",") if n.strip()]
    if not names:
        raise ValueError("bases names no candidate basis; leave it unset for the default ones")
    if not hasattr(classify, "SetBases"):
        raise RuntimeError(
            "the installed py4tsa has no WDF2Classify.SetBases and searches its default bases "
            f"only; {','.join(names)} need a py4tsa built with wavelet-packet bases")
    classify.SetBases(",".join(names))


class wdf(Observable):

    """
    The main WDF class responsible for the communication with the p4TSA library regarding the application of WDF onto data
    """

    def __init__(self, WdfParams: Parameters, wTh=WaveletThreshold.block):
        """
        The main WDF class responsible for the communication with the p4TSA library regarding the application of WDF onto data

        :type parameters: WdfParam
        :param parameters: Set of WDF parameters

        :type wth: py4tsa.WaveletThreshold
        :param wavThresh: the rule for the coefficients of a window; default `WaveletThreshold.block`,
            contiguous coefficients of one level judged together (Cai 1999). `dohonojohnston` is the
            universal threshold on each coefficient alone, which the method paper's runs used.

        The candidate bases are `WdfParams.bases` where it is given (`use_bases`), and the
        core's own ten pyramids otherwise; the list that competes is written back to
        `WdfParams.bases`, so the parameters a run dumps name the bases it searched.
        """
        Observable.__init__(self)
        self.parameters = WdfParams 
        self.wdf2classify = WDF2Classify(
            self.parameters.window,
            self.parameters.overlap,
            self.parameters.threshold,
            self.parameters.sigma,
            self.parameters.Ncoeff,
            wTh,
        )
        bases = getattr(self.parameters, "bases", None)
        if bases is not None:
            use_bases(self.wdf2classify, bases)
        if hasattr(self.wdf2classify, "GetBases"):
            self.parameters.bases = self.wdf2classify.GetBases().split(",")
        else:
            self.parameters.bases = None
        self.trigger = EventFullFeatured(self.parameters.Ncoeff)

    def SetData(self, data):
        """
        This methods sets sets the data for the p4TSA wdf2classify class for further search of triggers

        :type data: py4tsa.SeqViewDouble
        :param data: An py4tsa.SeqViewDouble object storing data to be processed

        """
       
        self.wdf2classify(data, self.parameters.sigma)

    def FindEvents(self):
        """
        This method calls wdf2classify function from py4tsa to search for triggers in the data

        :return: trigger
        """
        
        self.wdf2classify(self.trigger)
        return self.trigger

    def Process(self):
        """
        This method calls wdf2classify function from py4tsa to search for triggers in the data
        If the triggers are found, they are stored in tosend_triggers variable that is later on used for further processing

        """
        while self.wdf2classify.GetDataNeeded() > 0:
            m = self.wdf2classify(self.trigger)
            if m == 1:
                self.update_observers(self.trigger)
