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

    A name is a mother for its pyramidal transform (``"DaubC8"``), a mother
    followed by ``P`` and a depth for that uniform level of its wavelet-packet
    tree (``"Coif1P6"``, `wdf.analysis.wavelets.packet_depth`), or
    ``LocalCos`` followed by a segment length M for the local cosine basis
    (``"LocalCos128"``; under the block rule M <= window / round(ln window)).

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


def use_min_frequency(classify, f_min, fs):
    """Leave the coefficients at or below `f_min` out of the competition.

    Every candidate drops the coefficients whose tile's upper frequency edge
    is <= `f_min`, read on its own layout (pyramid levels, packet bands,
    LocalCos rows): they are out of the window's noise scale, out of the block
    rule and out of EnWDF, and the trigger holds zeros there (p4TSA
    `WDF2Classify.SetMinFrequency`). At a window of 1024 and 2048 Hz, 16 Hz
    drops pyramid indices 0-15 and LocalCos128 rows 0-1.

    :type classify: py4tsa.tsa.WDF2Classify
    :param classify: the search to restrict.
    :type f_min: float
    :param f_min: the minimum frequency, Hz; 0 or less keeps every coefficient
        and makes no call, so the search is the core's own.
    :type fs: float
    :param fs: the rate of the searched stream, Hz.
    :return: float -- the minimum frequency in force, 0.0 for none.
    :raises RuntimeError: if a cut is asked of a py4tsa without one.
    :raises ValueError: if `f_min` is not below fs / 2.
    """
    f_min = float(f_min or 0.0)
    if f_min <= 0.0:
        return 0.0
    if not hasattr(classify, "SetMinFrequency"):
        raise RuntimeError(
            f"the installed py4tsa has no WDF2Classify.SetMinFrequency; MinFrequency {f_min} Hz needs "
            "p4TSA feature/cosine-packets d335d33 or later")
    classify.SetMinFrequency(f_min, float(fs))
    return f_min


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

        The candidate bases are `WdfParams.Bases` (or `WdfParams.bases`) where it is given,
        a list or one comma-separated string (`use_bases`), and the core's own default list
        otherwise; the list that competes is written back to `WdfParams.bases` (and to
        `WdfParams.Bases` when that was the key given), so the parameters a run dumps name the
        bases it searched.

        `WdfParams.MinFrequency` (Hz, default 0: none) leaves the coefficients whose tile lies
        at or below it out of sigma, the block rule and EnWDF (`use_min_frequency`), read with
        the rate `WdfParams.resampling`; the value in force is written back.
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
        given = getattr(self.parameters, "Bases", None)
        bases = given if given is not None else getattr(self.parameters, "bases", None)
        if bases is not None:
            use_bases(self.wdf2classify, bases)
        if hasattr(self.wdf2classify, "GetBases"):
            self.parameters.bases = self.wdf2classify.GetBases().split(",")
        else:
            self.parameters.bases = None
        if given is not None:
            self.parameters.Bases = self.parameters.bases
        f_min = getattr(self.parameters, "MinFrequency", None)
        if f_min:
            self.parameters.MinFrequency = use_min_frequency(
                self.wdf2classify, f_min, getattr(self.parameters, "resampling", None) or 0.0)
        elif f_min is not None:
            self.parameters.MinFrequency = 0.0
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
