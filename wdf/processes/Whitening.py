"""Whitening by an adaptive autoregressive model of the noise.

The noise model is estimated from the data as they arrive and updated with
them, so a change in the spectrum is followed rather than being fixed at the
start of a run. What the search sees is the residual: a series whose spectrum is
flat where the model is right, on which one threshold means the same thing at
every frequency.

The filter estimated here is causal: its output is ``A(z) x``, with the
modulus that whitens and the phase of ``A``. `zero_phase_whitening` builds from
the same model the filter whose response is that modulus alone, which removes
the same colour without moving the transient in time, as the reconstruction
needs. `CausalWhitening` runs this causal filter behind the zero-phase
interface, for a worker asked for it (`ZeroPhaseFilter = "causal"`).
"""
__author__ = "Elena Cuoco"
__project__ = "py4tsa"

import numpy as np
from py4tsa.tsa import ArBurgEstimator,LatticeView,LatticeFilter
from wdf.processes.ar_lv_io import save_ar_burg, load_ar_burg, save_lattice_view, load_lattice_view


class Whitening(object):

    """
    This class is responsible for the communiction with whitening functions from py4tsa
    """

    def __init__(self, ARorder):
        """
        This class is responsible for the communiction with whitening functions from py4tsa

        :type ARorder: int
        :param ARorder: The order for AutoRegressive filter
        """
        self.ARorder = ARorder
        self.ADE = ArBurgEstimator(self.ARorder)
        self.LV = LatticeView(self.ARorder)
        self.LF = LatticeFilter(self.LV)

    def ParametersEstimate(self, data):
        """
        This method estimates parameters of data by calling proper methods from py4tsa

        :type data: py4tsa.SeqViewDouble
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
        This method whitens the data by calling proper function from py4tsa

        :param data: py4tsa.SeqViewDouble
        :param dataw: py4tsa.SeqViewDouble
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


class CausalWhitening(object):
    """The fitted causal lattice filter behind the zero-phase interface.

    Every chunk given is filtered at once by `Whitening.Process`, which is
    causal and keeps its state across calls, and buffered; `Output` emits
    `output_size` samples labelled with the time of the first. The output is
    ``A(z) x``: the causal whitening's modulus with ``A``'s phase, so a
    transient comes out displaced by that phase, which the zero-phase filters
    avoid. Nothing is read ahead: the latency is zero and the lookahead is only
    held, as the worker asks. An output sample depends on the ``ARorder``
    samples before it, which the warm-up has to supply.

    The interface is `ZeroPhaseWhitening`'s, so the worker drives any of them
    the same way.
    """

    def __init__(self, whitening, output_size, extra_size=0):
        """
        :type whitening: Whitening
        :param whitening: the fitted, or loaded, causal whitening; its lattice
            filter is the one run.
        :type output_size: int
        :param output_size: whitened samples produced per `Output` call.
        :type extra_size: int
        :param extra_size: samples buffered beyond the output block before it
            is produced.
        """
        self.whitening = whitening
        self.sigma = float(whitening.GetSigma())
        self.output_size, self.extra_size = int(output_size), int(extra_size)
        self._buffer = np.zeros(0)
        self._start = None
        self._interval = None

    @property
    def latency(self):
        """Samples of future data an output sample depends on: none."""
        return 0

    def Input(self, data):
        """Whiten one chunk and append it to the buffer.

        The stream's time is taken from the first chunk it is given, and every
        later chunk is taken to follow the previous one.

        :type data: py4tsa.tsa.SeqView_double_t
        :param data: input chunk, band-passed and decimated.
        :return: None
        """
        from py4tsa.tsa import SeqView_double_t

        from wdf.processes.BandPassDownSampling import SV_to_array

        out = SeqView_double_t()
        self.whitening.Process(data, out)
        if self._start is None:
            self._start, self._interval = data.GetStart(), data.GetSampling()
        self._buffer = np.concatenate([self._buffer, SV_to_array(out) * out.GetScale()])

    def Output(self, dataw):
        """Emit the next `output_size` whitened samples.

        :type dataw: py4tsa.tsa.SeqView_double_t
        :param dataw: output view, replaced by the whitened block, labelled
            with the time of its first sample.
        :return: None
        :raises RuntimeError: if fewer than ``output_size + extra_size``
            samples are buffered.
        """
        from wdf.structures.array2SeqView import array2SeqView

        n = self.output_size
        if self._buffer.size < n + self.extra_size:
            raise RuntimeError(
                f"CausalWhitening: {self._buffer.size} samples buffered, "
                f"{n + self.extra_size} needed")
        view = array2SeqView(self._start, 1.0 / self._interval, n)
        view.Fill(self._start, self._buffer[:n])
        view.SV.SetScale(1.0)
        dataw.assign(view.SV)
        self._buffer = self._buffer[n:]
        self._start += self._interval * n

    def Process(self, data, dataw):
        """Whiten one chunk and emit the next block.

        :type data: py4tsa.tsa.SeqView_double_t
        :param data: input chunk, band-passed and decimated.
        :type dataw: py4tsa.tsa.SeqView_double_t
        :param dataw: output view, replaced by the whitened block.
        :return: None
        """
        self.Input(data)
        self.Output(dataw)

    def DataNeeded(self):
        """Buffered samples beyond what the next output block needs.

        The quantity `ZeroPhaseWhitening.DataNeeded` returns, with the same
        sign: negative means the next block cannot be produced yet.

        :return: int
        """
        return int(self._buffer.size - (self.output_size + self.extra_size))

    def SetOutputSize(self, output_size, extra_size):
        """Change the output block size and the lookahead.

        :type output_size: int
        :param output_size: whitened samples produced per `Output` call.
        :type extra_size: int
        :param extra_size: samples buffered beyond the output block.
        :return: None
        """
        self.output_size, self.extra_size = int(output_size), int(extra_size)
