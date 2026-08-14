"""Band-pass and downsample, before anything is estimated from the data.

The strain is dominated by frequencies the search does not use, and modelling
noise it will not look at spends the model's order where it buys nothing. This
stage restricts the band and reduces the rate to match it, so the noise model
and the transform that follow work on the band the search actually searches.

The filter is applied so that its own settling is accounted for rather than
left in the output: a filter has a memory, and the samples that carry only that
memory are not data.
"""
__author__ = "Francesco Di Renzo, Elena Cuoco"
__project__ = "wdf"


import logging
from wdf.structures.array2SeqView import *
import numpy as np
from scipy.signal import cheby2, sosfilt, sosfiltfilt

def SV_to_array(seqView):
    """Copies a pytsa SeqView's single channel into a plain numpy array.

    :type seqView: pytsa.tsa.SeqView_double_t
    :param seqView: sequence view to read from (channel 0 only).
    :return: numpy.ndarray -- 1-D array of length `seqView.GetSize()`.
    """
    y = np.zeros(seqView.GetSize())
    for i in range(seqView.GetSize()):
        y[i] = seqView.GetY(0, i)
    return y


def settling_length(sos, sampling, floor=1e-12, limit_s=8.0):
    """How many samples the filter needs before its response has decayed.

    Measured from the impulse response rather than assumed from the order: a
    steep filter close to Nyquist rings far longer than its order suggests, and
    this length is the context a block needs on each side.

    The floor is set by what happens downstream, not by what looks negligible
    here. Whitening against a high-order autoregressive model applies its
    largest gain at the band edges, which is exactly where the conditioning
    residual lives, so a residual small enough to ignore in the conditioned data
    can still dominate the block edges once whitened, where it reads as a short
    broadband burst. The default is therefore well below what the conditioned
    data alone would justify.

    :type sos: numpy.ndarray
    :param sos: second-order sections.
    :type sampling: float
    :param sampling: sampling frequency, Hz.
    :type floor: float
    :param floor: fraction of the peak below which the response is spent.
    :type limit_s: float
    :param limit_s: longest response to look for, seconds.
    :return: int -- samples until the response has decayed below `floor`.
    """
    impulse = np.zeros(int(limit_s * sampling))
    impulse[0] = 1.0
    response = np.abs(sosfilt(sos, impulse))
    peak = response.max()
    if peak <= 0.0:
        return 1
    above = np.flatnonzero(response > floor * peak)
    return int(above[-1]) + 1 if above.size else 1


class BandPassDownSampling(object):
    """
    Band-pass with zero phase, then decimate.

    Over a stream, drive this through `read_conditioned` rather than calling
    `Process` once per read: a block is emitted only once the data following it
    has arrived, so `Process` returns None until then.
    """

    def __init__(self, Parameters, order=None, low_freq_hp=None, padlen=None,
                 estimation=False, stopband_attenuation_db=60.0):
        """
        The constructor

        :type Parameters: dict
        :param Parameters: The dictionary containing list of parameters
        :type order: int
        :order : the filter order; if None (the default), taken from
            `Parameters.FilterOrder`, falling back to 10 if that is unset. Pass a
            number to override the configured value.
        :type stopband_attenuation_db: float
        :stopband_attenuation_db: attenuation reached at the band edges, in dB.
            This is what suppresses aliasing: everything above the decimated
            Nyquist folds back into the analysed band, so the attenuation
            reached before it is the only thing keeping it out.
        :type padlen: int
        :padlen: samples of real future data the backward pass settles over
            before it reaches the stretch being emitted. Measured from the
            impulse response when None; it must not exceed the read block, since
            it is taken from the block that follows the one emitted.
        """
        try:
            self.sampling = int(Parameters.sampling)
        except ValueError:
            logging.error("sampling not defined")
        try:
            self.resampling = int(Parameters.resampling)
        except ValueError:
            logging.error("Resampling  not defined")
        try:
            self.ResamplingFactor = int(Parameters.ResamplingFactor)
        except ValueError:
            logging.error("Resampling factor not defined")

        self.nyquist_frequency = 0.5 * self.sampling
        self.cutoff_frequency = 0.90 * (self.nyquist_frequency / self.ResamplingFactor)
         
        if low_freq_hp is None:
            low_freq_hp = getattr(Parameters, "LowFrequencyCut", None)
        self.low_freq_hp = 4.0 if low_freq_hp is None else float(low_freq_hp)

        if order is None:
            order = getattr(Parameters, "FilterOrder", None)
        self.order = 10 if order is None else int(order)
        self.stopband_attenuation_db = float(stopband_attenuation_db)

         # Apply a low-pass filter to the data to prevent aliasing. Chebyshev
         # type II is flat in the pass band, with its ripple confined to the
         # stop band where nothing is read, and it reaches full attenuation at
         # the edges given here rather than merely starting to roll off there.
        self.sos = cheby2(self.order, self.stopband_attenuation_db,
                          [self.low_freq_hp, self.cutoff_frequency],
                          fs=self.sampling, btype='bandpass', output='sos')
        self.estimation=estimation
        

        # Measured from this filter's own impulse response, not fixed: a value
        # tuned to a gentler filter is one a steeper one rings past, which puts
        # the unsettled transient into the emitted block.
        if padlen is None:
            self.padlen = settling_length(self.sos, self.sampling)
        else:
            self.padlen = int(padlen)

        # Blocks read but not yet emitted, and the stretch already emitted.
        self.pending = []
        self.history = np.zeros(0)

        logging.info(
            "BandPassDownSampling: %d -> %d Hz, band %.1f-%.1f Hz, order %d, "
            "%.0f dB, settling %d samples (%.3f s)",
            self.sampling, self.resampling, self.low_freq_hp,
            self.cutoff_frequency, self.order, self.stopband_attenuation_db,
            self.padlen, self.padlen / self.sampling)

    def Process(self, data):
        """
        The method for the downsampling the data.

        With `estimation=True` the block is complete in itself -- it is the
        stretch the autoregressive fit is handed -- so it is band-passed with
        `sosfiltfilt` and decimated in one shot.

        Otherwise a block is filtered only once `padlen` samples of what follows
        it have been read. `sosfiltfilt` is then applied to the block together
        with its real past and its real future, and only the middle is kept, so
        the result is what filtering the whole stream at once would give there.
        None is returned while the future is still arriving, however many reads
        that takes.

        The cost is one block of latency, stated in `latency_s` and carried by
        the timestamps. It is not optional: a block cannot be filtered with zero
        phase before the filter has seen what follows it, and the residual left
        by assuming a boundary instead is small in the conditioned data but is
        amplified by the whitening, which applies its largest gain exactly at
        the band edges where that residual lives.

        :type data: pytsa.tsa.SeqView_double_t
        :param data: input data chunk at the original sampling rate.
        :return: pytsa.tsa.SeqView_double_t or None -- band-passed, decimated
            data at `self.resampling` Hz, or None while the future is filling.
        """
        y = SV_to_array(data)
        start = data.GetStart()

        if self.estimation:
            y_ds = sosfiltfilt(self.sos, y)[::self.ResamplingFactor]
            self.estimation = False
            return self._decimated_view(y_ds, data.GetStart())

        self.pending.append((y, start))
        if sum(len(s) for s, _ in self.pending[1:]) < self.padlen:
            return None

        block, block_start = self.pending.pop(0)
        lookahead = np.concatenate([s for s, _ in self.pending])[:self.padlen]

        joined = np.concatenate([self.history, block, lookahead])
        filtered = sosfiltfilt(self.sos, joined)

        first = len(self.history)
        emitted = filtered[first:first + len(block)]
        self.history = joined[:first + len(block)][-self.padlen:]

        y_ds = emitted[::self.ResamplingFactor]
        return self._decimated_view(y_ds, block_start)

    @property
    def latency_s(self):
        """Seconds of data read but not yet emitted. Zero in estimation mode."""
        return sum(len(s) for s, _ in self.pending) / self.sampling

    def _decimated_view(self, y_ds, start):
        """Wrap decimated samples in a SeqView starting at `start`.

        :type y_ds: numpy.ndarray
        :param y_ds: decimated samples.
        :type start: float
        :param start: GPS time of the first sample.
        :return: pytsa.tsa.SeqView_double_t
        """
        view = array2SeqView(start, self.resampling, len(y_ds))
        view.Fill(start, array=y_ds)
        return view.SV


def read_conditioned(streaming, block, downsampling):
    """Read from a stream until the conditioning front end returns a block.

    This is the supported way to drive `BandPassDownSampling` over a stream.
    The filter holds each block until the data following it has arrived, so it
    returns None for the first few reads and a caller that assumes one block
    per read will hand None to whatever it feeds. How many reads it takes
    depends on the filter's ringing and on the read size, neither of which the
    caller should have to know.

    :type streaming: pytsa.tsa.FrameIChannel
    :param streaming: the frame reader.
    :type block: pytsa.tsa.SeqView_double_t
    :param block: scratch view the reader fills.
    :type downsampling: BandPassDownSampling
    :param downsampling: the conditioning front end.
    :return: pytsa.tsa.SeqView_double_t -- one conditioned block, labelled with
        the time of the samples it holds.
    """
    while True:
        streaming.GetData(block)
        conditioned = downsampling.Process(block)
        if conditioned is not None:
            return conditioned
