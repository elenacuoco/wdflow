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
from scipy.signal import cheby2, sosfilt, sosfreqz

from wdf.filtering import sosfiltfilt


def SV_to_array(seqView):
    """Copies a py4tsa SeqView's single channel into a plain numpy array.

    :type seqView: py4tsa.tsa.SeqView_double_t
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


def discard_edges(view, seconds):
    """A view without `seconds` of samples at each end.

    What a filter run over a stretch in one shot leaves at the stretch's two
    ends is its own settling, not data: the band-pass started from an assumed
    boundary rather than from the samples that preceded it. When the stretch
    was read with real data on each side of the part that is wanted, dropping
    those sides leaves only samples the filter reached already settled.

    :type view: py4tsa.tsa.SeqView_double_t
    :param view: the filtered stretch.
    :type seconds: float
    :param seconds: seconds dropped at each end.
    :return: py4tsa.tsa.SeqView_double_t -- the remaining samples, starting at
        the time of the first one kept, taken from `view`'s own start.
    :raises ValueError: if nothing would remain.
    """
    interval = view.GetSampling()
    drop = round(float(seconds) / interval)
    size = view.GetSize()
    if 2 * drop >= size:
        raise ValueError(f"dropping {drop} samples at each end of {size} leaves nothing")
    kept = SV_to_array(view)[drop:size - drop]
    start = view.GetStart() + drop * interval
    out = array2SeqView(start, 1.0 / interval, kept.size)
    out.Fill(start, kept)
    return out.SV


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

        With `estimation=True` the block is complete in itself, so it is
        band-passed with `sosfiltfilt` and decimated in one shot. Its two ends
        then carry the filter's settling from an odd extension of a few dozen
        samples rather than from real data; a caller that needs settled samples
        reads real data on each side and drops it with `discard_edges`, as the
        worker does for the stretch the noise model is fitted on.

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

        :type data: py4tsa.tsa.SeqView_double_t
        :param data: input data chunk at the original sampling rate.
        :return: py4tsa.tsa.SeqView_double_t or None -- band-passed, decimated
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
        # The decimation below starts at this block's own first sample, so the
        # phase it picks is the phase of the stream only while every block
        # holds a whole number of decimated samples. A single block that does
        # not shifts every sample after it by one sample of the input, which is
        # a step at that join -- amplified by the whitening, whose gain at the
        # band edges is large, into something indistinguishable from a burst --
        # and a permanent timing bias thereafter. Nothing in the reader
        # enforces it, so it is checked here rather than assumed.
        if len(block) % self.ResamplingFactor:
            raise ValueError(
                f"a block of {len(block)} samples cannot be decimated by "
                f"{self.ResamplingFactor} without moving the decimation phase "
                f"of everything that follows it; read blocks whose length is a "
                f"multiple of {self.ResamplingFactor} samples "
                f"({self.ResamplingFactor / self.sampling:.6g} s)")
        lookahead = np.concatenate([s for s, _ in self.pending])[:self.padlen]

        joined = np.concatenate([self.history, block, lookahead])
        filtered = sosfiltfilt(self.sos, joined)

        first = len(self.history)
        emitted = filtered[first:first + len(block)]
        self.history = joined[:first + len(block)][-self.padlen:]

        y_ds = emitted[::self.ResamplingFactor]
        return self._decimated_view(y_ds, block_start)

    def passband(self, level_db=-3.0, resolution_hz=1.0 / 1024.0):
        """The band where the conditioning passes the data, read off its design.

        The filter is applied forward and backward, so the response the data
        receive is ``|H|^2``; the edges are the first and the last frequency of
        the decimated band at which ``20 log10 |H|^2`` is at least `level_db`.
        They are measured on the designed sections, not assumed from the
        corner frequencies the design was given: a Chebyshev type II reaches
        its attenuation at those corners, and its passband ends inside them.
        This is the band a whitening built after this stage is to whiten, and
        outside it the stop band is the conditioning's own and nothing to undo.

        :type level_db: float
        :param level_db: level of ``|H|^2`` that defines the edges, dB, negative.
        :type resolution_hz: float
        :param resolution_hz: spacing of the frequencies the response is read on.
        :return: tuple -- ``(f_lo, f_hi)`` in hertz.
        :raises ValueError: if the response nowhere reaches `level_db`.
        """
        freq = np.arange(0.0, 0.5 * self.resampling, float(resolution_hz))
        _, response = sosfreqz(self.sos, worN=freq, fs=self.sampling)
        gain_db = 20.0 * np.log10(np.abs(response) ** 2 + 1e-300)
        inside = np.flatnonzero(gain_db >= float(level_db))
        if inside.size == 0:
            raise ValueError(f"the band-pass response never reaches {level_db} dB")
        return float(freq[inside[0]]), float(freq[inside[-1]])

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
        :return: py4tsa.tsa.SeqView_double_t
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

    :type streaming: py4tsa.tsa.FrameIChannel
    :param streaming: the frame reader.
    :type block: py4tsa.tsa.SeqView_double_t
    :param block: scratch view the reader fills.
    :type downsampling: BandPassDownSampling
    :param downsampling: the conditioning front end.
    :return: py4tsa.tsa.SeqView_double_t -- one conditioned block, labelled with
        the time of the samples it holds.
    """
    while True:
        streaming.GetData(block)
        conditioned = downsampling.Process(block)
        if conditioned is not None:
            return conditioned
