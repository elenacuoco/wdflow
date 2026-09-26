"""Notch the detector's lines, band-pass and downsample, before anything is
estimated from the data.

The strain is dominated by frequencies the search does not use, and modelling
noise it will not look at spends the model's order where it buys nothing. This
stage restricts the band and reduces the rate to match it, so the noise model
and the transform that follow work on the band the search actually searches.
The detector's own lines are cut down to the floor first, by notches stacked in
front of the band-pass and applied with it (`wdf.processes.lines`): the
detectors share the band-pass, and each has its own lines.

The filter is applied so that its own settling is accounted for rather than
left in the output: a filter has a memory, and the samples that carry only that
memory are not data.
"""
__author__ = "Francesco Di Renzo, Elena Cuoco"
__project__ = "wdf"


import logging
from wdf.structures.array2SeqView import *
import numpy as np
from scipy.signal import cheby2, sosfilt

from wdf.filtering import sosfiltfilt
from wdf.processes.lines import notch_sections


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


def settling_length(sos, sampling, floor=1e-12, limit_s=120.0):
    """How many samples the filter needs before its response has decayed.

    Measured from the impulse response rather than assumed from the order: a
    steep filter close to Nyquist rings far longer than its order suggests, and
    a narrow notch longer still, since its poles sit next to the unit circle.
    This length is the real data a block needs on each side.

    The floor is set by what happens downstream, not by what looks negligible
    here. Whitening against a high-order autoregressive model applies its
    largest gain at the band edges, which is exactly where the conditioning
    residual lives, so a residual small enough to ignore in the conditioned data
    can still dominate the block edges once whitened, where it reads as a short
    broadband burst. The default is therefore well below what the conditioned
    data alone would justify.

    The response is followed for as long as it takes rather than over a fixed
    window: it is computed over twice the length at which it was last above the
    floor, so the stretch examined past the settling is always at least as long
    as the settling itself, and a stable filter, whose response is a sum of
    decaying modes, does not come back above the floor after staying below it
    that long. A filter still ringing at `limit_s` is refused rather than
    reported at the limit: a settling cut short puts the unsettled transient
    into every block the filter emits.

    :type sos: numpy.ndarray
    :param sos: second-order sections.
    :type sampling: float
    :param sampling: sampling frequency, Hz.
    :type floor: float
    :param floor: fraction of the peak below which the response is spent.
    :type limit_s: float
    :param limit_s: longest settling accepted, seconds.
    :return: int -- samples until the response has decayed below `floor`.
    :raises ValueError: if the response is still above `floor` after
        `limit_s` seconds.
    """
    sos = np.asarray(sos, dtype=float)
    limit = int(np.ceil(limit_s * sampling))
    state = np.zeros((sos.shape[0], 2))
    response = np.zeros(0)
    chunk = np.zeros(max(1, int(sampling)))
    chunk[0] = 1.0
    settled = 1
    while 2 * settled > len(response) and len(response) < 2 * limit:
        filtered, state = sosfilt(sos, chunk, zi=state)
        response = np.concatenate([response, np.abs(filtered)])
        chunk = np.zeros(len(response))
        peak = response.max()
        if peak <= 0.0:
            return 1
        above = np.flatnonzero(response > floor * peak)
        settled = int(above[-1]) + 1 if above.size else 1
    if settled > limit or 2 * settled > len(response):
        raise ValueError(
            f"the filter still rings above {floor:g} of its peak after "
            f"{limit_s:g} s: a notch this narrow, or a band edge this steep, "
            f"needs more real data on each side of a block than the settling "
            f"limit allows. Widen the filter or raise the limit")
    return settled


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
        `Parameters.LineNotches`, when present, lists the detector's lines as
        `(frequency, bandwidth, height)` rows, as
        `wdf.processes.lines.spectral_lines` returns them; their
        `notch_sections` are stacked in front of the band-pass sections.

        :type padlen: int
        :padlen: samples of real data each filter pass settles over before it
            reaches the stretch being emitted: real past for the forward pass,
            real future for the backward one. Measured from the impulse response
            when None. It may exceed the read block: a block is held until that
            much of what follows it has been read.
        :raises ValueError: if the filter does not settle within the limit
            `settling_length` accepts.
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
        self.bandpass_sos = cheby2(self.order, self.stopband_attenuation_db,
                                   [self.low_freq_hp, self.cutoff_frequency],
                                   fs=self.sampling, btype='bandpass', output='sos')
        # The detector's lines, cut down to the floor before the band is
        # restricted. The cascade is linear, so the order of the sections
        # changes nothing but the rounding; they are stacked in the order of the
        # chain they implement, and every pass of the filter, streamed or not,
        # applies all of them.
        lines = getattr(Parameters, "LineNotches", None)
        self.lines = np.asarray([] if lines is None else lines,
                                dtype=float).reshape(-1, 3)
        self.sos = np.vstack([notch_sections(self.lines, self.sampling),
                              self.bandpass_sos])
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
            "%.0f dB, %d notches, settling %d samples (%.3f s)",
            self.sampling, self.resampling, self.low_freq_hp,
            self.cutoff_frequency, self.order, self.stopband_attenuation_db,
            len(self.lines), self.padlen, self.padlen / self.sampling)

    @property
    def line_band(self):
        """The band a detector's lines are searched in, `(low, high)` in Hz.

        Every frequency whose content can reach the analysed stream: from the
        band-pass's low edge up to the frequency that folds onto its high edge
        when the stream is decimated. The band-pass's upper stop band is inside
        the analysed band, and the whitening lifts what is left there back up,
        so a line in it, or folded into it, reaches the search.
        """
        return (self.low_freq_hp, self.resampling - self.cutoff_frequency)

    def Process(self, data):
        """
        The method for the downsampling the data.

        With `estimation=True` the block is taken as complete in itself and is
        band-passed with `sosfiltfilt` and decimated in one shot. Its first and
        last `padlen` samples then carry the filter's start rather than the
        data, so a stretch whose edges matter, such as the one the noise model
        is fitted on, is read with its context and conditioned by
        `condition_stretch` instead.

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

    def condition_stretch(self, data, context):
        """Band-pass and decimate a stretch read with real data on each side.

        `data` holds the stretch and, before and after it, `context` samples of
        the real data around it. The whole read is filtered with zero phase and
        only the stretch between the two contexts is kept, so each pass of the
        filter has run over `context` samples of real data before it reaches the
        stretch: with `context` at least `padlen`, what is kept is what filtering
        the whole stream at once gives there, to the floor `padlen` was measured
        at. `Process` in estimation mode filters a stretch alone, and its first
        and last `padlen` samples carry the filter's start instead.

        :type data: py4tsa.tsa.SeqView_double_t
        :param data: the stretch with its context, at the original rate.
        :type context: int
        :param context: samples of real data before and after the stretch.
        :return: py4tsa.tsa.SeqView_double_t -- the stretch, band-passed and
            decimated, starting at the time of its own first sample.
        :raises ValueError: if `context` is shorter than `padlen`, or if the read
            holds nothing between its two contexts.
        """
        context = int(context)
        if context < self.padlen:
            raise ValueError(
                f"{context} samples of context on each side, but the filter "
                f"settles over {self.padlen}: the edges of the stretch would "
                f"carry its start")
        y = SV_to_array(data)
        if len(y) <= 2 * context:
            raise ValueError(
                f"a read of {len(y)} samples holds nothing between two contexts "
                f"of {context}")
        kept = sosfiltfilt(self.sos, y)[context:len(y) - context]
        return self._decimated_view(kept[::self.ResamplingFactor],
                                    data.GetStart() + context / self.sampling)

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
