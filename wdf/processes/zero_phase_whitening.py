"""Zero-phase AR whitening coefficients.

.. moduleauthor:: Elena Cuoco <elena.cuoco@unibo.it>

The lattice filter `ArBurgEstimator` fits whitens with magnitude ``|A|`` but
carries ``A``'s phase, which displaces the reconstructed waveform. Applying any
filter forward and then backward gives magnitude ``|B|^2`` and zero phase, so
the filter that whitens at zero phase when run in both directions is the one
whose magnitude response is the square root of ``|A|``.

That filter is fitted here as an AR model of the pseudo-spectrum ``1/|A(w)|``
and returned as a `LatticeView`, the same form the existing whitening already
consumes: the online filtering stays a time-domain lattice recursion, and the
transforms below run once per segment, next to the Burg fit itself.
"""
from __future__ import annotations

import numpy as np

from py4tsa.tsa import DoubleWhitening, LatticeView

DEFAULT_SQRT_ORDER = 256
DEFAULT_GRID = 1 << 15


def levinson(autocorrelation, order):
    """Fit an AR model to an autocorrelation sequence by Levinson-Durbin recursion.

    :type autocorrelation: numpy.ndarray
    :param autocorrelation: autocorrelation sequence, lag zero first. At least
        ``order + 1`` lags.
    :type order: int
    :param order: order of the fitted model.
    :return: the prediction polynomial with ``a[0] = 1``, the final prediction
        error, and the ``order`` reflection coefficients.
    """
    autocorrelation = np.asarray(autocorrelation, dtype=float).reshape(-1)

    if order < 1:
        raise ValueError("order must be positive")
    if autocorrelation.size < order + 1:
        raise ValueError(
            f"need {order + 1} autocorrelation lags, got {autocorrelation.size}"
        )
    if autocorrelation[0] <= 0.0:
        raise ValueError("autocorrelation at lag zero must be positive")

    a = np.zeros(order + 1)
    a[0] = 1.0
    error = float(autocorrelation[0])
    reflection = np.zeros(order)

    for m in range(1, order + 1):
        accumulated = autocorrelation[m]
        if m > 1:
            accumulated += np.dot(a[1:m], autocorrelation[m - 1:0:-1])
        k = -accumulated / error
        reflection[m - 1] = k
        a[1:m + 1] = a[1:m + 1] + k * a[m - 1::-1][:m]
        error *= (1.0 - k * k)

    return a, error, reflection


def sqrt_ar_polynomial(ar, order=DEFAULT_SQRT_ORDER, grid=DEFAULT_GRID):
    """Fit the prediction polynomial whose magnitude response is ``|A|^(1/2)``.

    Applied forward and then backward this polynomial whitens by ``|A|`` at
    zero phase. ``|A|`` is far smoother than ``|A|^2``, so ``order`` can be an
    order of magnitude below the order of the model it is derived from.

    :type ar: numpy.ndarray
    :param ar: AR coefficients as `ArBurgEstimator` holds them -- the noise
        scale in ``ar[0]`` and the prediction coefficients in ``ar[1:]``, for
        ``A(z) = 1 - sum_k ar[k] z^-k``.
    :type order: int
    :param order: order of the fitted square-root model.
    :type grid: int
    :param grid: FFT length the response is evaluated on.
    :return: the prediction polynomial with ``a[0] = 1``, the final prediction
        error, and the reflection coefficients.
    """
    ar = np.asarray(ar, dtype=float).reshape(-1)

    if ar.size < 2:
        raise ValueError("ar must hold a noise scale and at least one coefficient")
    if grid < 2 * ar.size:
        raise ValueError(f"grid {grid} is too short for an order {ar.size - 1} model")

    polynomial = np.concatenate([[1.0], -ar[1:]])
    response = np.abs(np.fft.rfft(polynomial, grid))

    if np.any(response <= 0.0):
        raise ValueError("AR model has a zero on the unit circle")

    autocorrelation = np.fft.irfft(1.0 / response, grid)

    return levinson(autocorrelation, order)


def _both_ways(polynomial, samples):
    """The filter applied forward and then backward, as the lattice runs it."""
    from scipy.signal import lfilter

    forward = lfilter(polynomial, [1.0], np.asarray(samples, dtype=float))
    return lfilter(polynomial, [1.0], forward[::-1])[::-1]


def held_outside(freq, psd, band):
    """A measured spectrum inside `band`, held at its edge values outside it.

    The conditioning leaves a stop band 120 dB down below its low edge and
    another above its high one. A fit that weighs relative error -- which is
    the point of fitting a measured spectrum rather than an autoregressive
    model of it -- would spend its order on those cliffs. Holding the spectrum
    flat outside the band says instead: do not whiten there.

    :type freq: numpy.ndarray
    :param freq: frequencies of `psd`, hertz, ascending.
    :type psd: numpy.ndarray
    :param psd: power spectral density at those frequencies.
    :type band: tuple
    :param band: ``(low, high)`` in hertz, the band to keep.
    :return: numpy.ndarray -- the spectrum, flat outside `band`.
    """
    held = np.array(psd, dtype=float)
    low = int(np.searchsorted(freq, band[0]))
    high = min(int(np.searchsorted(freq, band[1])), len(held) - 1)
    if not 0 <= low < high:
        raise ValueError(f"band {band} does not fall inside the spectrum")
    held[:low] = held[low]
    held[high:] = held[high]
    return held


def sqrt_polynomial_from_spectrum(freq, psd, order, grid=DEFAULT_GRID, band=None):
    """Fit the square-root filter to a measured spectrum rather than to a model.

    `sqrt_ar_polynomial` is Levinson on the autocorrelation of ``1/|A|``, and
    ``1/|A|`` is the square root of the autoregressive model's own spectrum. So
    that function already is "fit a filter whose forward-backward response is
    the square root of this spectrum", and handing it a measured spectrum is a
    change of input rather than of method. What it removes is the Burg fit that
    produced the model: one fit instead of two, and an error weighed in decibels
    across the band instead of in absolute power, which is dominated by
    whichever octave carries the most of it.

    Measured on O4b strain conditioned above 6 Hz, the autoregressive path
    leaves the whitened spectrum a factor 2.9 low at 8-32 Hz in H1 and L1 --
    Burg has no incentive to fit a region 60 dB down -- while this path is flat
    to a tenth in every octave from 8 Hz to Nyquist.

    :type freq: numpy.ndarray
    :param freq: frequencies of `psd`, hertz, ascending.
    :type psd: numpy.ndarray
    :param psd: power spectral density, as `scipy.signal.welch` returns it.
    :type order: int
    :param order: order of the fitted filter, which is also its latency.
    :type grid: int
    :param grid: FFT length the spectrum is interpolated onto.
    :type band: tuple or None
    :param band: ``(low, high)`` outside which the spectrum is held flat; the
        whole spectrum is used when None.
    :return: the prediction polynomial with ``a[0] = 1``, the final prediction
        error, and the reflection coefficients.
    :raises ValueError: if the spectrum is not positive where it is fitted.
    """
    freq = np.asarray(freq, dtype=float).reshape(-1)
    psd = np.asarray(psd, dtype=float).reshape(-1)
    if freq.size != psd.size:
        raise ValueError("the spectrum and its frequencies differ in length")
    if band is not None:
        psd = held_outside(freq, psd, band)
    if np.any(psd <= 0.0):
        raise ValueError("the spectrum is not positive everywhere it is fitted")

    sampling = 2.0 * freq[-1]
    grid_freq = np.fft.rfftfreq(int(grid), 1.0 / sampling)
    amplitude = np.sqrt(np.interp(grid_freq, freq, psd))
    autocorrelation = np.fft.irfft(amplitude, int(grid))

    return levinson(autocorrelation, int(order))


def sqrt_lattice_view(ar, order=DEFAULT_SQRT_ORDER, grid=DEFAULT_GRID):
    """Build the `LatticeView` that whitens at zero phase when run both ways.

    The returned view drives the existing `LatticeFilter`/`DoubleWhitening`
    unchanged. Feeding it to a forward-backward pass whitens the data by
    ``|A|`` instead of ``|A|^2``, so the output is flat and unit variance on
    its own noise scale rather than divided by the noise power spectrum.

    The whitened output has standard deviation ``error * ar[0]``, with ``error``
    the final prediction error returned by `sqrt_ar_polynomial`.

    :type ar: numpy.ndarray
    :param ar: AR coefficients as `ArBurgEstimator` holds them (see
        `sqrt_ar_polynomial`).
    :type order: int
    :param order: order of the fitted square-root model.
    :type grid: int
    :param grid: FFT length the response is evaluated on.
    :return: py4tsa.tsa.LatticeView -- reflection coefficients of the
        square-root filter.
    """
    _, error, reflection = sqrt_ar_polynomial(ar, order=order, grid=grid)
    return lattice_view(reflection, float(np.asarray(ar, dtype=float)[0]))


def lattice_view(reflection, scale=1.0):
    """The `LatticeView` a sequence of reflection coefficients defines.

    :type reflection: numpy.ndarray
    :param reflection: reflection coefficients, finest stage first.
    :type scale: float
    :param scale: the prediction error the running error starts from.
    :return: py4tsa.tsa.LatticeView
    """
    reflection = np.asarray(reflection, dtype=float).reshape(-1)
    order = len(reflection)
    view = LatticeView(order)
    view.SetOrder(order)

    running_error = float(scale)
    for j, k in enumerate(reflection):
        view.SetParcorF(j + 1, float(-k))
        view.SetParcorB(j + 1, float(-k))
        running_error *= (1.0 - float(k) * float(k))
        view.SetErrorForward(j, running_error)
        view.SetErrorBackward(j, running_error)

    return view


class ZeroPhaseWhitening(object):
    """Whiten a stream by the square root of its noise spectrum, at zero phase.

    The lattice filter is run forward and then backward, which makes the overall
    response the squared magnitude of the filter and its phase identically zero.
    Built from the square-root model, that response is ``|A|``: the data comes
    out divided by the square root of its power spectrum, flat and unit variance
    on the noise scale, with every transient left where the data put it.

    Filtering is a time-domain lattice recursion on the stream. The transforms
    that build the square-root model run once, here, in the constructor. The
    backward pass reads ``order`` samples ahead, which is the whole latency of
    the operation.
    """

    def __init__(self, ar, output_size, extra_size=0,
                 order=DEFAULT_SQRT_ORDER, grid=DEFAULT_GRID):
        """
        :type ar: numpy.ndarray
        :param ar: AR coefficients as `ArBurgEstimator` holds them -- the noise
            scale in ``ar[0]`` and the prediction coefficients in ``ar[1:]``.
        :type output_size: int
        :param output_size: number of whitened samples produced per `Process` call.
        :type extra_size: int
        :param extra_size: lookahead buffer, in samples. It must be at least
            ``order``, which is what the backward pass reads ahead; a smaller
            positive value is refused rather than silently truncated. Zero means
            the lookahead is supplied later through `SetOutputSize`, which is
            how the worker primes the buffer.
        :raises ValueError: if `extra_size` is positive and below ``order``.
        :type order: int
        :param order: order of the square-root model.
        :type grid: int
        :param grid: FFT length the response is evaluated on.
        """
        polynomial, error, reflection = sqrt_ar_polynomial(
            ar, order=order, grid=grid)
        scale = float(np.asarray(ar, dtype=float)[0])
        self._install(polynomial, error, reflection, order, scale * error,
                      scale, output_size, extra_size)

    def _install(self, polynomial, error, reflection, order, sigma, scale,
                 output_size, extra_size):
        """Hold the fitted filter and build the lattice that runs it.

        Shared by the two ways of fitting it -- from an autoregressive model,
        and from a measured spectrum -- so that the two differ in the fit and
        in nothing else.

        :raises ValueError: if `extra_size` is positive and below `order`.
        """
        self.order = int(order)
        self.polynomial, self.error, self.reflection = polynomial, error, reflection
        self.sigma = float(sigma)
        self.LV = lattice_view(reflection, scale)

        if 0 < extra_size < self.order:
            raise ValueError(
                f"the lookahead is {extra_size} samples but the backward pass "
                f"reads {self.order} ahead: it would be truncated at every "
                f"block join. Set WhiteningExtraSize to at least "
                f"SqrtWhiteningOrder.")

        self.filter = DoubleWhitening(self.LV, output_size, extra_size)
        self.filter.init(self.LV)

    @classmethod
    def from_spectrum(cls, samples, sampling, output_size, extra_size=0,
                      order=DEFAULT_SQRT_ORDER, grid=DEFAULT_GRID, band=None,
                      nperseg=8192, average="median"):
        """Build the whitening from the spectrum of a stretch, without Burg.

        The stretch is the one the model would have been fitted on. Its
        spectrum is measured, held flat outside `band`, and the filter is
        fitted to it by `sqrt_polynomial_from_spectrum`; the noise scale is
        then read on that same stretch, whitened, as the robust scale of the
        result, which is the statistic every stage downstream uses.

        `average` is how the periodograms are combined: the median is the
        default because a transient in the stretch moves it far less than it
        moves the mean, and an autoregressive fit has no such defence.

        :type samples: numpy.ndarray
        :param samples: the conditioned stretch to fit on.
        :type sampling: float
        :param sampling: its sampling frequency, hertz.
        :type output_size: int
        :param output_size: whitened samples produced per `Process` call.
        :type extra_size: int
        :param extra_size: lookahead buffer, in samples.
        :type order: int
        :param order: order of the fitted filter, which is also its latency.
        :type grid: int
        :param grid: FFT length the spectrum is interpolated onto.
        :type band: tuple or None
        :param band: ``(low, high)`` outside which the spectrum is held flat.
        :type nperseg: int
        :param nperseg: segment length of the spectral estimate.
        :type average: str
        :param average: how the periodograms are combined, "median" or "mean".
        :return: ZeroPhaseWhitening
        :raises ValueError: if the stretch is shorter than one segment.
        """
        from scipy.signal import welch

        samples = np.asarray(samples, dtype=float).reshape(-1)
        if samples.size < nperseg:
            raise ValueError(
                f"the stretch holds {samples.size} samples, fewer than the "
                f"{nperseg} one segment of the spectral estimate needs")

        freq, psd = welch(samples, fs=float(sampling), nperseg=int(nperseg),
                          average=average)
        polynomial, error, reflection = sqrt_polynomial_from_spectrum(
            freq, psd, order, grid=grid, band=band)

        whitened = _both_ways(polynomial, samples)
        edge = min(int(order), whitened.size // 4)
        inside = whitened[edge:whitened.size - edge] if edge else whitened
        sigma = float(np.median(np.abs(inside)) / 0.6745)

        self = cls.__new__(cls)
        self._install(polynomial, error, reflection, order, sigma, 1.0,
                      output_size, extra_size)
        return self

    @property
    def latency(self):
        """Samples of lookahead the backward pass needs; the whole latency."""
        return self.order

    def Process(self, data, dataw):
        """Whiten one chunk, blocking until a full output block is available.

        :type data: py4tsa.tsa.SeqView_double_t
        :param data: input chunk, band-passed and decimated.
        :type dataw: py4tsa.tsa.SeqView_double_t
        :param dataw: output sequence view, filled in place.
        :return: None
        """
        self.filter(data, dataw)

    def Input(self, data):
        """Feed one chunk in without reading output.

        :type data: py4tsa.tsa.SeqView_double_t
        :param data: input chunk.
        :return: None
        """
        self.filter.Input(data)

    def Output(self, dataw):
        """Read whatever output is available.

        :type dataw: py4tsa.tsa.SeqView_double_t
        :param dataw: output sequence view, filled in place.
        :return: None
        """
        self.filter.Output(dataw)

    def DataNeeded(self):
        """How many more input samples are needed before output is available.

        Zero or less means a whitened block can be read out without feeding
        anything in, which is what tells a drain when the buffer is empty.

        :return: int -- samples still needed.
        """
        return int(self.filter.GetDataNeeded())

    def SetOutputSize(self, output_size, extra_size):
        """Change the output block size and the lookahead.

        :type output_size: int
        :param output_size: whitened samples produced per `Process` call.
        :type extra_size: int
        :param extra_size: lookahead buffer, in samples.
        :return: None
        """
        if 0 < extra_size < self.order:
            raise ValueError(
                f"the lookahead is {extra_size} samples but the backward pass "
                f"reads {self.order} ahead: it would be truncated at every "
                f"block join. Set WhiteningExtraSize to at least "
                f"SqrtWhiteningOrder.")
        self.filter.SetOutputSize(output_size, extra_size)
