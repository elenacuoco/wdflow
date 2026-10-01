"""Zero-phase AR whitening.

.. moduleauthor:: Elena Cuoco <elena.cuoco@unibo.it>

The lattice filter `ArBurgEstimator` fits whitens with magnitude ``|A|`` but
carries ``A``'s phase, which displaces the reconstructed waveform. Two filters
here remove the same colour at zero phase.

`MagnitudeWhitening`, which the worker runs when asked for it (`ZeroPhaseFilter
= "magnitude"`), is the filter whose frequency response *is* ``|A(e^{iw})|``. A
real, non-negative response has zero phase exactly, so nothing is fitted and
nothing is approximated in the band: the whitened spectrum is the causal one,
bin by bin. Its impulse response is symmetric and, when ``A`` has zeros close
to the unit circle -- the narrow lines of a detector -- long; it is measured
from the response itself, like the settling of the band-pass, and applied by
FFT convolution with real past and future data on each side of the block.

Given a duration, `MagnitudeWhitening` is instead the inverse-spectrum
truncation gwpy whitens with (`truncated_taps`): the model's spectrum at the
resolution of the filter, inverted, held, and Hann-truncated to that duration,
read half of it ahead.

`ZeroPhaseWhitening` is the worker's default. Applying any filter forward and
then backward gives magnitude ``|B|^2`` and
zero phase, so the filter that whitens at zero phase when run in both
directions is the one whose magnitude response is the square root of ``|A|``.
That filter is fitted as an AR model of the pseudo-spectrum ``1/|A(w)|`` and
returned as a `LatticeView`, the form the lattice recursion consumes. The fit
is an approximation of ``|A|`` at a finite order, and where ``A`` has deep
narrow zeros the root does not follow them.

Both whiten the band the conditioning passes and nothing else. The model is
fitted on band-passed data, so ``|A|`` also inverts the band-pass's stop band,
a gain of up to 1e6 where there is nothing to whiten; that gain is what makes
the magnitude filter's support hundreds of seconds and the root's fit ripple
in the band. Given the passband, both build their target from the held
modulus `held_modulus`: ``|A|`` inside, its edge values outside, joined by a
raised cosine a fixed number of hertz wide. The stop band stays the band-pass's
alone; the whitening does not undo it.
"""
from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from py4tsa.tsa import DoubleWhitening, LatticeView

#: Floor for the square-root order when the caller does not state one. The
#: order used is this or the order of the model, whichever is larger: a root
#: far below the model it is taken of cannot follow it, and the error is paid
#: twice, since the filter runs both ways and the response is the square of its
#: magnitude.
DEFAULT_SQRT_ORDER = 256
DEFAULT_GRID = 1 << 15
#: Width in hertz of the raised cosine that joins the held modulus to ``|A|``
#: inside each passband edge (`held_modulus`): wide enough that the target
#: has no corner, whose impulse response would decay slowly, and narrow
#: enough to leave the band ``|A|`` itself.
DEFAULT_BAND_BLEND_HZ = 1.0


def _order_for(ar, order):
    """The square-root order: the floor, or the model's own, whichever is more.

    Stating an order overrides both. The latency of the filter is its order.
    """
    if order is not None:
        return int(order)
    return max(DEFAULT_SQRT_ORDER, len(np.asarray(ar).reshape(-1)) - 1)


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


def sqrt_ar_polynomial(ar, order=None, grid=DEFAULT_GRID, band=None,
                       sampling=None, blend=DEFAULT_BAND_BLEND_HZ):
    """Fit the prediction polynomial whose magnitude response is ``|A|^(1/2)``.

    Applied forward and then backward this polynomial whitens by ``|A|`` at
    zero phase. ``|A|`` is smoother than ``|A|^2``, but not smooth enough to be
    fitted at an order well below the model's: the narrow lines of the model
    need an order comparable to its own.

    :type ar: numpy.ndarray
    :param ar: AR coefficients as `ArBurgEstimator` holds them -- the noise
        scale in ``ar[0]`` and the prediction coefficients in ``ar[1:]``, for
        ``A(z) = 1 - sum_k ar[k] z^-k``.
    :type order: int or None
    :param order: order of the fitted square-root model. ``None`` takes the
        order of ``ar`` itself.
    :type grid: int
    :param grid: FFT length the response is evaluated on.
    :type band: tuple or None
    :param band: ``(f_lo, f_hi)`` in hertz: the root is fitted to the held
        modulus `held_modulus` of ``|A|`` instead of ``|A|`` itself, so that
        the forward-backward response follows ``|A|`` in the band and is flat
        outside it. None fits the full band.
    :type sampling: float or None
    :param sampling: sampling frequency of the model, hertz; needed with `band`.
    :type blend: float
    :param blend: width of the raised cosine inside each edge, hertz.
    :return: the prediction polynomial with ``a[0] = 1``, the final prediction
        error, and the reflection coefficients. The response of the polynomial
        is ``|B|^2 ~ error * G``, ``G`` the target.
    """
    ar = np.asarray(ar, dtype=float).reshape(-1)
    order = _order_for(ar, order)

    if ar.size < 2:
        raise ValueError("ar must hold a noise scale and at least one coefficient")
    if grid < 2 * ar.size:
        raise ValueError(f"grid {grid} is too short for an order {ar.size - 1} model")

    _, response = held_response(ar, grid, band, sampling, blend)

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


def held_modulus(freq, modulus, band, blend=DEFAULT_BAND_BLEND_HZ):
    """A whitening gain inside `band`, held at its edge values outside it, no corners.

    The one target every zero-phase whitening here is built from, whatever it
    is measured on: ``|A|`` of an autoregressive model, or ``1/sqrt(S)`` of a
    measured spectrum. Outside the band it is `held_outside`'s edge value, so
    the conditioning's stop band is not whitened. Inside each edge, over
    `blend` hertz, it rises from the held value to `modulus` along a raised
    cosine, so the target is continuous with a continuous slope at the edge
    rather than cornered there.

    :type freq: numpy.ndarray
    :param freq: frequencies of `modulus`, hertz, ascending.
    :type modulus: numpy.ndarray
    :param modulus: the gain at those frequencies.
    :type band: tuple
    :param band: ``(f_lo, f_hi)`` in hertz, the band the gain is kept in.
    :type blend: float
    :param blend: width of each raised cosine, hertz; zero for none.
    :return: numpy.ndarray -- the held gain ``G``, ``modulus`` on
        ``[f_lo + blend, f_hi - blend]``.
    :raises ValueError: if the band is not inside the frequencies, or the two
        blends would overlap.
    """
    freq = np.asarray(freq, dtype=float).reshape(-1)
    modulus = np.asarray(modulus, dtype=float).reshape(-1)
    low, high = float(band[0]), float(band[1])
    blend = float(blend)
    if blend < 0.0 or 2.0 * blend > high - low:
        raise ValueError(f"a blend of {blend} Hz does not fit in the band {band}")
    held = held_outside(freq, modulus, band)
    if blend > 0.0:
        edge_low, edge_high = held[0], held[-1]
        rising = (freq >= low) & (freq < low + blend)
        weight = 0.5 * (1.0 - np.cos(np.pi * (freq[rising] - low) / blend))
        held[rising] = edge_low + weight * (modulus[rising] - edge_low)
        falling = (freq > high - blend) & (freq <= high)
        weight = 0.5 * (1.0 - np.cos(np.pi * (high - freq[falling]) / blend))
        held[falling] = edge_high + weight * (modulus[falling] - edge_high)
    return held


def held_response(ar, grid, band=None, sampling=None, blend=DEFAULT_BAND_BLEND_HZ):
    """``|A|`` of a model on an FFT grid, and the target the whitening applies.

    :type ar: numpy.ndarray
    :param ar: AR coefficients as `ArBurgEstimator` holds them.
    :type grid: int
    :param grid: FFT length ``|A|`` is sampled on.
    :type band: tuple or None
    :param band: ``(f_lo, f_hi)`` in hertz; None for the full band, where the
        target is ``|A|`` itself.
    :type sampling: float or None
    :param sampling: sampling frequency of the model, hertz; needed with `band`.
    :type blend: float
    :param blend: width of the raised cosine inside each edge, hertz.
    :return: tuple -- ``|A|`` and the target ``G`` on the ``grid // 2 + 1``
        non-negative frequencies.
    :raises ValueError: if `band` is given without `sampling`.
    """
    modulus = np.abs(np.fft.rfft(prediction_error_polynomial(ar), int(grid)))
    if band is None:
        return modulus, modulus
    if sampling is None:
        raise ValueError("a band in hertz needs the sampling frequency")
    freq = np.fft.rfftfreq(int(grid), 1.0 / float(sampling))
    return modulus, held_modulus(freq, modulus, band, blend)


def in_band_scale(scale, response, modulus, freq=None, band=None):
    """The level of the whitened output in band, on noise the model describes.

    The model's spectrum is ``scale**2 / |A|**2``; through a zero-phase filter
    of response `response` it becomes ``scale**2 (response / |A|)**2``. The
    scale returned is the square root of the median of that over the band,
    in units of white noise: the output divided by it has unit density in
    band, whatever the stop band holds and whatever lines the filter leaves.

    It is the level in band and not the variance over the circle. Held, the
    stream carries power in the passband alone, and the variance over the
    circle is the band's share of it: dividing by that put the whitened level
    in band at ``1/sqrt(share)``, 1.17 on GW150914 with a 16-744 Hz passband at
    2048 Hz, where it has to be 1. The median, and not the mean, so that a
    line the filter does not follow sets nothing: on the band the response
    is ``|A|``, the ratio is 1 and the median is exact.

    :type scale: float
    :param scale: the model's noise scale, ``ar[0]``.
    :type response: numpy.ndarray
    :param response: the response the filter applies, on the non-negative
        frequencies of an FFT grid, zero to Nyquist.
    :type modulus: numpy.ndarray
    :param modulus: ``|A|`` on the same frequencies.
    :type freq: numpy.ndarray or None
    :param freq: those frequencies, hertz; needed with `band`.
    :type band: tuple or None
    :param band: ``(f_lo, f_hi)`` in hertz, the band the level is read on;
        None for the whole grid.
    :return: float
    """
    ratio = (np.asarray(response, dtype=float) / np.asarray(modulus, dtype=float)) ** 2
    if band is not None:
        freq = np.asarray(freq, dtype=float)
        ratio = ratio[(freq >= float(band[0])) & (freq <= float(band[1]))]
    return float(scale) * float(np.sqrt(np.median(ratio)))


def whitened_level(whitened, sampling, band=None, nperseg=8192):
    """The level in band of a whitened stretch, read on the stretch itself.

    The median over the band of its median-averaged spectrum, in units of the
    density ``2 / sampling`` of unit-variance white noise, square-rooted: the
    measured counterpart of `in_band_scale`, for the filters fitted to a
    measured spectrum. The robust scale of the samples in time is not it: a
    line the filter leaves raises it, and on GW150914 L1 it read 1.68 on a
    stretch whose spectrum is 1.00 in band, which put the whitened level at
    0.6.

    :type whitened: numpy.ndarray
    :param whitened: the whitened stretch.
    :type sampling: float
    :param sampling: its sampling frequency, hertz.
    :type band: tuple or None
    :param band: ``(f_lo, f_hi)`` in hertz; None for zero to Nyquist.
    :type nperseg: int
    :param nperseg: segment length of the spectral estimate.
    :return: float
    """
    from scipy.signal import welch

    whitened = np.asarray(whitened, dtype=float).reshape(-1)
    freq, psd = welch(whitened, fs=float(sampling),
                      nperseg=int(min(nperseg, whitened.size)), average="median")
    if band is not None:
        psd = psd[(freq >= float(band[0])) & (freq <= float(band[1]))]
    return float(np.sqrt(np.median(psd) * float(sampling) / 2.0))


def sqrt_polynomial_from_spectrum(freq, psd, order, grid=DEFAULT_GRID, band=None,
                                  blend=DEFAULT_BAND_BLEND_HZ):
    """Fit the square-root filter to a measured spectrum rather than to a model.

    `sqrt_ar_polynomial` is Levinson on the autocorrelation of ``1/|A|``, and
    ``1/|A|`` is the square root of the autoregressive model's own spectrum. So
    that function already is "fit a filter whose forward-backward response is
    the square root of this spectrum", and handing it a measured spectrum is a
    change of input rather than of method. What it removes is the Burg fit that
    produced the model: one fit instead of two, and an error weighed in decibels
    across the band instead of in absolute power, which is dominated by
    whichever octave carries the most of it, so that a region tens of decibels
    below the rest of the band is fitted as closely as the rest.

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
    :type blend: float
    :param blend: width of the raised cosine inside each edge of `band`,
        hertz (`held_modulus`).
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
    if band is not None:
        amplitude = 1.0 / held_modulus(grid_freq, 1.0 / amplitude, band, blend)
    autocorrelation = np.fft.irfft(amplitude, int(grid))

    return levinson(autocorrelation, int(order))


def sqrt_lattice_view(ar, order=None, grid=DEFAULT_GRID):

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
    :type order: int or None
    :param order: order of the fitted square-root model. ``None`` takes the
        order of ``ar`` itself.
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
                 order=None, grid=DEFAULT_GRID, band=None, sampling=None,
                 blend=DEFAULT_BAND_BLEND_HZ):
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
        :type band: tuple or None
        :param band: ``(f_lo, f_hi)`` in hertz, the passband of the
            conditioning: the root is fitted to the held modulus
            (`held_modulus`) and `sigma` is the in-band scale of the response
            obtained (`in_band_scale`). None fits ``|A|`` over the full band,
            with `sigma` the model's ``ar[0] * error``.
        :type sampling: float or None
        :param sampling: sampling frequency of the model, hertz; needed with `band`.
        :type blend: float
        :param blend: width of the raised cosine inside each edge, hertz.
        """
        order = _order_for(ar, order)
        polynomial, error, reflection = sqrt_ar_polynomial(
            ar, order=order, grid=grid, band=band, sampling=sampling, blend=blend)
        scale = float(np.asarray(ar, dtype=float)[0])
        if band is None:
            sigma = scale * error
        else:
            # The response the lattice applies both ways is |B|^2, measured
            # here rather than taken as the target it approximates.
            modulus, _ = held_response(ar, grid)
            applied = np.abs(np.fft.rfft(polynomial, int(grid))) ** 2
            sigma = in_band_scale(scale, applied, modulus,
                                  np.fft.rfftfreq(int(grid), 1.0 / float(sampling)), band)
        self.band = None if band is None else (float(band[0]), float(band[1]))
        self._install(polynomial, error, reflection, order, sigma,
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
                      nperseg=8192, average="median", blend=DEFAULT_BAND_BLEND_HZ):
        """Build the whitening from the spectrum of a stretch, without Burg.

        The stretch is the one the model would have been fitted on. Its
        spectrum is measured, held flat outside `band`, and the filter is
        fitted to it by `sqrt_polynomial_from_spectrum`; the noise scale is
        then read on that same stretch, whitened, as its level in band
        (`whitened_level`).

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
            freq, psd, order, grid=grid, band=band, blend=blend)

        whitened = _both_ways(polynomial, samples)
        edge = min(int(order), whitened.size // 4)
        inside = whitened[edge:whitened.size - edge] if edge else whitened
        sigma = whitened_level(inside, sampling, band, nperseg)

        self = cls.__new__(cls)
        self.band = None if band is None else (float(band[0]), float(band[1]))
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
        """Buffered samples beyond what the next output block needs.

        `DoubleWhitening::GetDataNeeded`: the samples buffered minus the output
        size plus the lookahead. Zero or more means a whitened block can be read
        out without feeding anything in; negative means it cannot.

        :return: int
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


#: Fraction of the impulse response's peak below which its tail is spent. The
#: tail beyond this is what the truncation removes from the response, and it
#: matters where the conditioned data are loud: at a narrow line the whitened
#: output is the line's amplitude times the response's error there, so the
#: floor is set by the dynamic range of the lines, not by what looks negligible
#: in the response alone. Near a zero of ``A`` on the unit circle ``|A|``
#: has a corner rather than a smooth minimum, and the Fourier coefficients of
#: a corner fall as the inverse square of the lag: the support grows about as the
#: inverse square root of the floor, and the error left at the line falls about
#: as the inverse of the support. The floor is therefore a trade between the
#: residual of the narrowest lines and the latency, which is the support; it
#: is an empirical choice, to be validated on the detector's own lines.
DEFAULT_RESPONSE_FLOOR = 1e-8
#: FFT length on which ``|A|`` is sampled to obtain its impulse response. The
#: response found is one period of the true one; the support is refused when it
#: reaches a quarter of this length, so the period is never what ends it.
DEFAULT_RESPONSE_GRID = 1 << 23
#: Fraction of the kept support over which its two ends are tapered, so that
#: truncating a tail already below the floor does not leave a step.
RESPONSE_TAPER = 0.1
#: Length in seconds of the truncated filter (`truncated_taps`), gwpy's
#: ``fduration`` default: the whitening reads half of it ahead.
DEFAULT_TRUNCATION_S = 4.0


def prediction_error_polynomial(ar):
    """The prediction-error polynomial ``A`` of an `ArBurgEstimator` model.

    :type ar: numpy.ndarray
    :param ar: AR coefficients as `ArBurgEstimator` holds them -- the noise
        scale in ``ar[0]`` and the prediction coefficients in ``ar[1:]``.
    :return: numpy.ndarray -- ``[1, -ar[1], ..., -ar[p]]``, the taps of
        ``A(z) = 1 - sum_k ar[k] z^-k``.
    :raises ValueError: if `ar` holds no prediction coefficient.
    """
    ar = np.asarray(ar, dtype=float).reshape(-1)
    if ar.size < 2:
        raise ValueError("ar must hold a noise scale and at least one coefficient")
    return np.concatenate([[1.0], -ar[1:]])


def response_support(impulse, floor=DEFAULT_RESPONSE_FLOOR):
    """Half-length of a symmetric impulse response, measured on the response.

    The length is not read off the model's order: a zero of ``A`` at radius
    ``r`` from the origin contributes a term decaying as ``r^n``, so a zero
    close to the unit circle -- a narrow line -- makes ``|A|`` ring for many
    times the order. The support is the last lag at which the response is still
    above `floor` of its peak.

    :type impulse: numpy.ndarray
    :param impulse: one period of a real, even impulse response, lag zero
        first, as `numpy.fft.irfft` of a real response returns it.
    :type floor: float
    :param floor: fraction of the peak below which the response is spent.
    :return: int -- the support ``K`` in samples, at least 1; the taps kept are
        the lags ``-K ... K``.
    :raises ValueError: if the response has not decayed below `floor` within a
        quarter of the period, where the period would start to fold its own
        tail back onto the lags kept.
    """
    impulse = np.asarray(impulse, dtype=float).reshape(-1)
    half = np.abs(impulse[:impulse.size // 2 + 1])
    above = np.flatnonzero(half > float(floor) * half.max())
    support = int(above[-1]) if above.size else 0
    limit = impulse.size // 4
    if support >= limit:
        raise ValueError(
            f"the impulse response is still above {floor:g} of its peak at "
            f"lag {limit}, a quarter of the {impulse.size}-point grid it was "
            f"sampled on; use a longer grid or a higher floor")
    return max(support, 1)


def symmetric_taps(response, floor=DEFAULT_RESPONSE_FLOOR, support=None,
                   taper=RESPONSE_TAPER):
    """The symmetric FIR filter whose frequency response is `response`.

    A real, non-negative response has an even impulse response and zero phase.
    It is taken to the support `response_support` measures, or to the one
    stated, and its two ends are tapered over `RESPONSE_TAPER` of that support.

    :type response: numpy.ndarray
    :param response: the response on the non-negative frequencies of an FFT
        grid, ``grid // 2 + 1`` points from zero to Nyquist.
    :type floor: float
    :param floor: fraction of the peak below which the response is spent.
    :type support: int or None
    :param support: half-length in samples; None to measure it.
    :type taper: float
    :param taper: fraction of the taps inside the Tukey window's tapers;
        1 is a Hann window over the whole support, the truncation of
        `truncated_taps`.
    :return: numpy.ndarray -- ``2 K + 1`` taps, lag zero at index ``K``.
    :raises ValueError: if the stated support does not fit in the grid.
    """
    from scipy.signal.windows import tukey

    response = np.asarray(response, dtype=float).reshape(-1)
    grid = 2 * (response.size - 1)
    impulse = np.fft.irfft(response, grid)
    if support is None:
        support = response_support(impulse, floor)
    support = int(support)
    if not 0 < support < grid // 2:
        raise ValueError(f"support {support} does not fit in a {grid}-point grid")
    taps = np.concatenate([impulse[grid - support:], impulse[:support + 1]])
    taps = taps * tukey(2 * support + 1, float(taper))
    # Even to the last bit, so that the response is real and the phase zero
    # exactly, not to the rounding of the inverse transform and the taper.
    return 0.5 * (taps + taps[::-1])


def magnitude_taps(ar, floor=DEFAULT_RESPONSE_FLOOR, grid=DEFAULT_RESPONSE_GRID,
                   support=None, band=None, sampling=None, blend=DEFAULT_BAND_BLEND_HZ):
    """The zero-phase filter with response ``|A(e^{iw})|`` of an AR model.

    Applied to data whose spectrum the model describes, its output has the
    spectrum of the causal whitening ``A(z) x`` -- the same modulus -- and the
    same variance, ``ar[0]**2``, while leaving every transient at the time the
    data put it.

    :type ar: numpy.ndarray
    :param ar: AR coefficients as `ArBurgEstimator` holds them.
    :type floor: float
    :param floor: fraction of the peak below which the impulse response is spent.
    :type grid: int
    :param grid: FFT length ``|A|`` is sampled on.
    :type support: int or None
    :param support: half-length in samples; None to measure it.
    :type band: tuple or None
    :param band: ``(f_lo, f_hi)`` in hertz: the response is the held modulus
        (`held_modulus`) of ``|A|``, flat outside the band. None for ``|A|``.
    :type sampling: float or None
    :param sampling: sampling frequency of the model, hertz; needed with `band`.
    :type blend: float
    :param blend: width of the raised cosine inside each edge, hertz.
    :return: numpy.ndarray -- ``2 K + 1`` symmetric taps, lag zero at ``K``.
    :raises ValueError: if the grid is shorter than twice the model, or the
        response does not decay within it.
    """
    polynomial = prediction_error_polynomial(ar)
    if grid < 2 * polynomial.size:
        raise ValueError(f"grid {grid} is too short for an order "
                         f"{polynomial.size - 1} model")
    _, target = held_response(ar, grid, band, sampling, blend)
    return symmetric_taps(target, floor, support)


def magnitude_taps_from_spectrum(freq, psd, band=None, blend=DEFAULT_BAND_BLEND_HZ,
                                 taper=RESPONSE_TAPER):
    """The zero-phase filter with response ``1 / sqrt(S)`` of a measured spectrum.

    The same filter as `magnitude_taps`, with the measured spectrum in place of
    the model's ``ar[0]**2 / |A|**2``. The response is scaled so that noise
    with spectrum `psd` comes out white with unit variance.

    A spectrum estimated on segments of ``n`` samples resolves ``fs / n`` and
    nothing finer, so the filter it defines is the one whose transform on
    those ``n`` points is the response: ``n`` taps, the support half a
    segment. Nothing is interpolated -- an interpolated estimate has a corner
    at every bin, and its impulse response the slow tail of a corner -- and
    nothing is measured, since the support is set by the estimate itself.

    :type freq: numpy.ndarray
    :param freq: frequencies of `psd`, hertz, the non-negative frequencies of
        an FFT of even length, zero to Nyquist, as `scipy.signal.welch`
        returns them.
    :type psd: numpy.ndarray
    :param psd: one-sided power spectral density at those frequencies.
    :type band: tuple or None
    :param band: ``(low, high)`` in hertz outside which the spectrum is held
        flat (`held_outside`); the whole spectrum when None.
    :type blend: float
    :param blend: width of the raised cosine inside each edge of `band`,
        hertz (`held_modulus`).
    :type taper: float
    :param taper: Tukey fraction of the truncation (`symmetric_taps`); 1,
        a Hann window, is gwpy's ``TimeSeries.whiten``.
    :return: numpy.ndarray -- ``2 K + 1`` symmetric taps, lag zero at ``K``,
        with ``K`` one less than half the segment.
    :raises ValueError: if the spectrum is not positive where it is used, or
        its frequencies are not such a grid.
    """
    freq = np.asarray(freq, dtype=float).reshape(-1)
    psd = np.asarray(psd, dtype=float).reshape(-1)
    if freq.size != psd.size:
        raise ValueError("the spectrum and its frequencies differ in length")
    if freq.size < 3 or freq[0] != 0.0 or not np.allclose(np.diff(freq), freq[1]):
        raise ValueError("the spectrum is not on the frequencies of an FFT, "
                         "zero to Nyquist")
    if band is not None:
        psd = held_outside(freq, psd, band)
    if np.any(psd <= 0.0):
        raise ValueError("the spectrum is not positive everywhere it is used")
    sampling = 2.0 * freq[-1]
    # White noise of unit variance has the one-sided density 2 / sampling.
    response = np.sqrt(2.0 / (sampling * psd))
    if band is not None:
        response = held_modulus(freq, response, band, blend)
    return symmetric_taps(response, support=freq.size - 2, taper=taper)


def spectrum_at_resolution(ar, length, grid=DEFAULT_RESPONSE_GRID):
    """The model's spectrum as a segment of `length` samples resolves it.

    ``1/|A|**2`` averaged over frequency by the spectral window of a Hann
    segment of `length` samples: what a Welch estimate with that segment would
    measure, in expectation, on data the model describes. In lags it is the
    model's autocorrelation times the Hann window's own, normalised, which is
    zero from `length` on, so the result is exact on any grid of at least
    twice that.

    A filter truncated to `length` samples cannot follow ``|A|`` closer than
    this. Truncating ``|A|`` itself smears each narrow notch into a shallow
    one while the line it faces keeps its power, and the line comes through:
    with an order-3000 model and 4 s at 2048 Hz, GW150914 L1 whitened that
    way has a standard deviation of 4.3. The line's power spread over the
    resolution, as here, is what a filter of that length can whiten.

    :type ar: numpy.ndarray
    :param ar: AR coefficients as `ArBurgEstimator` holds them.
    :type length: int
    :param length: segment length in samples, even.
    :type grid: int
    :param grid: FFT length ``|A|`` is sampled on to obtain the
        autocorrelation; long enough that its period does not fold the
        lags kept.
    :return: tuple -- the frequencies as fractions of the sampling rate and
        the spectrum, in units of ``ar[0]**2`` two-sided per unit of those, on
        the ``4 length // 2 + 1`` non-negative frequencies of a ``4 length``
        grid.
    """
    from scipy.signal.windows import hann

    length = int(length)
    modulus = np.abs(np.fft.rfft(prediction_error_polynomial(ar), int(grid)))
    autocorrelation = np.fft.irfft(1.0 / modulus ** 2, int(grid))[:length]
    window = hann(length, sym=False)
    # `np.correlate` makes one BLAS dot product per lag. Above ten thousand
    # samples OpenBLAS runs each of them on all its threads, and the cost of
    # starting them on every lag makes a 16 s filter at 2048 Hz take minutes
    # of wall time and hours of CPU; on one thread it takes a fraction of a
    # second, with the same result.
    with threadpool_limits(1, user_api="blas"):
        lag_window = np.correlate(window, window, "full")[length - 1:] / np.dot(window, window)
    kept = autocorrelation * lag_window
    size = 4 * length
    lags = np.zeros(size)
    lags[:length] = kept
    lags[size - length + 1:] = kept[1:][::-1]
    spectrum = np.fft.rfft(lags).real
    # Positive by construction (a positive spectrum smoothed by a positive
    # window); rounding, where the stop band is 1e-12 of the band, is not.
    spectrum = np.maximum(spectrum, np.finfo(float).eps * spectrum.max())
    return np.fft.rfftfreq(size), spectrum


def truncated_taps(ar, duration, sampling, band=None, blend=DEFAULT_BAND_BLEND_HZ):
    """Inverse-spectrum truncation of an AR model: gwpy's whitening filter.

    The response is ``1 / sqrt(S_T)``, ``S_T`` the model's spectrum at the
    resolution of the filter (`spectrum_at_resolution`), held outside `band`
    (`held_modulus`), taken to `duration` seconds and truncated there by a Hann
    window: ``TimeSeries.whiten(asd=..., fduration=duration)`` with the ASD of
    the Burg model in place of a Welch estimate. The response is in units of
    ``1 / ar[0]``, like ``|A|``, whose smoothed counterpart it is.

    :type ar: numpy.ndarray
    :param ar: AR coefficients as `ArBurgEstimator` holds them.
    :type duration: float
    :param duration: length of the filter, seconds; the support is half of it.
    :type sampling: float
    :param sampling: sampling frequency of the model, hertz.
    :type band: tuple or None
    :param band: ``(f_lo, f_hi)`` in hertz outside which the response is held;
        None for the full band.
    :type blend: float
    :param blend: width of the raised cosine inside each edge, hertz.
    :return: numpy.ndarray -- ``2 K + 1`` symmetric taps, lag zero at ``K``,
        ``K = duration * sampling / 2``.
    :raises ValueError: if the filter would hold fewer than four taps.
    """
    length = 2 * int(round(0.5 * float(duration) * float(sampling)))
    if length < 4:
        raise ValueError(f"a {duration} s filter at {sampling} Hz holds fewer than four taps")
    freq, spectrum = spectrum_at_resolution(ar, length)
    response = 1.0 / np.sqrt(spectrum)
    if band is not None:
        response = held_modulus(freq * float(sampling), response, band, blend)
    return symmetric_taps(response, support=length // 2, taper=1.0)


class _SymmetricTapsWhitening:
    """Whiten a stream through a fixed set of symmetric taps.

    The streaming shared by the zero-phase filters whose impulse response is
    held as taps, whatever response they were built from. The taps are even,
    ``2 K + 1`` of them with lag zero at ``K``, so the phase is zero at every
    frequency. They are applied by FFT convolution over each output block
    together with ``K`` samples of the real stream before it and ``K`` after
    it, ``K`` the support. That is linear convolution with a fixed filter, so
    the output does not depend on where the blocks begin: a stream whitened
    block by block is the stream whitened at once.

    The interface is `ZeroPhaseWhitening`'s, so the worker drives either the
    same way. The output of a block is ready once ``output_size +
    extra_size`` samples are buffered; with ``extra_size`` at least ``K``, all
    of its future is real data. The past is the ``K`` input samples preceding
    the block, zeros before the stream has supplied them, so the first ``K``
    samples a stream emits are not whitened data and are to be discarded as
    the warm-up is. The latency is ``K`` samples. The stream is not divided by
    `sigma`: the output is in the units the taps give it, and `sigma` is its
    level, for the search to read.
    """

    def _install(self, taps, sigma, output_size, extra_size):
        """Hold the filter and an empty stream.

        Every way of obtaining the response ends here, so that the filters
        differ in the response and in nothing else.

        :type taps: numpy.ndarray
        :param taps: ``2 K + 1`` symmetric taps, lag zero at ``K``.
        :type sigma: float
        :param sigma: level in band of the whitened output.
        :type output_size: int
        :param output_size: whitened samples produced per `Output` call.
        :type extra_size: int
        :param extra_size: samples buffered beyond the output block.
        :return: None
        :raises ValueError: if `extra_size` is positive and below the support.
        """
        self.taps = np.asarray(taps, dtype=float)
        self.support = (self.taps.size - 1) // 2
        self.sigma = float(sigma)
        self._check_lookahead(extra_size)
        self.output_size, self.extra_size = int(output_size), int(extra_size)
        self._buffer = np.zeros(0)
        self._history = np.zeros(self.support)
        self._start = None
        self._interval = None


    @property
    def latency(self):
        """Samples of future data an output sample depends on: the support."""
        return self.support

    def _check_lookahead(self, extra_size):
        """Refuse a lookahead that would put zeros in place of the future."""
        if 0 < extra_size < self.support:
            raise ValueError(
                f"the lookahead is {extra_size} samples but the filter reads "
                f"{self.support} ahead: the future would be zeros at every "
                f"block join. Set WhiteningExtraSize to at least the filter's "
                f"support.")

    def Input(self, data):
        """Append one chunk to the stream without producing output.

        The stream's time is taken from the first chunk it is given, and every
        later chunk is taken to follow the previous one.

        :type data: py4tsa.tsa.SeqView_double_t
        :param data: input chunk, band-passed and decimated.
        :return: None
        """
        values = np.array([data.GetY(0, i) for i in range(data.GetSize())])
        if self._start is None:
            self._start, self._interval = data.GetStart(), data.GetSampling()
        self._buffer = np.concatenate([self._buffer, values * data.GetScale()])

    def Output(self, dataw):
        """Whiten the next `output_size` samples of the stream.

        :type dataw: py4tsa.tsa.SeqView_double_t
        :param dataw: output view, replaced by the whitened block, labelled
            with the time of its first sample.
        :return: None
        :raises RuntimeError: if fewer than ``output_size + extra_size``
            samples are buffered.
        """
        from scipy.signal import fftconvolve

        from wdf.structures.array2SeqView import array2SeqView

        n, k = self.output_size, self.support
        if self._buffer.size < n + self.extra_size:
            raise RuntimeError(
                f"{type(self).__name__}: {self._buffer.size} samples buffered, "
                f"{n + self.extra_size} needed")
        future = self._buffer[n:n + k]
        future = np.concatenate([future, np.zeros(k - future.size)])
        joined = np.concatenate([self._history, self._buffer[:n], future])
        whitened = fftconvolve(joined, self.taps, mode="valid")
        self._history = joined[n:n + k]
        self._buffer = self._buffer[n:]

        view = array2SeqView(self._start, 1.0 / self._interval, n)
        view.Fill(self._start, whitened)
        view.SV.SetScale(1.0)
        dataw.assign(view.SV)
        self._start += self._interval * n

    def Process(self, data, dataw):
        """Append one chunk and whiten the next block.

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
        :raises ValueError: if `extra_size` is positive and below the support.
        """
        self._check_lookahead(extra_size)
        self.output_size, self.extra_size = int(output_size), int(extra_size)


class MagnitudeWhitening(_SymmetricTapsWhitening):
    """Whiten a stream by the modulus of its prediction-error filter.

    The response is ``|A(e^{iw})|``, real and non-negative, so the phase is
    zero at every frequency and the whitened spectrum is the causal
    whitening's, bin by bin: the two differ in phase and in nothing else. The
    impulse response is the symmetric `magnitude_taps`, streamed as
    `_SymmetricTapsWhitening` describes, with latency ``K``, the support.
    """

    def __init__(self, ar, output_size, extra_size=0, floor=DEFAULT_RESPONSE_FLOOR,
                 grid=DEFAULT_RESPONSE_GRID, support=None, band=None, sampling=None,
                 blend=DEFAULT_BAND_BLEND_HZ, duration=None):
        """
        :type ar: numpy.ndarray
        :param ar: AR coefficients as `ArBurgEstimator` holds them -- the noise
            scale in ``ar[0]`` and the prediction coefficients in ``ar[1:]``.
        :type output_size: int
        :param output_size: whitened samples produced per `Output` call.
        :type extra_size: int
        :param extra_size: samples buffered beyond the output block before it
            is produced. Zero, or at least the support; a positive value below
            the support is refused, since it would replace real future data by
            zeros at every block join.
        :type floor: float
        :param floor: fraction of the peak below which the impulse response
            is spent (`response_support`).
        :type grid: int
        :param grid: FFT length ``|A|`` is sampled on.
        :type support: int or None
        :param support: half-length of the filter in samples; None to measure it.
        :type band: tuple or None
        :param band: ``(f_lo, f_hi)`` in hertz, the passband of the
            conditioning: the response is the held modulus ``G``
            (`held_modulus`) and `sigma` its level in band (`in_band_scale`),
            ``ar[0]``. None for ``|A|`` over the full band, with `sigma`
            ``ar[0]``.
        :type sampling: float or None
        :param sampling: sampling frequency of the model, hertz; needed with `band`.
        :type blend: float
        :param blend: width of the raised cosine inside each edge, hertz.
        :type duration: float or None
        :param duration: seconds; the filter is then `truncated_taps`, the
            model's inverse spectrum at the resolution of `duration`,
            Hann-truncated to it, with support and latency half of it, and
            `sigma` the level in band of the response it applies
            (`in_band_scale`; the full band when `band` is None). Needs
            `sampling`; `floor` and `support` are not used.
        :raises ValueError: if `extra_size` is positive and below the support,
            or the response does not decay within the grid.
        """
        polynomial = prediction_error_polynomial(ar)
        if grid < 2 * polynomial.size:
            raise ValueError(f"grid {grid} is too short for an order "
                             f"{polynomial.size - 1} model")
        scale = float(np.asarray(ar, dtype=float).reshape(-1)[0])
        if duration is not None:
            if sampling is None:
                raise ValueError("a duration in seconds needs the sampling frequency")
            taps = truncated_taps(ar, duration, sampling, band, blend)
            modulus, _ = held_response(ar, grid)
            applied = np.abs(np.fft.rfft(taps, int(grid)))
            freq = np.fft.rfftfreq(int(grid), 1.0 / float(sampling))
            sigma = in_band_scale(scale, applied, modulus, freq, band)
            self.duration = float(duration)
        else:
            modulus, target = held_response(ar, grid, band, sampling, blend)
            taps = symmetric_taps(target, floor, support)
            sigma = scale if band is None else in_band_scale(
                scale, target, modulus,
                np.fft.rfftfreq(int(grid), 1.0 / float(sampling)), band)
            self.duration = None
        self.band = None if band is None else (float(band[0]), float(band[1]))
        self._install(taps, sigma, output_size, extra_size)


    @classmethod
    def from_spectrum(cls, samples, sampling, output_size, extra_size=0, band=None,
                      nperseg=8192, average="median", blend=DEFAULT_BAND_BLEND_HZ,
                      taper=RESPONSE_TAPER):
        """Build the whitening from the spectrum of a stretch, without Burg.

        The spectrum is measured on the stretch, held flat outside `band`, and
        the filter is `magnitude_taps_from_spectrum`, whose support is half of
        `nperseg`; the noise scale is then read on that same stretch, whitened,
        as its level in band (`whitened_level`). With `taper` 1 and `nperseg`
        four seconds of data this is gwpy's ``TimeSeries.whiten``.

        :type samples: numpy.ndarray
        :param samples: the conditioned stretch to measure.
        :type sampling: float
        :param sampling: its sampling frequency, hertz.
        :type output_size: int
        :param output_size: whitened samples produced per `Output` call.
        :type extra_size: int
        :param extra_size: samples buffered beyond the output block.
        :type band: tuple or None
        :param band: ``(low, high)`` outside which the spectrum is held flat.
        :type nperseg: int
        :param nperseg: segment length of the spectral estimate, even; it is
            also the length of the filter.
        :type average: str
        :param average: how the periodograms are combined, "median" or "mean".
        :type taper: float
        :param taper: Tukey fraction of the truncation (`symmetric_taps`).
        :return: MagnitudeWhitening
        :raises ValueError: if the stretch is shorter than one segment, or
            than the filter it would be read through.
        """
        from scipy.signal import fftconvolve, welch

        samples = np.asarray(samples, dtype=float).reshape(-1)
        if samples.size < nperseg:
            raise ValueError(
                f"the stretch holds {samples.size} samples, fewer than the "
                f"{nperseg} one segment of the spectral estimate needs")
        freq, psd = welch(samples, fs=float(sampling), nperseg=int(nperseg),
                          average=average)
        taps = magnitude_taps_from_spectrum(freq, psd, band=band, blend=blend,
                                            taper=taper)
        if samples.size <= taps.size:
            raise ValueError(
                f"the stretch holds {samples.size} samples, no more than the "
                f"{taps.size} taps of the filter it is whitened through")
        whitened = fftconvolve(samples, taps, mode="valid")
        self = cls.__new__(cls)
        self.band = None if band is None else (float(band[0]), float(band[1]))
        self.duration = None
        self._install(taps, whitened_level(whitened, sampling, band, nperseg),
                      output_size, extra_size)
        return self
