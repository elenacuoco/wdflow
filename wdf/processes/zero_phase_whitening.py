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

from pytsa.tsa import DoubleWhitening, LatticeView

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
    :return: pytsa.tsa.LatticeView -- reflection coefficients of the
        square-root filter.
    """
    _, error, reflection = sqrt_ar_polynomial(ar, order=order, grid=grid)

    view = LatticeView(order)
    view.SetOrder(order)

    running_error = float(np.asarray(ar, dtype=float)[0])
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
        :param extra_size: lookahead buffer, in samples. Anything at or above
            ``order`` is exact; below it the backward pass is truncated.
        :type order: int
        :param order: order of the square-root model.
        :type grid: int
        :param grid: FFT length the response is evaluated on.
        """
        self.order = int(order)
        self.polynomial, self.error, self.reflection = sqrt_ar_polynomial(
            ar, order=order, grid=grid)
        self.sigma = float(np.asarray(ar, dtype=float)[0]) * self.error
        self.LV = sqrt_lattice_view(ar, order=order, grid=grid)

        self.filter = DoubleWhitening(self.LV, output_size, extra_size)
        self.filter.init(self.LV)

    @property
    def latency(self):
        """Samples of lookahead the backward pass needs; the whole latency."""
        return self.order

    def Process(self, data, dataw):
        """Whiten one chunk, blocking until a full output block is available.

        :type data: pytsa.tsa.SeqView_double_t
        :param data: input chunk, band-passed and decimated.
        :type dataw: pytsa.tsa.SeqView_double_t
        :param dataw: output sequence view, filled in place.
        :return: None
        """
        self.filter(data, dataw)

    def Input(self, data):
        """Feed one chunk in without reading output.

        :type data: pytsa.tsa.SeqView_double_t
        :param data: input chunk.
        :return: None
        """
        self.filter.Input(data)

    def Output(self, dataw):
        """Read whatever output is available.

        :type dataw: pytsa.tsa.SeqView_double_t
        :param dataw: output sequence view, filled in place.
        :return: None
        """
        self.filter.Output(dataw)

    def SetOutputSize(self, output_size, extra_size):
        """Change the output block size and the lookahead.

        :type output_size: int
        :param output_size: whitened samples produced per `Process` call.
        :type extra_size: int
        :param extra_size: lookahead buffer, in samples.
        :return: None
        """
        self.filter.SetOutputSize(output_size, extra_size)
