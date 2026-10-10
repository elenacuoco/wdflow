"""The dechirped local cosine: a chirp's sweep removed before the local cosine.

A window is turned into its analytic signal, multiplied by
``exp(-i pi alpha (t - t_mid)^2)`` (``t_mid`` the window's mean sample time),
and its real part is taken through the local cosine (`local_cosine`). A
component sweeping at ``alpha`` Hz/s through the window comes out at a fixed
frequency, so its energy falls into a few tiles of one row rather than across
the rows it sweeps through. In the original frame each tile is the
parallelogram ``f in [f_lo, f_hi) + alpha (t - t_mid)`` over its segment
(`dechirped_geometry`); the re-chirp inverts the transform
(`inverse_dechirped_local_cosine`).

**It is not an orthonormal basis.** The real part folds the frequencies the
dechirp moves below zero or above the Nyquist frequency onto their mirror, so
a few singular values of the map (those of the bands within ``|alpha| T / 2``
of 0 Hz and of the Nyquist frequency, T the window duration) leave 1: at a
window of 1024 samples, 2048 Hz and alpha 20 Hz/s they run from about 0.05 to
1.4. The analytic signal is computed on the window padded with zeros
(`analytic_signal`), so nothing wraps from one end of the window to the other;
the fold remains. It cannot therefore enter WDF2Classify's competition, whose
candidates are orthonormal; it is a library transform, offered to studies and
to the ridge's maps, and in no default configuration.

The sweep tolerance is set by the tile, not by the window: a sweep mismatch
``d alpha`` moves a component by ``d alpha T_seg`` across a segment of
duration ``T_seg``, so the grid step is about ``1 / (2 T_seg^2)``
(`alpha_grid`).
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

import numpy as np

from wdf.analysis.local_cosine import inverse_local_cosine, local_cosine, local_cosine_geometry


def analytic_signal(x, pad=None):
    """The analytic signal of each window, padded with zeros, not circular.

    :type x: numpy.ndarray
    :param x: samples, ``(N,)`` or ``(windows, N)``.
    :type pad: int or None
    :param pad: zeros added at each end before the Hilbert transform, samples;
        N when None.
    :return: numpy.ndarray -- complex, same shape.
    """
    from scipy.signal import hilbert

    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    pad = n if pad is None else int(pad)
    widths = [(0, 0)] * (x.ndim - 1) + [(pad, pad)]
    return hilbert(np.pad(x, widths), axis=-1)[..., pad:pad + n]


def chirp_phase(n, fs, alpha):
    """``exp(-i pi alpha (t - t_mid)^2)`` on a window of `n` samples.

    :type n: int
    :param n: window length, samples.
    :type fs: float
    :param fs: sampling rate, Hz.
    :type alpha: float
    :param alpha: sweep, Hz/s.
    :return: numpy.ndarray -- complex, length n.
    """
    t = np.arange(int(n)) / float(fs)
    return np.exp(-1j * np.pi * float(alpha) * (t - t.mean()) ** 2)


def dechirp(x, fs, alpha, pad=None):
    """``Re(analytic(x) exp(-i pi alpha (t - t_mid)^2))``.

    :type x: numpy.ndarray
    :param x: samples, ``(N,)`` or ``(windows, N)``.
    :type fs: float
    :param fs: sampling rate, Hz.
    :type alpha: float
    :param alpha: sweep removed, Hz/s.
    :type pad: int or None
    :param pad: zeros of the analytic signal (`analytic_signal`).
    :return: numpy.ndarray -- real, same shape.
    """
    x = np.asarray(x, dtype=float)
    return np.real(analytic_signal(x, pad) * chirp_phase(x.shape[-1], fs, alpha))


def rechirp(y, fs, alpha, pad=None):
    """The sweep put back: ``Re(analytic(y) exp(+i pi alpha (t - t_mid)^2))``.

    :return: numpy.ndarray -- real, same shape as `y`.
    """
    y = np.asarray(y, dtype=float)
    return np.real(analytic_signal(y, pad) * np.conj(chirp_phase(y.shape[-1], fs, alpha)))


def dechirped_local_cosine(x, fs, alpha, m, pad=None, overlap=None):
    """The dechirped local cosine coefficients, laid out as `LocalCos<M>`.

    :type x: numpy.ndarray
    :param x: samples, ``(N,)`` or ``(windows, N)``.
    :type fs: float
    :param fs: sampling rate, Hz.
    :type alpha: float
    :param alpha: sweep, Hz/s; 0 is the local cosine itself.
    :type m: int
    :param m: segment length, samples.
    :type pad: int or None
    :param pad: zeros of the analytic signal.
    :type overlap: int or None
    :param overlap: bell half-width, samples.
    :return: numpy.ndarray -- the coefficients, same shape.
    """
    if float(alpha) == 0.0:
        return local_cosine(x, m, overlap)
    return local_cosine(dechirp(x, fs, alpha, pad), m, overlap)


def inverse_dechirped_local_cosine(c, fs, alpha, m, pad=None, overlap=None):
    """The original-frame waveform of dechirped local cosine coefficients: the
    inverse local cosine, re-chirped.

    Exact for alpha 0; otherwise the inverse of the map on the bands the fold
    does not reach (see the module's note).

    :return: numpy.ndarray -- samples, same shape as `c`.
    """
    y = inverse_local_cosine(c, m, overlap)
    if float(alpha) == 0.0:
        return y
    return rechirp(y, fs, alpha, pad)


def dechirped_geometry(n, m, fs, alpha):
    """Each coefficient's tile in the original frame.

    The segment ``[t_lo, t_hi)`` (s from the window's start) and the dechirped
    bin ``[f_lo, f_hi)`` give the parallelogram
    ``f in [f_lo, f_hi) + alpha (t - t_mid)``; its corners and its bounding
    rectangle ``[bf_lo, bf_hi)`` are returned with it, the rectangle being
    the fallback where a tile must be a rectangle.

    :type n: int
    :param n: window length, samples.
    :type m: int
    :param m: segment length, samples.
    :type fs: float
    :param fs: sampling rate, Hz.
    :type alpha: float
    :param alpha: sweep, Hz/s.
    :return: dict -- arrays of length N: ``t_lo``, ``t_hi``, ``f_lo``,
        ``f_hi`` (dechirped bin), ``shift_lo``, ``shift_hi`` (the sweep's
        offset at the segment's two ends, Hz), ``bf_lo``, ``bf_hi`` (the
        bounding rectangle, clipped to [0, fs / 2]), ``corners`` (N, 4, 2):
        (t, f) of the parallelogram, counter-clockwise from (t_lo, f_lo +
        shift_lo), and ``row``, ``pos``.
    """
    g = local_cosine_geometry(n, m, fs)
    t_mid = (n - 1) / (2.0 * fs)
    s0, s1 = alpha * (g["t_lo"] - t_mid), alpha * (g["t_hi"] - t_mid)
    corners = np.stack([np.c_[g["t_lo"], g["f_lo"] + s0], np.c_[g["t_hi"], g["f_lo"] + s1],
                        np.c_[g["t_hi"], g["f_hi"] + s1], np.c_[g["t_lo"], g["f_hi"] + s0]], axis=1)
    return dict(g, shift_lo=s0, shift_hi=s1,
                bf_lo=np.clip(g["f_lo"] + np.minimum(s0, s1), 0.0, 0.5 * fs),
                bf_hi=np.clip(g["f_hi"] + np.maximum(s0, s1), 0.0, 0.5 * fs),
                corners=corners)


def alpha_grid(segment_s, alpha_max, step=None):
    """A symmetric grid of sweeps, 0 included.

    :type segment_s: float
    :param segment_s: the tile's duration T (the local cosine segment, M / fs),
        s; the tolerance is set by it, not by the window.
    :type alpha_max: float
    :param alpha_max: largest ``|alpha|`` on the grid, Hz/s (the physical range:
        a BBH within 50 ms of its merger sweeps at a few hundred Hz/s).
    :type step: float or None
    :param step: grid step, Hz/s; ``1 / (2 T^2)`` when None.
    :return: numpy.ndarray -- ``-k step .. k step``, ``k = floor(alpha_max /
        step)``.
    :raises ValueError: on a non-positive segment, step or a negative range.
    """
    segment_s = float(segment_s)
    if segment_s <= 0.0 or float(alpha_max) < 0.0:
        raise ValueError("the segment must be positive and the range non-negative")
    step = 1.0 / (2.0 * segment_s ** 2) if step is None else float(step)
    if step <= 0.0:
        raise ValueError(f"the alpha step must be positive, got {step}")
    k = int(np.floor(float(alpha_max) / step + 1e-9))
    return np.arange(-k, k + 1) * step
