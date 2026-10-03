"""The local cosine basis in NumPy, laid out as p4TSA's `LocalCos<M>`.

The orthonormal local cosine transform of Coifman and Meyer: the window is cut
into segments of M samples, each edge folded with a smooth bell of half-width
`overlap`, and each folded segment taken through an orthonormal DCT-IV. The
window is treated as a circle, as p4TSA treats it. The coefficients are laid
out bin-major, index ``bin * (N / M) + segment``, which is the layout of the
packet level of depth log2 M (`wdf.analysis.wavelets.packet_depth`): row
``bin`` covers ``[bin, bin + 1) fs / (2 M)`` Hz, position ``segment`` covers
samples ``[segment M, (segment + 1) M)``.

With the default bell (Coifman-Meyer, three iterated sines, half-width M/2)
the forward transform equals p4TSA's ``WaveletTransform(N, LocalCos, log2 M)``
to rounding, so a trigger that won `LocalCos<M>` can be read and inverted here
without the compiled core, many windows at a time.
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

import numpy as np

#: Iterations of the sine that make the Coifman-Meyer bell smooth.
DEFAULT_BELL_ITERATIONS = 3


def bell(m, overlap=None, iterations=DEFAULT_BELL_ITERATIONS):
    """The rising cut-off r(t) of the folding, ``r(t)^2 + r(-t)^2 = 1``.

    :type m: int
    :param m: segment length, samples.
    :type overlap: int or None
    :param overlap: half-width of the bell, samples; M / 2 when None.
    :type iterations: int
    :param iterations: iterated sines of the profile (0 is the sine bell).
    :return: numpy.ndarray -- r on t = -overlap .. overlap - 1 (+1/2).
    """
    overlap = m // 2 if overlap is None else int(overlap)
    theta = (np.arange(-overlap, overlap) + 0.5) / overlap
    for _ in range(int(iterations)):
        theta = np.sin(0.5 * np.pi * theta)
    return np.sin(0.25 * np.pi * (1.0 + theta))


def _check(n, m, overlap):
    if m < 2 or m & (m - 1) or n % m:
        raise ValueError(f"a local cosine of segment {m} on {n} samples needs M a power of 2 "
                         "dividing the window")
    if not 0 < overlap <= m // 2:
        raise ValueError(f"the bell half-width {overlap} must lie in (0, M / 2]")


def _edges(n, m, overlap):
    """Sample indices to the right and to the left of every segment edge."""
    a = np.arange(0, n, m)[:, None]
    t = np.arange(overlap)[None, :]
    return (a + t) % n, (a - 1 - t) % n


def local_cosine(x, m, overlap=None, iterations=DEFAULT_BELL_ITERATIONS):
    """The local cosine coefficients of one window or of many.

    :type x: numpy.ndarray
    :param x: samples, ``(N,)`` or ``(windows, N)``.
    :type m: int
    :param m: segment length M, samples, a power of 2 dividing N.
    :type overlap: int or None
    :param overlap: bell half-width, samples; M / 2 when None.
    :type iterations: int
    :param iterations: iterated sines of the bell.
    :return: numpy.ndarray -- the coefficients, same shape, bin-major.
    """
    from scipy.fft import dct

    x = np.asarray(x, dtype=float)
    X = np.atleast_2d(x)
    w, n = X.shape
    overlap = m // 2 if overlap is None else int(overlap)
    _check(n, m, overlap)
    r = bell(m, overlap, iterations)
    rp, rm = r[overlap:], r[:overlap][::-1]
    right, left = _edges(n, m, overlap)
    Y = X.copy()
    xr, xl = X[:, right], X[:, left]
    Y[:, right] = rp * xr + rm * xl
    Y[:, left] = rp * xl - rm * xr
    C = dct(Y.reshape(w, n // m, m), type=4, norm="ortho", axis=-1)
    C = C.transpose(0, 2, 1).reshape(w, n)
    return C if x.ndim == 2 else C[0]


def inverse_local_cosine(c, m, overlap=None, iterations=DEFAULT_BELL_ITERATIONS):
    """The samples of local cosine coefficients: the inverse of `local_cosine`.

    :type c: numpy.ndarray
    :param c: coefficients, ``(N,)`` or ``(windows, N)``, bin-major.
    :type m: int
    :param m: segment length M, samples.
    :type overlap: int or None
    :param overlap: bell half-width, samples; M / 2 when None.
    :type iterations: int
    :param iterations: iterated sines of the bell.
    :return: numpy.ndarray -- the samples, same shape.
    """
    from scipy.fft import idct

    c = np.asarray(c, dtype=float)
    C = np.atleast_2d(c)
    w, n = C.shape
    overlap = m // 2 if overlap is None else int(overlap)
    _check(n, m, overlap)
    segments = C.reshape(w, m, n // m).transpose(0, 2, 1)
    Y = idct(segments, type=4, norm="ortho", axis=-1).reshape(w, n)
    r = bell(m, overlap, iterations)
    rp, rm = r[overlap:], r[:overlap][::-1]
    right, left = _edges(n, m, overlap)
    X = Y.copy()
    yr, yl = Y[:, right], Y[:, left]
    X[:, right] = rp * yr - rm * yl
    X[:, left] = rp * yl + rm * yr
    return X if c.ndim == 2 else X[0]


def local_cosine_geometry(n, m, fs):
    """Each coefficient's tile: segment in time, bin in frequency.

    :type n: int
    :param n: window length, samples.
    :type m: int
    :param m: segment length, samples.
    :type fs: float
    :param fs: sampling rate, Hz.
    :return: dict -- arrays of length N: ``t_lo``, ``t_hi`` (s from the
        window's start), ``f_lo``, ``f_hi`` (Hz), ``row`` (bin), ``pos``
        (segment).
    """
    _check(n, m, m // 2)
    segments = n // m
    b = np.repeat(np.arange(m), segments)
    k = np.tile(np.arange(segments), m)
    return dict(t_lo=k * m / fs, t_hi=(k + 1) * m / fs, f_lo=b * fs / (2.0 * m),
                f_hi=(b + 1) * fs / (2.0 * m), row=b, pos=k)
