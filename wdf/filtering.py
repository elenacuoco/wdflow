"""Zero-phase filtering whose output does not depend on the scipy version.

Kept apart from `wdf.processes` so that `wdf.mock` can use it without the
compiled core.
"""
import numpy as np
from scipy.signal import sosfilt


def sosfilt_zi(sos):
    """Initial conditions for `sos` at the steady state of a unit step.

    scipy's own `sosfilt_zi` gave these until 1.17; 1.18 rewrote the solver
    underneath it, and the zero-phase filtering below moved in its last bits --
    a few parts in 1e10, but enough to shift every trigger a golden reference
    pins, and to do so according to which scipy happens to be installed. This
    is the computation scipy used up to 1.17, done here so the conditioned data
    no longer depend on the scipy version. `sosfilt` itself did not change.

    :type sos: numpy.ndarray
    :param sos: second-order sections, shape (n_sections, 6).
    :return: numpy.ndarray -- shape (n_sections, 2).
    """
    sos = np.asarray(sos, dtype=float)
    zi = np.empty((sos.shape[0], 2))
    scale = 1.0
    for k in range(sos.shape[0]):
        b = sos[k, :3]
        a = sos[k, 3:]
        bn, an = (b / a[0], a / a[0]) if a[0] != 1.0 else (b, a)
        companion = np.zeros((2, 2))
        companion[0, :] = -an[1:] / (1.0 * an[0:1])
        companion[1, 0] = 1
        zi[k, ...] = scale * np.linalg.solve(np.eye(2) - companion.T,
                                             bn[1:] - an[1:] * bn[0])
        # b.sum()/a.sum() is the section's gain at DC, the level its step
        # response settles to, which the next section starts from.
        scale *= np.sum(b) / np.sum(a)
    return zi


def sosfiltfilt(sos, x, padlen=None):
    """Zero-phase filtering of a 1-D array, as scipy's `sosfiltfilt` does it.

    Same padding (odd extension, by default three times the filter's tap count
    at each end), same initial conditions, same forward-backward passes; only
    the initial conditions come from `sosfilt_zi` above instead of scipy's.

    :type sos: numpy.ndarray
    :param sos: second-order sections, shape (n_sections, 6).
    :type x: numpy.ndarray
    :param x: 1-D data.
    :type padlen: int or None
    :param padlen: samples of odd extension at each end; None for the default.
    :return: numpy.ndarray -- the filtered data, same length as `x`.
    """
    sos = np.asarray(sos)
    x = np.asarray(x)
    if padlen is None:
        ntaps = 2 * sos.shape[0] + 1
        ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
        edge = 3 * ntaps
    else:
        edge = int(padlen)
    if x.shape[0] <= edge:
        raise ValueError(f"The length of the input vector x must be greater "
                         f"than padlen, which is {edge}.")
    if edge > 0:
        ext = np.concatenate((2 * x[0:1] - x[edge:0:-1],
                              x,
                              2 * x[-1:] - x[-2:-(edge + 2):-1]))
    else:
        ext = x
    zi = sosfilt_zi(sos)
    y, _ = sosfilt(sos, ext, zi=zi * ext[0:1])
    y, _ = sosfilt(sos, y[::-1], zi=zi * y[-1:])
    y = y[::-1]
    return y[edge:-edge] if edge > 0 else y
