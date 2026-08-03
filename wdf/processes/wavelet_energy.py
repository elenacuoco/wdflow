"""Energy-based statistic derived directly from WDF's own wavelet coefficients,
used as the trigger SNR (see `ParameterEstimationObserver`).
"""
from __future__ import annotations

import numpy as np


def donoho_johnstone_threshold(sigma: float, n_coeff: int) -> float:
    """The universal wavelet-denoising threshold (Donoho & Johnstone, 1994):
    `sigma * sqrt(2 * ln(n_coeff))`. Depends only on the noise scale and the
    number of coefficients, not on any assumption about signal shape.
    """
    return sigma * np.sqrt(2.0 * np.log(n_coeff))


def wavelet_energy_snr(wt, sigma: float) -> dict:
    """Integrated wavelet-domain energy and the SNR it implies on a noise scale.

    Sums the energy of every coefficient. `wt` is expected in the form
    `WaveletThreshold` emits: coefficients that did not pass thresholding are
    exactly 0, and under soft thresholding the survivors are shrunk by the
    threshold, so the zeros already carry the significance decision.

    :type wt: numpy.ndarray
    :param wt: thresholded wavelet coefficients of one analysis window.
    :type sigma: float
    :param sigma: noise scale the SNR is expressed in.
    :return: dict -- `energy` (sum of `|wt|^2`), `snr` (`sqrt(energy)/sigma`,
        `nan` when `sigma <= 0`), `n_nonzero` (count of non-zero coefficients).
    """
    wt = np.asarray(wt, dtype=float)
    energy = float(np.sum(wt ** 2))
    return dict(
        energy=energy,
        snr=float(np.sqrt(energy) / sigma) if sigma > 0 else float("nan"),
        n_nonzero=int(np.count_nonzero(wt)),
    )
