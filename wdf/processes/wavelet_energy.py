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
    """Energy-based SNR from wavelet coefficients `wt`: keeps coefficients
    above `donoho_johnstone_threshold(sigma, len(wt))` and sums their energy.

    - `energy`: sum of surviving coefficients' `|wt|^2`.
    - `snr`: `sqrt(energy) / sigma`.
    - `n_above_threshold`: how many of `len(wt)` coefficients survived.
    - `threshold`: the threshold value used.
    """
    wt = np.asarray(wt, dtype=float)
    threshold = donoho_johnstone_threshold(sigma, len(wt))
    mask = np.abs(wt) >= threshold
    energy = float(np.sum(wt[mask] ** 2))
    return dict(
        energy=energy,
        snr=float(np.sqrt(energy) / sigma) if sigma > 0 else float("nan"),
        n_above_threshold=int(mask.sum()),
        threshold=float(threshold),
    )
