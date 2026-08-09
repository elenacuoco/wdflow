"""The event's parameters, read off the wavelet coefficients that produced it.

Each surviving coefficient occupies a known tile in time and frequency, fixed by
its index for a given window length and sampling rate. The event's extent, its
band and the time it is centred on are therefore moments of the energy those
tiles carry, and need neither an inverse transform nor a periodogram.

The time an event is placed at is its energy centroid rather than its loudest
tile. Which tile is loudest depends on the noise realisation and on which basis
won in that detector, so the peak need not fall at the same instant in two
detectors seeing the same signal; the first moment is defined whether or not one
tile dominates, and moves continuously as the coefficients change.
"""
from __future__ import annotations

import numpy as np

from wdf.analysis.wavelets import (
    coeff_freq_bands,
    coeff_time_bounds,
    tile_frequency,
)

META_FEATURES = [
    "gpsStart", "gpsCentroid", "tSpread", "gpsPeak", "duration",
    "snrPeak", "freqMin", "freqMean", "freqMax",
]


def _empty() -> dict:
    return {name: float("nan") for name in META_FEATURES}


def meta_features(index, value, n_coeff: int, fs: float, sigma: float,
                  gps: float = 0.0) -> dict:
    """Derive an event's parameters from its surviving wavelet coefficients.

    :type index: array-like
    :param index: coefficient indices of the survivors.
    :type value: array-like
    :param value: coefficient values, in the same order as `index`.
    :type n_coeff: int
    :param n_coeff: length of the analysis window's coefficient vector.
    :type fs: float
    :param fs: sampling frequency the coefficients were computed at, Hz.
    :type sigma: float
    :param sigma: noise scale the amplitudes are expressed in.
    :type gps: float
    :param gps: GPS time of the analysis window's first sample, which the
        returned times are absolute against.
    :return: dict -- `gpsStart`, `gpsCentroid`, `tSpread`, `gpsPeak`,
        `duration`, `snrPeak`, `freqMin`, `freqMean`, `freqMax`;
        all `nan` when no coefficient survived.
    """
    index = np.asarray(index, dtype=np.int64).reshape(-1)
    value = np.asarray(value, dtype=float).reshape(-1)
    if index.size == 0:
        return _empty()

    magnitude = np.abs(value)
    energy = magnitude * magnitude
    total = float(energy.sum())
    if total <= 0.0:
        return _empty()

    t_lo, t_hi = coeff_time_bounds(int(n_coeff), float(fs))
    f_lo, f_hi = coeff_freq_bands(int(n_coeff), float(fs))
    t_lo, t_hi = t_lo[index], t_hi[index]
    f_lo, f_hi = f_lo[index], f_hi[index]
    t_mid = 0.5 * (t_lo + t_hi)

    centroid = float(energy @ t_mid) / total
    spread = np.sqrt(max(float(energy @ (t_mid - centroid) ** 2) / total, 0.0))

    loudest = int(np.argmax(magnitude))
    frequency = np.array([tile_frequency(a, b) for a, b in zip(f_lo, f_hi)])
    frequency = np.maximum(frequency, np.finfo(float).tiny)

    start = float(t_lo.min())
    return dict(
        gpsStart=gps + start,
        gpsCentroid=gps + centroid,
        tSpread=float(spread),
        gpsPeak=gps + float(t_mid[loudest]),
        duration=float(t_hi.max()) - start,
        snrPeak=float(magnitude[loudest] / sigma) if sigma > 0 else float("nan"),
        freqMin=float(f_lo.min()),
        freqMean=float(np.exp(float(energy @ np.log(frequency)) / total)),
        freqMax=float(f_hi.max()),
    )
