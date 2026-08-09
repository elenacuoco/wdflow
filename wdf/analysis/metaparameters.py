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
    "gpsStart", "gpsCentroid", "tSpread", "gpsPeak", "duration", "duration90",
    "snrPeak", "freqMin", "freqMean", "freqMax", "freqQ05", "freqQ95",
]


def _empty() -> dict:
    return {name: float("nan") for name in META_FEATURES}


def _energy_quantile(lo, hi, energy, quantiles):
    """Quantiles of energy spread uniformly over a set of intervals.

    Each tile holds its energy over its own extent rather than at a point, so
    the mixture is piecewise uniform and its quantiles are read by inverting the
    cumulative distribution. Hard support bounds are the extremes of the same
    distribution, and one marginal coefficient moves them arbitrarily far; these
    do not move until the energy does.

    :param lo: lower edge of each interval.
    :param hi: upper edge of each interval.
    :param energy: energy carried by each interval.
    :param quantiles: the quantiles wanted, between 0 and 1.
    :return: numpy.ndarray -- one value per requested quantile.
    """
    order = np.argsort(lo, kind="mergesort")
    lo, hi, energy = lo[order], hi[order], energy[order]
    total = energy.sum()
    if total <= 0.0:
        return np.full(len(quantiles), np.nan)

    edges = np.concatenate(([0.0], np.cumsum(energy) / total))
    out = np.empty(len(quantiles))
    for slot, q in enumerate(quantiles):
        k = min(int(np.searchsorted(edges, q, side="right")) - 1, len(lo) - 1)
        k = max(k, 0)
        width = edges[k + 1] - edges[k]
        within = (q - edges[k]) / width if width > 0 else 0.0
        out[slot] = lo[k] + within * (hi[k] - lo[k])
    return out


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
    # A tile is an interval, not a point, so its own width contributes to the
    # spread: uniformly distributed energy over a width w has variance w^2/12.
    # It matters most where the tiles differ in width, which is exactly what
    # searching at several window lengths produces.
    variance = float(energy @ ((t_mid - centroid) ** 2 + (t_hi - t_lo) ** 2 / 12.0))
    spread = np.sqrt(max(variance / total, 0.0))

    loudest = int(np.argmax(magnitude))
    frequency = np.array([tile_frequency(a, b) for a, b in zip(f_lo, f_hi)])
    frequency = np.maximum(frequency, np.finfo(float).tiny)

    start = float(t_lo.min())
    # The support is what a marginal coefficient can stretch without carrying
    # energy; these follow the energy instead. The frequency quantiles are taken
    # in log frequency, the coordinate the dyadic tiling is uniform in.
    t05, t95 = _energy_quantile(t_lo, t_hi, energy, (0.05, 0.95))
    logf05, logf95 = _energy_quantile(
        np.log(np.maximum(f_lo, np.finfo(float).tiny)), np.log(f_hi), energy,
        (0.05, 0.95))
    return dict(
        gpsStart=gps + start,
        gpsCentroid=gps + centroid,
        tSpread=float(spread),
        gpsPeak=gps + float(t_mid[loudest]),
        duration=float(t_hi.max()) - start,
        snrPeak=float(magnitude[loudest] / sigma) if sigma > 0 else float("nan"),
        duration90=float(t95 - t05),
        freqMin=float(f_lo.min()),
        freqMean=float(np.exp(float(energy @ np.log(frequency)) / total)),
        freqMax=float(f_hi.max()),
        freqQ05=float(np.exp(logf05)),
        freqQ95=float(np.exp(logf95)),
    )
