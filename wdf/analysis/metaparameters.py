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

EPS = np.finfo(float).tiny

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


def energy_quantile(lo, hi, energy, quantiles):
    """Quantiles of energy spread uniformly over a set of intervals.

    Each tile holds its energy over its own extent rather than at a point, so
    the distribution is the mixture of one uniform density per interval. The
    intervals overlap freely -- tiles of different octaves cover the same
    instants, and an event's members cover the same band -- so the mixture is
    accumulated over the breakpoints of their union and inverted there, rather
    than by walking the intervals in order.

    Hard support bounds are the extremes of this same distribution, and one
    marginal coefficient moves them arbitrarily far; these do not move until the
    energy does.

    :param lo: lower edge of each interval.
    :param hi: upper edge of each interval.
    :param energy: energy carried by each interval.
    :param quantiles: the quantiles wanted, between 0 and 1.
    :return: numpy.ndarray -- one value per requested quantile, ascending with
        the quantiles asked for.
    """
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    energy = np.asarray(energy, dtype=float).reshape(-1)
    quantiles = np.asarray(quantiles, dtype=float).reshape(-1)

    total = energy.sum()
    if lo.size == 0 or not np.isfinite(total) or total <= 0.0:
        return np.full(quantiles.shape, np.nan)

    breaks = np.unique(np.concatenate([lo, hi]))
    if breaks.size < 2:
        return np.full(quantiles.shape, float(breaks[0]))

    width = np.maximum(hi - lo, 0.0)
    left, right = breaks[:-1], breaks[1:]
    # Energy a segment receives from each interval covering it, in proportion to
    # how much of that interval the segment is. A zero-width interval puts all
    # of its energy on the segment starting at it.
    #
    # The segments an interval covers are contiguous, the breakpoints being
    # sorted, so each interval adds a constant density over a range of segments
    # and the sum over intervals is the cumulative sum of those range additions.
    # Asking instead which of every interval and segment pair overlap builds a
    # matrix of their product, which for an event of many coefficients is where
    # the time goes.
    spread = width > 0.0
    density = np.zeros(left.size + 1)
    if spread.any():
        first = np.searchsorted(left, lo[spread], side="left")
        last = np.searchsorted(right, hi[spread], side="right")
        height = energy[spread] / np.maximum(width[spread], EPS)
        np.add.at(density, np.minimum(first, left.size), height)
        np.add.at(density, np.minimum(last, left.size), -height)
    mass = np.cumsum(density)[:left.size] * (right - left)

    if not spread.all():
        # A zero-width interval sits on a breakpoint and gives its energy to the
        # segment starting there.
        at = np.searchsorted(left, lo[~spread], side="left")
        inside = at < left.size
        np.add.at(mass, at[inside], energy[~spread][inside])

    cumulative = np.concatenate(([0.0], np.cumsum(mass)))
    if cumulative[-1] <= 0.0:
        return np.full(quantiles.shape, float(breaks[0]))
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, breaks)


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
    # It matters where the tiles differ in width, which within one window is
    # what a transient spread across octaves produces.
    variance = float(energy @ ((t_mid - centroid) ** 2 + (t_hi - t_lo) ** 2 / 12.0))
    spread = np.sqrt(max(variance / total, 0.0))

    loudest = int(np.argmax(magnitude))
    frequency = np.array([tile_frequency(a, b) for a, b in zip(f_lo, f_hi)])
    frequency = np.maximum(frequency, np.finfo(float).tiny)

    start = float(t_lo.min())
    # The support is what a marginal coefficient can stretch without carrying
    # energy; these follow the energy instead. The frequency quantiles are taken
    # in log frequency, the coordinate the dyadic tiling is uniform in.
    t05, t95 = energy_quantile(t_lo, t_hi, energy, (0.05, 0.95))
    logf05, logf95 = energy_quantile(
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
