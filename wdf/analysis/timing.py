"""The arrival-time difference of a candidate, read off the reconstructions.

A trigger records two instants, the energy centroid and the centre of the
tile carrying its largest coefficient. Both are moments of the tiling: the
centroid follows how much of the transient survived thresholding in that
detector, and the tile centre cannot resolve better than the tile it sits on.
The reconstruction carries the waveform at the sample, so the lag that
maximises the cross-correlation of two events' stitched reconstructions
measures their arrival-time difference on the morphology the two detectors
share, below the tile.

Nothing here reaches back to the strain: a stitched series is a function of
the coefficients a trigger already carries (see
:func:`wdf.analysis.reconstruction.stitch`), so the estimator runs wherever
the analysis runs and adds nothing to the front end.
"""
from __future__ import annotations

import numpy as np

#: Half-width of the lag search, seconds. A bound is required: two events
#: admitted in coincidence overlap within the light travel time plus their
#: spreads, and an unbounded search would let a distant accidental overlap of
#: unrelated structure win the maximum. The value bounds what the estimator
#: can return, not what a signal can do, and it must comfortably exceed the
#: light travel time of the network it is used on.
MAX_LAG_S = 0.25


def arrival_time_difference(first, second, fs, max_lag_s=MAX_LAG_S):
    """The arrival-time difference of two reconstructed series, in seconds.

    Both series are placed on one absolute time grid before correlating, so
    the result is a difference of arrival times and not a sample offset; a
    common error in placing the pair cancels. The returned uncertainty is the
    one the pair itself declares: the half-width of the correlation peak
    above half its maximum, floored at one sample. That width is a candidate
    definition rather than an established one; coverage against known
    injections is what judges it.

    :type first: tuple
    :param first: ``(gps_start, samples)`` of the first series, as
        :func:`wdf.analysis.reconstruction.stitch` returns it.
    :type second: tuple
    :param second: ``(gps_start, samples)`` of the second series.
    :type fs: float
    :param fs: sampling frequency of both series, Hz.
    :type max_lag_s: float
    :param max_lag_s: half-width of the lag search, seconds; see `MAX_LAG_S`.
    :return: tuple -- ``(dt, sigma)`` in seconds. ``dt`` is positive when the
        first series arrives after the second.
    :raises ValueError: if either series is empty or the sampling frequency
        is not positive.
    """
    (first_start, a), (second_start, b) = first, second
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not len(a) or not len(b):
        raise ValueError("both series must carry samples")
    if not fs > 0:
        raise ValueError(f"sampling frequency must be positive, got {fs}")

    start = min(float(first_start), float(second_start))
    span = max(float(first_start) + len(a) / fs,
               float(second_start) + len(b) / fs) - start
    n = int(np.ceil(span * fs)) + 1
    on_grid_a, on_grid_b = np.zeros(n), np.zeros(n)
    offset_a = int(round((float(first_start) - start) * fs))
    offset_b = int(round((float(second_start) - start) * fs))
    on_grid_a[offset_a:offset_a + len(a)] = a
    on_grid_b[offset_b:offset_b + len(b)] = b

    correlation = np.abs(np.correlate(on_grid_a, on_grid_b, mode="full"))
    lags = np.arange(-(n - 1), n)
    keep = np.abs(lags) <= int(round(max_lag_s * fs))
    correlation, lags = correlation[keep], lags[keep]

    best = int(np.argmax(correlation))
    above = correlation >= 0.5 * correlation[best]
    low = best
    while low > 0 and above[low - 1]:
        low -= 1
    high = best
    while high < len(correlation) - 1 and above[high + 1]:
        high += 1

    dt = float(lags[best]) / fs
    sigma = max((high - low) / 2.0, 1.0) / fs
    return dt, sigma
