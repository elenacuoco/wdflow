"""Gates on the whitened stream: loud instrumental transients taken out before
the search reads them.

A detector's noise holds transients no noise model describes and no search is
meant to report: glitches hundreds of noise standard deviations tall, often
below 100 Hz and unflagged by the detector's own data quality. Left in, one of
them fills every wavelet level it touches with coefficients far above the
threshold, and its ringing, spread by the conditioning, reaches seconds either
side of it.

The gate is applied to the whitened stream, the stream the search reads, and
not to the strain. A gate on the strain is a step in the data the band-pass and
the whitening then filter: both ring on it, and what is left at the edges of
the gated stretch is itself a transient in the whitened stream. On the whitened
stream nothing downstream spreads the gate, and its taper is the only edge
there is.

A transient is found where the whitened stream, broadband or band-passed into
any octave the search reads, stands above `flag` times its own robust standard
deviation. Its extent is every sample above that level, with stretches closer
than `merge_s` joined into one, so it holds the whole of the transient's
excursion in every band; it is gated when its peak, over all bands, reaches the
gate threshold, and its extent is then zeroed and tapered on each side. The
robust standard deviation is a median absolute deviation, which the transients
themselves do not move.

Nothing here depends on what a transient looks like: the census is a level
crossing, band by band, and the gate a threshold on its height.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wdf.filtering import sosfiltfilt

#: Median absolute deviation of a unit Gaussian: the robust standard deviation
#: is the median absolute value divided by it.
MAD_GAUSSIAN = 0.6745

#: Order of the Butterworth filters that split the stream into octaves.
OCTAVE_FILTER_ORDER = 8


def octave_bands(rate, low):
    """The octaves of a stream, from its Nyquist frequency down to `low`.

    Each band is half the one above it, which makes them the bands the levels
    of a dyadic wavelet transform of the stream cover; the lowest is cut at
    `low` when `low` does not fall on an octave edge.

    :type rate: float
    :param rate: sampling rate of the stream, Hz.
    :type low: float
    :param low: lowest frequency, Hz; positive.
    :return: tuple -- `(low, high)` per band, Hz, ascending.
    :raises ValueError: if `low` is not inside `(0, rate / 2)`.
    """
    top = 0.5 * float(rate)
    if not 0.0 < low < top:
        raise ValueError(f"the lowest frequency {low} is not inside (0, {top}) Hz")
    bands = []
    while top > low:
        bands.append((max(0.5 * top, float(low)), top))
        top *= 0.5
    return tuple(reversed(bands))


def octave_filter(rate, band):
    """The Butterworth sections that keep one octave of a stream.

    A high-pass for the band that reaches the Nyquist frequency, a band-pass
    for the others; run forward and backward, as `band_stream` runs them.

    :type rate: float
    :param rate: sampling rate, Hz.
    :type band: tuple
    :param band: `(low, high)`, Hz.
    :return: numpy.ndarray -- second-order sections.
    """
    from scipy.signal import butter

    low, high = band
    if high >= 0.5 * rate:
        return butter(OCTAVE_FILTER_ORDER, low, btype="highpass", fs=rate, output="sos")
    return butter(OCTAVE_FILTER_ORDER, [low, high], btype="bandpass", fs=rate, output="sos")


def band_stream(samples, rate, band):
    """One octave of a stream, filtered with zero phase.

    :type samples: numpy.ndarray
    :param samples: the stream.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :type band: tuple
    :param band: `(low, high)`, Hz.
    :return: numpy.ndarray -- the octave, same length as `samples`.
    """
    return sosfiltfilt(octave_filter(rate, band), np.asarray(samples, dtype=float))


def robust_sigma(samples):
    """The standard deviation of a zero-mean stream, read off its median.

    :type samples: numpy.ndarray
    :param samples: the stream.
    :return: float -- `median(|samples|) / MAD_GAUSSIAN`.
    """
    return float(np.median(np.abs(samples)) / MAD_GAUSSIAN)


@dataclass
class Transients:
    """Where a stream stands above its noise, as sample intervals.

    :param start: first sample of each transient.
    :param stop: one past its last sample.
    :param peak: its height, the largest ratio of the stream to its robust
        standard deviation over the broadband stream and every octave.
    """

    start: np.ndarray
    stop: np.ndarray
    peak: np.ndarray

    def __len__(self):
        return int(self.start.size)


def transients(samples, rate, bands, flag=6.0, merge_s=0.25):
    """The census of a whitened stream's transients.

    At each sample the level is the largest ratio of the stream to its own
    robust standard deviation, taken over the broadband stream and each octave
    in `bands`. The samples above `flag` are grouped into transients, joined
    when fewer than `merge_s` seconds apart.

    :type samples: numpy.ndarray
    :param samples: the whitened stream.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :type bands: tuple
    :param bands: the octaves, as `octave_bands` returns them.
    :type flag: float
    :param flag: level a sample must exceed to belong to a transient, in robust
        standard deviations.
    :type merge_s: float
    :param merge_s: largest gap, seconds, inside one transient.
    :return: Transients
    """
    samples = np.asarray(samples, dtype=float)
    level = np.abs(samples) / robust_sigma(samples)
    for band in bands:
        stream = band_stream(samples, rate, band)
        np.maximum(level, np.abs(stream) / robust_sigma(stream), out=level)

    flagged = np.flatnonzero(level > flag)
    if flagged.size == 0:
        empty = np.zeros(0, dtype=int)
        return Transients(empty, empty, np.zeros(0))
    breaks = np.flatnonzero(np.diff(flagged) > merge_s * rate)
    start = flagged[np.concatenate([[0], breaks + 1])]
    stop = flagged[np.concatenate([breaks, [flagged.size - 1]])] + 1
    # Each maximum runs from one start to the next; the samples between a
    # transient's stop and the next start are set to zero first, and every
    # sample of a transient stands above `flag`, so the maximum is the peak.
    inside = _inside(start, stop, level.size)
    peak = np.maximum.reduceat(np.where(inside, level, 0.0), start)
    return Transients(start, stop, peak)


def _inside(start, stop, size):
    """Mask of the samples that lie inside one of the intervals."""
    edges = np.zeros(size + 1, dtype=int)
    np.add.at(edges, start, 1)
    np.add.at(edges, stop, -1)
    return np.cumsum(edges[:-1]) > 0


def merged(intervals):
    """The union of time intervals, as sorted, disjoint intervals.

    :type intervals: numpy.ndarray
    :param intervals: shape `(n, 2)`, start and stop of each.
    :return: numpy.ndarray -- shape `(m, 2)`, sorted and disjoint.
    """
    intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
    if intervals.shape[0] == 0:
        return intervals
    intervals = intervals[np.argsort(intervals[:, 0])]
    reach = np.maximum.accumulate(intervals[:, 1])
    new = np.concatenate([[True], intervals[1:, 0] > reach[:-1]])
    group = np.cumsum(new) - 1
    stops = np.zeros(group[-1] + 1)
    np.maximum.at(stops, group, intervals[:, 1])
    return np.column_stack([intervals[new, 0], stops])


def gate_weights(times, gates, taper_s):
    """The weight each sample of a stream is multiplied by under the gates.

    Zero inside a gate, one farther than `taper_s` from every gate, and a
    raised cosine in between, which is the inverse of a Tukey window: the
    stream is taken to zero smoothly, so the gate adds no edge of its own.

    :type times: numpy.ndarray
    :param times: GPS time of each sample.
    :type gates: numpy.ndarray
    :param gates: shape `(n, 2)`, GPS start and stop of each gate's zeroed
        stretch.
    :type taper_s: float
    :param taper_s: length of the taper on each side of a gate, seconds.
    :return: numpy.ndarray -- the weights, one per sample.
    """
    times = np.asarray(times, dtype=float)
    gates = merged(gates)
    if gates.shape[0] == 0:
        return np.ones_like(times)
    after = np.clip(np.searchsorted(gates[:, 0], times, side="right") - 1, 0, None)
    before = np.clip(after + 1, None, gates.shape[0] - 1)
    distance = np.minimum(
        np.maximum.reduce([gates[after, 0] - times, times - gates[after, 1],
                           np.zeros_like(times)]),
        np.maximum.reduce([gates[before, 0] - times, times - gates[before, 1],
                           np.zeros_like(times)]))
    if taper_s <= 0.0:
        return (distance > 0.0).astype(float)
    ramp = np.clip(distance / taper_s, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * ramp))
