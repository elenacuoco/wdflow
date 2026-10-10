"""The census of a whitened stream's transients.

Where a whitened stream stands above its own noise, broadband and octave by
octave, as sample intervals. It is what seeds the glitch subtraction
(`wdf.processes.glitch_subtraction`) and what labels a stretch as holding a
transient; it decides nothing by itself.

Ported from the census of wdflow feature/gated-search (9ca09ed), without the
gates: a gate before the whitening spreads its edges over the whitening's
length, which is why the transients are subtracted rather than gated.
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

from dataclasses import dataclass

import numpy as np

from wdf.config.constants import MAD_GAUSSIAN  # noqa: F401 (re-exported)
from wdf.config.defaults import defaults
from wdf.filtering import sosfiltfilt

#: The defaults of the census: table ``[census]`` of wdf/config/defaults.toml.
_CENSUS = defaults("census")

#: Default order of the Butterworth filters that split the stream into octaves
#: (`[census] GlitchOctaveOrder`).
DEFAULT_OCTAVE_FILTER_ORDER = _CENSUS["GlitchOctaveOrder"]


def octave_bands(rate, low):
    """The octaves of a stream, from its Nyquist frequency down to `low`.

    Each band is half the one above it, the bands a dyadic wavelet transform
    of the stream covers; the lowest is cut at `low` when `low` does not fall
    on an octave edge.

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


def octave_filter(rate, band, order=DEFAULT_OCTAVE_FILTER_ORDER):
    """The Butterworth sections that keep one octave of a stream.

    A high-pass for the band that reaches the Nyquist frequency, a band-pass
    for the others; run forward and backward, as `band_stream` runs them.

    :type rate: float
    :param rate: sampling rate, Hz.
    :type band: tuple
    :param band: `(low, high)`, Hz.
    :type order: int
    :param order: filter order.
    :return: numpy.ndarray -- second-order sections.
    """
    from scipy.signal import butter

    low, high = band
    if high >= 0.5 * rate:
        return butter(int(order), low, btype="highpass", fs=rate, output="sos")
    return butter(int(order), [low, high], btype="bandpass", fs=rate, output="sos")


def band_stream(samples, rate, band, order=DEFAULT_OCTAVE_FILTER_ORDER):
    """One octave of a stream, filtered with zero phase.

    :type samples: numpy.ndarray
    :param samples: the stream.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :type band: tuple
    :param band: `(low, high)`, Hz.
    :type order: int
    :param order: order of the octave filter.
    :return: numpy.ndarray -- the octave, same length as `samples`.
    """
    return sosfiltfilt(octave_filter(rate, band, order), np.asarray(samples, dtype=float))


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


def census_level(samples, rate, bands, order=DEFAULT_OCTAVE_FILTER_ORDER):
    """Per sample, the largest ratio of the stream to its robust standard
    deviation, over the broadband stream and each octave of `bands`.

    :type samples: numpy.ndarray
    :param samples: the whitened stream.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :type bands: tuple
    :param bands: the octaves, as `octave_bands` returns them.
    :type order: int
    :param order: order of the octave filters.
    :return: numpy.ndarray -- the level, robust standard deviations.
    """
    samples = np.asarray(samples, dtype=float)
    level = np.abs(samples) / robust_sigma(samples)
    for band in bands:
        stream = band_stream(samples, rate, band, order)
        np.maximum(level, np.abs(stream) / robust_sigma(stream), out=level)
    return level


def transients(samples, rate, bands, flag, merge_s, order=DEFAULT_OCTAVE_FILTER_ORDER):
    """The census of a whitened stream's transients.

    The samples whose `census_level` is above `flag` are grouped into
    transients, joined when fewer than `merge_s` seconds apart.

    :type samples: numpy.ndarray
    :param samples: the whitened stream.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :type bands: tuple
    :param bands: the octaves, as `octave_bands` returns them.
    :type flag: float
    :param flag: level a sample must exceed to belong to a transient, robust
        standard deviations.
    :type merge_s: float
    :param merge_s: largest gap, seconds, inside one transient.
    :type order: int
    :param order: order of the octave filters.
    :return: Transients
    """
    level = census_level(samples, rate, bands, order)
    flagged = np.flatnonzero(level > flag)
    if flagged.size == 0:
        empty = np.zeros(0, dtype=int)
        return Transients(empty, empty, np.zeros(0))
    breaks = np.flatnonzero(np.diff(flagged) > merge_s * rate)
    start = flagged[np.concatenate([[0], breaks + 1])]
    stop = flagged[np.concatenate([breaks, [flagged.size - 1]])] + 1
    inside = _inside(start, stop, level.size)
    peak = np.maximum.reduceat(np.where(inside, level, 0.0), start)
    return Transients(start, stop, peak)


def _inside(start, stop, size):
    """Mask of the samples that lie inside one of the intervals."""
    edges = np.zeros(size + 1, dtype=int)
    np.add.at(edges, start, 1)
    np.add.at(edges, stop, -1)
    return np.cumsum(edges[:-1]) > 0
