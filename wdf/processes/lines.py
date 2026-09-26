"""Spectral lines: where a detector's noise has them, and the notches that take
them down to the floor.

A line is a feature far narrower than the broadband noise around it: the power
mains and their harmonics, calibration lines, the violin modes of the
suspensions. Every detector has its own, at its own frequencies and heights,
so they are found per detector, on the stretch its noise model is fitted on,
and removed before the band-pass the detectors share.

The noise model is what the lines are removed for. A model that has them to
represent spends its order on them and fits the floor between them worse, and
a model is only as good as its fit of the floor. What it should be handed is
the floor itself, and each notch is designed to leave exactly that: it cuts a
line down to the floor under it and no further. A notch with its zeros on the
unit circle removes the floor under the line as well, and the hole it leaves is
one more feature for the model to follow; a notch that stops short leaves part
of the line. The depth is therefore matched to the line's measured height.

The notches are second-order sections, stacked onto the band-pass and applied
with it, both ways: the response is the square of each section's magnitude, so
the phase stays identically zero and each pass supplies half of the depth.

Nothing here assumes what a line is made of: it is a narrow excess over a local
floor, found on a median-averaged spectrum, which a transient in the stretch
does not move.
"""
from __future__ import annotations

import numpy as np

#: A line extends over the bins where it carries at least as much power as the
#: floor beneath it, that is, where the spectrum stands above twice the floor.
LINE_EXTENT = 2.0

#: Bins of the spectral estimate spanned by the main lobe of its Hann window. A
#: line narrower than this is measured as this wide, so no notch is narrower.
MAIN_LOBE_BINS = 4


def median_spectrum(samples, sampling, segment_s=16.0):
    """The power spectral density of a stretch, as the median of its periodograms.

    The median rather than the mean, so that a transient inside the stretch,
    which lifts the periodograms of the segments holding it, does not lift the
    floor the lines are measured against.

    :type samples: numpy.ndarray
    :param samples: the stretch, one channel.
    :type sampling: float
    :param sampling: its sampling rate, Hz.
    :type segment_s: float
    :param segment_s: length of each periodogram, seconds; its inverse is the
        frequency resolution.
    :return: tuple -- `(frequency, psd)`, in Hz and in the units of `samples`
        squared per Hz, one-sided.
    :raises ValueError: if the stretch is shorter than one segment.
    """
    from scipy.signal import welch

    samples = np.asarray(samples, dtype=float).reshape(-1)
    nperseg = int(round(segment_s * sampling))
    if samples.size < nperseg:
        raise ValueError(
            f"the stretch holds {samples.size} samples, fewer than one "
            f"segment of {nperseg}")
    return welch(samples, fs=float(sampling), window="hann", nperseg=nperseg,
                 noverlap=nperseg // 2, average="median", detrend="constant")


def spectral_lines(frequency, psd, low, high, threshold=5.0, floor_hz=8.0,
                   widen=1.5):
    """The lines standing above the local floor of a spectrum.

    The floor at each frequency is the running median of the spectrum over
    `floor_hz`, a band much wider than any line, so a line does not raise the
    floor it is measured against. A line is a run of bins above `LINE_EXTENT`
    times the floor; its frequency is the run's highest bin, its height the
    square root of the ratio there (amplitude spectral density over the floor's),
    and it is reported when that height reaches `threshold`. The bandwidth its
    notch is given is `widen` times the run's width, and never less than the
    main lobe of the spectral window.

    :type frequency: numpy.ndarray
    :param frequency: frequencies of `psd`, Hz, ascending and evenly spaced.
    :type psd: numpy.ndarray
    :param psd: the power spectral density, as `median_spectrum` returns it.
    :type low: float
    :param low: lowest frequency searched, Hz.
    :type high: float
    :param high: highest frequency searched, Hz.
    :type threshold: float
    :param threshold: smallest height reported, as a ratio of amplitude
        spectral densities; greater than one.
    :type floor_hz: float
    :param floor_hz: width of the running median that estimates the floor, Hz.
    :type widen: float
    :param widen: bandwidth of a notch over the measured width of its line.
    :return: numpy.ndarray -- shape `(n, 3)`: frequency (Hz), notch bandwidth
        (Hz) and height, one row per line, ascending in frequency.
    :raises ValueError: if `threshold` is not greater than one.
    """
    from scipy.ndimage import median_filter

    if not threshold > 1.0:
        raise ValueError(f"threshold {threshold} does not stand above the floor")
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    psd = np.asarray(psd, dtype=float).reshape(-1)
    step = frequency[1] - frequency[0]
    floor = median_filter(psd, size=int(round(floor_hz / step)) | 1, mode="nearest")
    ratio = psd / np.maximum(floor, np.finfo(float).tiny)

    above = (frequency >= low) & (frequency <= high) & (ratio > LINE_EXTENT)
    edges = np.diff(np.concatenate([[0], above.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if starts.size == 0:
        return np.zeros((0, 3))

    # The highest bin of each run: sort the bins of all runs by run, then by
    # ratio, and take the last of each run.
    run = np.repeat(np.arange(starts.size), stops - starts)
    bins = np.flatnonzero(above)
    order = np.lexsort((ratio[bins], run))
    peaks = bins[order][np.cumsum(stops - starts) - 1]

    height = np.sqrt(ratio[peaks])
    bandwidth = np.maximum(widen * (stops - starts) * step, MAIN_LOBE_BINS * step)
    kept = height >= threshold
    return np.column_stack([frequency[peaks], bandwidth, height])[kept]


def notch_sections(lines, sampling):
    """Second-order sections that cut each line down to the floor, run both ways.

    Each line gets a peaking cut (the equaliser section of R. Bristow-Johnson's
    audio cookbook) centred on it, of the line's bandwidth, whose gain at the
    centre is the inverse square root of the line's height: run forward and then
    backward, the gains multiply, and the line comes out at the floor. Its zeros
    lie inside the unit circle, not on it, so the floor under the line is kept;
    its poles lie inside too, so the section is stable, and away from the line
    its gain is one.

    :type lines: numpy.ndarray
    :param lines: shape `(n, 3)`, frequency (Hz), bandwidth (Hz) and height, as
        `spectral_lines` returns them.
    :type sampling: float
    :param sampling: sampling rate of the data the sections filter, Hz.
    :return: numpy.ndarray -- shape `(n, 6)`, second-order sections as
        `scipy.signal.sosfilt` takes them.
    :raises ValueError: if a line lies outside `(0, sampling / 2)`, has no
        bandwidth, or does not stand above the floor.
    """
    lines = np.asarray(lines, dtype=float).reshape(-1, 3)
    centre, bandwidth, height = lines.T
    if np.any((centre <= 0.0) | (centre >= 0.5 * sampling)):
        raise ValueError(f"a line lies outside (0, {0.5 * sampling}) Hz")
    if np.any(bandwidth <= 0.0):
        raise ValueError("a line has no bandwidth")
    if np.any(height < 1.0):
        raise ValueError("a line does not stand above the floor")

    gain = height ** -0.25
    omega = 2.0 * np.pi * centre / float(sampling)
    alpha = np.sin(omega) * bandwidth / (2.0 * centre)
    cosine = -2.0 * np.cos(omega)
    a0 = 1.0 + alpha / gain
    return np.column_stack([
        (1.0 + alpha * gain) / a0, cosine / a0, (1.0 - alpha * gain) / a0,
        np.ones_like(a0), cosine / a0, (1.0 - alpha / gain) / a0])
