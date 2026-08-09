"""Reconstruct a transient that spans more than one analysis window.

WDF scores one window at a time, so a signal longer than the window is split
across consecutive triggers and each of them carries only the part that fell
inside its own window. For a short burst that costs nothing; for an inspiral
lasting many windows the statistic of the best single window is a small fraction
of the signal-to-noise ratio the whole signal carries.

The windows step by ``window - overlap`` samples, so their step regions tile the
time axis without gaps or double counting. Taking each trigger's contribution
from its own step region therefore gives a single time series that can be scored
as a whole:

    rho = || stitched ||_2 / sigma

which is again the matched-filter signal-to-noise ratio of the reconstruction,
now over the signal's whole extent rather than over one window.
"""
from __future__ import annotations

import numpy as np

from wdf.analysis.coefficients import to_dense
import pandas as pd


def inverse_transform(coefficients, wave):
    """Invert one window's wavelet coefficients back to the time domain.

    :type coefficients: numpy.ndarray
    :param coefficients: the ``wt*`` coefficients of one trigger.
    :type wave: str
    :param wave: name of the basis that produced them, as recorded on the
        trigger.
    :return: numpy.ndarray -- the reconstructed samples.
    """
    from pytsa.tsa import WaveletTransform
    from wdf.structures.array2SeqView import array2SeqView

    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    n = coefficients.size

    view = array2SeqView(0.0, 1.0, n)
    view = view.Fill(0.0, coefficients.copy())
    WaveletTransform(n, getattr(WaveletTransform, wave)).Inverse(view)

    return np.array([view.GetY(0, i) for i in range(n)])


def stitch(triggers, fs, window, overlap):
    """Stitch consecutive triggers into one reconstructed time series.

    Each trigger contributes the step region of its own window, so the pieces
    tile the covered span exactly once. Gaps between non-consecutive triggers
    are filled with zeros, which is what the search says about them.

    :type triggers: pandas.DataFrame
    :param triggers: triggers of one cluster, carrying ``gps``, ``wave`` and the
        wavelet coefficients.
    :type fs: float
    :param fs: analysis sampling frequency, Hz.
    :type window: int
    :param window: analysis window length, samples.
    :type overlap: int
    :param overlap: overlap between consecutive windows, samples.
    :return: ``(gps_start, samples)`` -- GPS time of the first sample and the
        stitched reconstruction.
    """
    from wdf.analysis.coefficients import coefficient_matrix

    if len(triggers) == 0:
        raise ValueError("no triggers to stitch")

    step = int(window) - int(overlap)
    if step <= 0:
        raise ValueError("overlap must be smaller than the window")

    ordered = triggers.sort_values("gps")
    coefficients = coefficient_matrix(ordered)
    times = ordered["gps"].to_numpy(dtype=float)
    waves = ordered["wave"].to_numpy()

    gps_start = float(times[0])
    gps_end = float(times[-1]) + window / float(fs)
    length = int(round((gps_end - gps_start) * fs))

    stitched = np.zeros(length)
    for row in range(len(ordered)):
        samples = inverse_transform(coefficients[row], waves[row])
        first = int(round((times[row] - gps_start) * fs))

        # The step region of this window; the last one contributes to its end.
        stop = min(first + step, length)
        if row == len(ordered) - 1:
            stop = min(first + len(samples), length)
        take = stop - first
        if take > 0:
            stitched[first:stop] = samples[:take]

    return gps_start, stitched


def combined_snr(triggers, fs, window, overlap, sigma_column="sigma"):
    """Signal-to-noise ratio of the reconstruction stitched across windows.

    :type triggers: pandas.DataFrame
    :param triggers: triggers of one cluster.
    :type fs: float
    :param fs: analysis sampling frequency, Hz.
    :type window: int
    :param window: analysis window length, samples.
    :type overlap: int
    :param overlap: overlap between consecutive windows, samples.
    :type sigma_column: str
    :param sigma_column: column holding each trigger's noise scale.
    :return: dict -- `EnWDF` of the whole reconstruction, the loudest single
        window it was built from, the stitched span in seconds, and the number
        of windows that contributed.
    """
    _, stitched = stitch(triggers, fs, window, overlap)
    sigma = float(np.median(triggers[sigma_column].to_numpy(dtype=float)))

    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("noise scale must be positive")

    return dict(
        EnWDF=float(np.linalg.norm(stitched) / sigma),
        sigma=sigma,
        loudest_window=float(triggers.EnWDF.max()),
        span_s=len(stitched) / float(fs),
        windows=int(len(triggers)),
    )


def reconstruct_clusters(triggers, fs, window, overlap, cluster_column="cluster_id"):
    """Combined signal-to-noise ratio for every cluster in a trigger set.

    :type triggers: pandas.DataFrame
    :param triggers: labelled triggers, one row per window.
    :type fs: float
    :param fs: analysis sampling frequency, Hz.
    :type window: int
    :param window: analysis window length, samples.
    :type overlap: int
    :param overlap: overlap between consecutive windows, samples.
    :type cluster_column: str
    :param cluster_column: column holding the cluster label; rows labelled -1
        are singletons and are scored on their own.
    :return: pandas.DataFrame -- one row per cluster.
    """
    rows = []
    for label, group in triggers.groupby(cluster_column):
        summary = combined_snr(group, fs, window, overlap)
        summary.update(cluster_id=label,
                       gps=float(group.gps.min()),
                       gpsPeak=float(group.loc[group.EnWDF.idxmax()].gpsPeak))
        rows.append(summary)

    return pd.DataFrame(rows).sort_values("gps").reset_index(drop=True)


def spectral_centroid(triggers) -> np.ndarray:
    """Each trigger's energy-weighted frequency, from its reconstruction.

    The tile moment cannot resolve inside an octave, because every tile of one
    octave carries the same band; a trigger whose surviving coefficients all
    fall in one octave therefore reports that band's centre, and a fifth of
    triggers do. The reconstruction is not a set of tiles but a time series, and
    its spectrum is not tied to the dyadic ladder: two basis functions of the
    same octave at different times interfere, and the interference depends on
    their spacing, which is exactly the continuous information the tile moment
    discards.

    Only a trigger keeping a single coefficient stays quantised, since one
    coefficient reconstructs to one basis function with a fixed spectrum. About
    one trigger in a hundred is in that state.

    This is an analysis quantity and is deliberately not computed in the search:
    it costs an inverse transform and a Fourier transform per trigger, and the
    front end has to stream in real time. `freqMin` and `freqMax` remain the
    tile support, which is what the band-overlap tests want.

    :type triggers: pandas.DataFrame
    :param triggers: triggers carrying `n_coeff`, `fs`, `wave` and the
        coefficient columns.
    :return: numpy.ndarray -- the centroid in Hz, `nan` where undefined.
    """
    import pandas as pd

    out = np.full(len(triggers), np.nan)
    if not len(triggers):
        return out

    position = {label: slot for slot, label in enumerate(triggers.index)}
    for (n_coeff, fs, wave), group in triggers.groupby(["n_coeff", "fs", "wave"],
                                                       sort=False):
        n_coeff, fs = int(n_coeff), float(fs)
        frequency = np.fft.rfftfreq(n_coeff, 1.0 / fs)
        for label, row in group.iterrows():
            dense = to_dense(np.asarray(row["wt_index"]), np.asarray(row["wt_value"]),
                             n_coeff)
            power = np.abs(np.fft.rfft(inverse_transform(dense, str(wave)))) ** 2
            # The mean carries no frequency and would drag the moment to zero.
            power[0] = 0.0
            total = power.sum()
            if total > 0.0:
                out[position[label]] = float((power * frequency).sum() / total)
    return out
