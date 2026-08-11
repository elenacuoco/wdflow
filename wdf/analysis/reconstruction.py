"""Reconstruct a transient that spans more than one analysis window.

WDF scores one window at a time, so a signal longer than the window is split
across consecutive triggers and each of them carries only the part that fell
inside its own window. For a short burst that costs nothing; for an inspiral
lasting many windows the statistic of the best single window is a small fraction
of the signal-to-noise ratio the whole signal carries.

The windows step by ``window - overlap`` samples and overlap on the rest, so a
sample in the overlapped region has an estimate from each of the two windows
covering it, and the two differ wherever thresholding kept different
coefficients. Reconciling them --- rather than choosing one, or adding both ---
gives a single time series that can be scored as a whole:

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


OVERLAP_POLICIES = ("overlap_add", "central_window")


def synthesis_weight(window, overlap):
    """The weight one window carries into the stitched series.

    The search applies no window function: it transforms the block as it is, so
    there is no analysis taper for the synthesis to be the counterpart of, and
    the weight below is introduced by the stitching rather than inherited from
    the transform. Its shape is therefore a choice, and what the choice has to
    satisfy is that the weight vanish at the block edges: two windows overlap on
    samples each has its own estimate of, and a weight that does not vanish
    leaves a step wherever the estimates disagree.

    A raised cosine over the overlapped samples, flat in between, is the
    smallest such choice: it is continuous, it is one wherever only a single
    window covers a sample once the sum below is normalized, and its ramp is
    exactly as long as the region where two estimates coexist.

    :type window: int
    :param window: analysis window length, samples.
    :type overlap: int
    :param overlap: overlap between consecutive windows, samples.
    :return: numpy.ndarray -- the weight, one entry per sample of a window.
    """
    window, overlap = int(window), int(overlap)
    weight = np.ones(window)
    if overlap <= 0:
        return weight
    ramp = np.arange(1, overlap + 1) / (overlap + 1.0)
    taper = 0.5 * (1.0 - np.cos(np.pi * ramp))
    weight[:overlap] = taper
    weight[window - overlap:] = taper[::-1]
    return weight


def stitch(triggers, fs, window, overlap, overlap_policy="overlap_add",
           taper_edges=False):
    """Stitch consecutive triggers into one reconstructed time series.

    Consecutive windows overlap, so a sample in the overlapped region has one
    estimate from each window, and the two differ: the windows thresholded
    different populations of coefficients. How the two are reconciled is a
    policy, and it is deliberate rather than implied.

    ``overlap_add`` averages them, weighted by `synthesis_weight` and normalized
    by the weight actually present, so the series crosses from one window's
    estimate to the next continuously and no sample is counted twice.
    ``central_window`` gives each window the samples of its own step, which
    tiles the span exactly once but leaves a step at every boundary wherever the
    two estimates disagree.

    Neither policy sums the duplicated coefficients. Gaps between
    non-consecutive triggers carry no estimate at all and are left zero, which
    is what the search says about them.

    Where coverage begins and ends --- the first and last window, and either
    side of a gap --- there is only one estimate and nothing to cross-fade with,
    so the series steps from zero to whatever that window reconstructs.
    `taper_edges` ramps it instead, over the overlapped length. The two ramps of
    `synthesis_weight` are complementary, so the weight present in the interior
    is exactly one and normalizing by that constant leaves the overlapped
    samples untouched while keeping the ramp at the edges. It is off by default
    because it does attenuate a window with no neighbour on that side, an event
    of one window at both ends, and the statistics read amplitudes off this
    series.

    :type triggers: pandas.DataFrame
    :param triggers: triggers of one cluster, carrying ``gps``, ``wave`` and the
        wavelet coefficients.
    :type fs: float
    :param fs: analysis sampling frequency, Hz.
    :type window: int
    :param window: analysis window length, samples.
    :type overlap: int
    :param overlap: overlap between consecutive windows, samples.
    :type overlap_policy: str
    :param overlap_policy: how overlapping estimates are reconciled, one of
        `OVERLAP_POLICIES`.
    :type taper_edges: bool
    :param taper_edges: ramp the series in and out where coverage begins and
        ends, instead of stepping. Applies to `overlap_add` only.
    :return: ``(gps_start, samples)`` -- GPS time of the first sample and the
        stitched reconstruction.
    :raises ValueError: if there are no triggers, if the overlap does not fit
        inside the window, or if the policy is not one of `OVERLAP_POLICIES`.
    """
    from wdf.analysis.coefficients import coefficient_matrix

    if len(triggers) == 0:
        raise ValueError("no triggers to stitch")
    if overlap_policy not in OVERLAP_POLICIES:
        raise ValueError(
            f"unknown overlap policy {overlap_policy!r}; expected one of "
            f"{', '.join(OVERLAP_POLICIES)}")

    step = int(window) - int(overlap)
    if step <= 0:
        raise ValueError("overlap must be smaller than the window")
    if taper_edges and int(overlap) > int(window) // 2:
        # Beyond half a window a sample is covered by three windows or more and
        # the ramps no longer sum to one, so the interior would be scaled by an
        # amount that varies along the series.
        raise ValueError(
            "taper_edges requires an overlap of at most half the window; "
            f"got {overlap} of {window}")

    ordered = triggers.sort_values("gps")
    coefficients = coefficient_matrix(ordered)
    times = ordered["gps"].to_numpy(dtype=float)
    waves = ordered["wave"].to_numpy()

    gps_start = float(times[0])
    gps_end = float(times[-1]) + window / float(fs)
    length = int(round((gps_end - gps_start) * fs))

    stitched = np.zeros(length)
    weights = np.zeros(length)
    taper = synthesis_weight(window, overlap)

    for row in range(len(ordered)):
        samples = inverse_transform(coefficients[row], waves[row])
        first = int(round((times[row] - gps_start) * fs))

        if overlap_policy == "central_window":
            # The step region of this window; the last one reaches its end.
            stop = min(first + step, length)
            if row == len(ordered) - 1:
                stop = min(first + len(samples), length)
            take = stop - first
            if take > 0:
                stitched[first:stop] = samples[:take]
            continue

        stop = min(first + len(samples), length)
        take = stop - first
        if take > 0:
            stitched[first:stop] += samples[:take] * taper[:take]
            weights[first:stop] += taper[:take]

    if overlap_policy == "overlap_add":
        if taper_edges:
            # The two ramps are complementary, so wherever a sample is covered
            # by its full set of windows the weight already sums to one and
            # nothing has to be divided out. Leaving it undivided is what keeps
            # the ramp standing where a window has no neighbour.
            pass
        else:
            covered = weights > 0.0
            stitched[covered] /= weights[covered]

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


def analytic_signal(samples):
    """The analytic signal, whose modulus and argument are envelope and phase.

    Built by suppressing the negative frequencies of the real series and
    doubling the positive ones, which is the definition rather than an
    approximation to it, so no filter length or window enters.

    :type samples: array-like
    :param samples: a real series.
    :return: numpy.ndarray -- the complex analytic signal, same length.
    """
    samples = np.asarray(samples, dtype=float).reshape(-1)
    n = samples.size
    if n == 0:
        return np.zeros(0, dtype=complex)

    spectrum = np.fft.fft(samples)
    weight = np.zeros(n)
    weight[0] = 1.0
    if n % 2 == 0:
        weight[1:n // 2] = 2.0
        weight[n // 2] = 1.0
    else:
        weight[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * weight)


def waveform_overlap(reconstructed, injected, max_lag=0):
    """How much of a known waveform a reconstruction reproduces.

    The overlap is the normalized inner product

        O = (a . b) / sqrt((a . a)(b . b)),

    which is one only when the two series agree in shape *and* in phase: a
    reconstruction carrying the right envelope with the wrong phase cancels term
    by term and scores near zero, so this is a phase-sensitive measure and not a
    comparison of amplitudes. The series are assumed to be on the same time base
    and, when the search whitens, already whitened; a noise-weighted inner
    product would be the form to use on coloured data.

    :type reconstructed: array-like
    :param reconstructed: the series to test.
    :type injected: array-like
    :param injected: the series it should reproduce, same length.
    :type max_lag: int
    :param max_lag: largest shift, in samples, the overlap may be maximized
        over. Zero asks whether the reconstruction sits where the signal is,
        which is the question when the time base is meant to be exact.
    :return: dict -- `overlap` (at the best lag), `lag` (samples, positive when
        the reconstruction is late) and `overlap_at_zero_lag`.
    :raises ValueError: if the two series have different lengths.
    """
    a = np.asarray(reconstructed, dtype=float).reshape(-1)
    b = np.asarray(injected, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"series of {a.size} and {b.size} samples cannot be "
                         "compared without a common time base")

    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm <= 0.0:
        return dict(overlap=float("nan"), lag=0, overlap_at_zero_lag=float("nan"))

    at_zero = float(a @ b / norm)
    best, best_lag = at_zero, 0
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag == 0:
            continue
        shifted = np.roll(a, -lag)
        value = float(shifted @ b / norm)
        if value > best:
            best, best_lag = value, lag
    return dict(overlap=best, lag=best_lag, overlap_at_zero_lag=at_zero)


def phase_residual(reconstructed, injected, amplitude_floor=0.1):
    """The phase a reconstruction carries, against the phase it should carry.

    Both series are taken to their analytic signal and their unwrapped phases
    subtracted, sample by sample. An aggregate agreement can hide a phase that
    drifts: a reconstruction assembled from pieces, each correct on its own but
    each placed slightly wrong, keeps a high overlap while its phase walks. The
    residual returned here is per sample, so a drift is visible as a slope and a
    mis-stitched boundary as a step.

    The residual is only defined where there is a signal to have a phase, so it
    is returned masked to the samples whose envelope reaches `amplitude_floor`
    of the injected peak.

    :type reconstructed: array-like
    :param reconstructed: the series to test.
    :type injected: array-like
    :param injected: the series it should reproduce, same length.
    :type amplitude_floor: float
    :param amplitude_floor: fraction of the injected envelope's peak below which
        the phase is not defined well enough to be compared.
    :return: dict -- `where` (sample indices kept), `residual` (radians, with
        its median removed, so a constant offset does not read as a drift) and
        `median_abs` over those samples.
    """
    a = analytic_signal(reconstructed)
    b = analytic_signal(injected)
    envelope = np.abs(b)
    peak = float(envelope.max()) if envelope.size else 0.0
    where = np.flatnonzero(envelope >= amplitude_floor * peak) if peak > 0 \
        else np.zeros(0, dtype=int)
    if not where.size:
        return dict(where=where, residual=np.zeros(0),
                    median_abs=float("nan"))

    residual = np.unwrap(np.angle(a[where]) - np.angle(b[where]))
    residual = residual - np.median(residual)
    return dict(where=where, residual=residual,
                median_abs=float(np.median(np.abs(residual))))
