"""Checks a written mock set is what its truth table says it is.

A benchmark is only as good as the claim that its noise is the noise it was
meant to be and that each injection carries the amplitude recorded for it. Both
are measured here on the frames as written, read back the way a search reads
them, rather than on the arrays before they were written:

* the spectrum, at a resolution fine enough to show a line, band by band,
  against the flat density of unit-variance white noise --- never the overall
  variance, which a spectrum can have exactly right while being wrong
  everywhere;
* Gaussianity band by band, since noise that is Gaussian overall can hide a
  band that is not;
* the amplitude of every injection in every detector, by filtering the data
  with the true template, regenerated from the truth table's parameters and
  not taken from the generator's memory.

The last check has two parts because it answers two questions. Filtering the
difference between the foreground and the background --- which is the injection
alone, since the two share one noise realisation --- must return the recorded
signal-to-noise ratio to the precision of the arithmetic: this is what says the
table and the frames agree. Filtering the foreground returns the recorded value
plus the noise's projection on the template, which is a standard normal: this
is what says the noise the signal sits in is the noise the value was computed
against. A recovered value within a few per cent of the injected one, one
injection at a time, is the first test and not the second; at a signal-to-noise
ratio of eight the noise alone moves the second by an eighth.
"""
from __future__ import annotations

import os

import numpy as np

from wdf.mock.dataset import _polarisations, _sensitivity, project_cbc

DEFAULT_BANDS = ((8.0, 16.0), (16.0, 32.0), (32.0, 64.0), (64.0, 128.0),
                 (128.0, 256.0), (256.0, 512.0), (512.0, 1000.0))


def read_ffl(ffl, channel, start=None, end=None):
    """Read the stretch an FFL index lists as one series.

    :type ffl: str
    :param ffl: the index, one ``path gps length 0 0`` line per frame, as
        :func:`wdf.mock.dataset._write_frames` writes it.
    :type channel: str
    :param channel: the channel to read, prefix included.
    :type start: float | None
    :param start: first GPS time to return; the index's first when None.
    :type end: float | None
    :param end: GPS time to stop at; the index's last when None.
    :return: tuple -- ``(samples, sample_rate, t0)``.
    """
    from gwpy.timeseries import TimeSeries

    with open(ffl, encoding="utf-8") as handle:
        entries = [line.split() for line in handle if line.strip()]
    paths = []
    for path, gps, length, *_ in entries:
        gps, length = float(gps), float(length)
        if start is not None and gps + length <= start:
            continue
        if end is not None and gps >= end:
            continue
        paths.append(path)
    series = TimeSeries.read(paths, channel, start=start, end=end)
    return (np.asarray(series.value, dtype=float),
            float(series.sample_rate.value), float(series.t0.value))


def spectrum_check(samples, sample_rate, resolution=1.0 / 16.0,
                   bands=DEFAULT_BANDS):
    """How flat the spectrum of a white series is, band by band.

    The spectrum is averaged over consecutive, non-overlapping, unwindowed
    segments of ``1 / resolution`` seconds. For white noise this is exact ---
    there is no leakage to guard against --- and it makes every frequency bin an
    average of `K` independent squared normals, so its ratio to the flat
    density has mean one and scatter ``1 / sqrt(K)`` exactly, and a band of
    `M` bins has a mean known to ``1 / sqrt(K M)``. Each band is reported
    against those two numbers, so a departure is read in units of what chance
    allows and not by eye.

    :type samples: numpy.ndarray
    :param samples: the series, unit variance expected.
    :type sample_rate: float
    :param sample_rate: its rate, Hz.
    :type resolution: float
    :param resolution: frequency resolution, Hz.
    :param bands: ``(low, high)`` pairs in Hz.
    :return: list[dict] -- one per band: ``band``, ``mean_ratio``,
        ``mean_sigma`` (what chance allows the mean), ``bin_scatter`` and
        ``bin_sigma`` (the scatter of single bins and what chance allows it),
        ``max_ratio`` and ``n_bins``.
    """
    from scipy.signal import welch

    nperseg = int(round(float(sample_rate) / float(resolution)))
    frequency, density = welch(np.asarray(samples, dtype=float),
                               fs=float(sample_rate), window="boxcar",
                               nperseg=nperseg, noverlap=0, detrend=False)
    n_segments = len(samples) // nperseg
    ratio = density / (2.0 / float(sample_rate))
    out = []
    for low, high in bands:
        inside = (frequency >= low) & (frequency < high)
        values = ratio[inside]
        out.append({
            "band": (float(low), float(high)),
            "mean_ratio": float(values.mean()),
            "mean_sigma": float(1.0 / np.sqrt(n_segments * values.size)),
            "bin_scatter": float(values.std()),
            "bin_sigma": float(1.0 / np.sqrt(n_segments)),
            "max_ratio": float(values.max()),
            "n_bins": int(values.size),
        })
    return out


def gaussianity_check(samples, sample_rate, bands=DEFAULT_BANDS,
                      chunk_s=256.0):
    """How Gaussian a series is inside each band.

    Each band is cut out with an ideal filter, applied in the frequency domain
    chunk by chunk, and its samples normalised by their own deviation. Their
    excess kurtosis and the share of them beyond three deviations are then
    compared with what a Gaussian allows for the number of independent samples
    the band holds, ``2 B T`` for a band `B` wide over `T` seconds --- not the
    number of samples, which a narrow band makes strongly correlated. The
    largest excursion is reported beside the one a Gaussian of that many
    independent samples would typically reach.

    Skewness is not reported, because inside an octave it is zero whatever the
    noise: the third moment of a band-limited series is a sum over triples of
    its frequencies adding to zero, and no two frequencies of a band narrower
    than an octave add up to a third one inside it.

    :type samples: numpy.ndarray
    :param samples: the series.
    :type sample_rate: float
    :param sample_rate: its rate, Hz.
    :param bands: ``(low, high)`` pairs in Hz.
    :type chunk_s: float
    :param chunk_s: chunk length the band filter is applied over, seconds.
    :return: list[dict] -- one per band: ``band``, ``excess_kurtosis`` and
        ``kurtosis_sigma``, ``tail_3sigma`` and ``tail_3sigma_expected`` with
        ``tail_3sigma_sigma``, ``variance`` and ``variance_expected``,
        ``max_abs`` and ``max_abs_expected``.
    """
    x = np.asarray(samples, dtype=float)
    n_chunk = int(round(float(chunk_s) * float(sample_rate)))
    n_chunks = x.size // n_chunk
    frequency = np.fft.rfftfreq(n_chunk, 1.0 / float(sample_rate))
    sums = {band: np.zeros(3) for band in bands}
    tails = {band: 0 for band in bands}
    peak = {band: 0.0 for band in bands}
    for k in range(n_chunks):
        spectrum = np.fft.rfft(x[k * n_chunk:(k + 1) * n_chunk])
        for band in bands:
            mask = (frequency >= band[0]) & (frequency < band[1])
            y = np.fft.irfft(spectrum * mask, n_chunk)
            sums[band] += (y.sum(), (y ** 2).sum(), (y ** 4).sum())
            # The band's measured deviation is known only once every chunk is
            # in; the tail is counted against the one unit-variance white noise
            # gives the band, its share of the spectrum, which is reported
            # beside the measured variance so the two can be told apart.
            expected_sd = np.sqrt(2.0 * (band[1] - band[0]) / float(sample_rate))
            tails[band] += int((np.abs(y) > 3.0 * expected_sd).sum())
            peak[band] = max(peak[band], float(np.abs(y).max()))
    n = float(n_chunks * n_chunk)
    duration = n / float(sample_rate)
    out = []
    for band in bands:
        # The band has no mean: its zero frequency is outside it.
        _, variance, s4 = sums[band] / n
        sd = np.sqrt(variance)
        independent = 2.0 * (band[1] - band[0]) * duration
        tail_expected = 2.0 * 0.0013498980316301
        out.append({
            "band": (float(band[0]), float(band[1])),
            "excess_kurtosis": float(s4 / variance ** 2 - 3.0),
            "kurtosis_sigma": float(np.sqrt(24.0 / independent)),
            "tail_3sigma": float(tails[band] / n),
            "tail_3sigma_expected": tail_expected,
            "tail_3sigma_sigma": float(np.sqrt(
                tail_expected * (1.0 - tail_expected) / independent)),
            "variance": float(variance),
            "variance_expected": float(2.0 * (band[1] - band[0])
                                       / float(sample_rate)),
            "max_abs": float(peak[band] / sd),
            "max_abs_expected": float(np.sqrt(2.0 * np.log(independent))),
        })
    return out


def true_template(row, ifo, sample_rate, relative_sensitivity=None):
    """One detector's injected waveform, rebuilt from its truth-table row.

    Generated again from the recorded masses, spins, inclination, sky position
    and polarisation, projected onto `ifo`, and multiplied by the detector's
    relative sensitivity: everything but the overall scale, which is what the
    check this serves is about.

    :param row: the injection's row, as a mapping.
    :type ifo: str
    :param ifo: the detector.
    :type sample_rate: float
    :param sample_rate: rate of the frames, Hz.
    :type relative_sensitivity: dict | None
    :param relative_sensitivity: ``{ifo: factor}`` the set was written with.
    :return: tuple -- ``(template, gps_first_sample)``: the waveform in units
        of an arbitrary scale, and the GPS time of its first sample in `ifo`.
    """
    spec = {key: row[key] for key in (
        "category", "mass1", "mass2", "spin1z", "spin2z", "inclination",
        "f_lower", "approximant")}
    hp, hc, start_offset = _polarisations(spec, int(sample_rate))
    projected = project_cbc(hp, hc, float(row["ra"]), float(row["dec"]),
                            float(row["polarization"]), float(row["gps"]),
                            (ifo,))
    strain, arrival = projected[ifo]
    strain = strain * _sensitivity(relative_sensitivity, ifo)
    return strain, float(arrival) + float(start_offset)


def matched_filter_check(foreground, background, t0, sample_rate, truth, ifo,
                         relative_sensitivity=None, max_lag_s=0.01):
    """Recover every injection's amplitude in one detector with its template.

    For each compact binary the template is rebuilt from the truth table
    (:func:`true_template`), normalised to unit norm --- in unit-variance white
    noise the matched filter is the plain inner product of the samples --- and
    read against three series at the recorded arrival:

    * ``injected``: the foreground less the background, which is the injection
      alone; it must equal the recorded value.
    * ``recovered``: the foreground at the true time and phase, the recorded
      value plus a standard normal.
    * ``peak``: the foreground maximised over phase, with the quadrature
      template, and over lags within ``max_lag_s``, which is the number a
      search that knew the template would report; it lies above the recorded
      value by the maximisation's own bias.

    The lag at which the injection alone peaks is returned too, in samples: it
    is zero when the arrival time the table records is the one the frames hold.

    :type foreground: numpy.ndarray
    :param foreground: the detector's foreground samples.
    :type background: numpy.ndarray
    :param background: its background samples, on the same times.
    :type t0: float
    :param t0: GPS time of the first sample of both.
    :type sample_rate: float
    :param sample_rate: their rate, Hz.
    :param truth: the truth table, a pandas.DataFrame.
    :type ifo: str
    :param ifo: the detector.
    :type relative_sensitivity: dict | None
    :param relative_sensitivity: ``{ifo: factor}`` the set was written with.
    :type max_lag_s: float
    :param max_lag_s: half-width of the lag search, seconds.
    :return: pandas.DataFrame -- one row per injection: ``injection_id``,
        ``detector``, ``snr``, ``injected``, ``recovered``, ``peak``,
        ``injected_lag``.
    """
    import pandas as pd
    from scipy.signal import fftconvolve, hilbert

    rate = float(sample_rate)
    max_lag = int(round(float(max_lag_s) * rate))
    rows = []
    for _, row in truth[truth["category"] == "cbc"].iterrows():
        template, gps_first = true_template(row, ifo, rate, relative_sensitivity)
        norm = float(np.sqrt(template @ template))
        unit = template / norm
        i0 = int(np.rint((gps_first - float(t0)) * rate))
        span = slice(i0, i0 + unit.size)
        fg = foreground[span]
        injected_only = fg - background[span]

        wide = slice(i0 - max_lag, i0 + unit.size + max_lag)
        analytic = hilbert(unit)
        kernel = np.conj(analytic[::-1])
        lags_fg = fftconvolve(foreground[wide], kernel, mode="valid")
        lags_inj = fftconvolve(foreground[wide] - background[wide],
                               unit[::-1], mode="valid")
        rows.append({
            "injection_id": int(row["injection_id"]),
            "detector": ifo,
            "snr": float(row[f"snr_{ifo}"]),
            "injected": float(unit @ injected_only),
            "recovered": float(unit @ fg),
            "peak": float(np.abs(lags_fg).max()),
            "injected_lag": int(np.argmax(lags_inj)) - max_lag,
        })
    return pd.DataFrame(rows)


def light_travel_check(truth, detectors):
    """Largest arrival-time difference of each pair, against its light travel time.

    :param truth: the truth table.
    :param detectors: the detectors.
    :return: dict -- ``{"H1-L1": (largest |dt|, light travel time)}`` in seconds.
    """
    from pycbc.detector import Detector

    cbc = truth[truth["category"] == "cbc"]
    out = {}
    for i, a in enumerate(detectors):
        for b in detectors[i + 1:]:
            dt = float((cbc[f"gps_{a}"] - cbc[f"gps_{b}"]).abs().max())
            bound = float(Detector(a).light_travel_time_to_detector(Detector(b)))
            out[f"{a}-{b}"] = (dt, bound)
    return out


def validate_white_set(outdir, detectors, channel_suffix="MOCK-STRAIN",
                       relative_sensitivity=None, bands=DEFAULT_BANDS):
    """Run every check on a white mock set as written to `outdir`.

    One detector is held at a time, foreground and background together.

    :type outdir: str
    :param outdir: where :func:`wdf.mock.generate_dataset` wrote the set.
    :param detectors: its detectors.
    :type channel_suffix: str
    :param channel_suffix: channel name after the detector prefix.
    :type relative_sensitivity: dict | None
    :param relative_sensitivity: ``{ifo: factor}`` the set was written with.
    :param bands: bands the noise is checked in, Hz.
    :return: dict -- ``spectrum`` and ``gaussianity``, ``{ifo: {kind: [...]}}``
        for the foreground and the background; ``matched_filter``, a
        pandas.DataFrame over every injection and detector; and
        ``light_travel``.
    """
    import pandas as pd

    truth = pd.read_parquet(os.path.join(outdir, "injections.parquet"))
    spectrum, gaussianity, filters = {}, {}, []
    for ifo in detectors:
        channel = f"{ifo}:{channel_suffix}"
        series = {}
        for kind in ("FOREGROUND", "BACKGROUND"):
            ffl = os.path.join(outdir, f"{ifo}-MOCK-{kind}.ffl")
            series[kind], rate, t0 = read_ffl(ffl, channel)
        spectrum[ifo] = {kind: spectrum_check(values, rate, bands=bands)
                         for kind, values in series.items()}
        gaussianity[ifo] = {kind: gaussianity_check(values, rate, bands=bands)
                            for kind, values in series.items()}
        filters.append(matched_filter_check(
            series["FOREGROUND"], series["BACKGROUND"], t0, rate, truth, ifo,
            relative_sensitivity))
        del series
    return {
        "spectrum": spectrum,
        "gaussianity": gaussianity,
        "matched_filter": pd.concat(filters, ignore_index=True),
        "light_travel": light_travel_check(truth, tuple(detectors)),
    }
