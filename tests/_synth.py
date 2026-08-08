"""Synthetic raw-trigger generators shared by the wdf.analysis test suite. No
WDF run required -- fabricates data matching the real eventPE-derived schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

def forward(samples, wave):
    """One window's wavelet coefficients, untouched by thresholding."""
    from pytsa.tsa import WaveletTransform
    from wdf.structures.array2SeqView import array2SeqView

    view = array2SeqView(0.0, 1.0, len(samples))
    view = view.Fill(0.0, np.asarray(samples, dtype=float).copy())
    WaveletTransform(len(samples), getattr(WaveletTransform, wave)).Forward(view)
    return np.array([view.GetY(0, i) for i in range(len(samples))])


def triggers_from_signal(signal, fs, window, overlap, gps0=1000.0, ifo="H1",
                         wave="DaubC12", sigma=1.0):
    """One trigger per analysis window covering `signal`, as WDF would emit.

    :param signal: samples to cut into windows.
    :param fs: sampling frequency, Hz.
    :param window: analysis window length, samples.
    :param overlap: overlap between consecutive windows, samples.
    :param gps0: GPS time of the first sample.
    :param ifo: detector name to tag the rows with.
    :param wave: basis to transform with.
    :param sigma: noise scale to express the amplitudes on.
    :return: pandas.DataFrame -- the trigger schema, one row per window.
    """
    from wdf.analysis.coefficients import from_dense
    from wdf.analysis.metaparameters import meta_features

    step = window - overlap
    rows = []
    for first in range(0, len(signal) - window + 1, step):
        coefficients = forward(signal[first:first + window], wave)
        index, value = from_dense(coefficients)
        gps = gps0 + first / fs
        rows.append(dict(
            meta_features(index, value, window, fs, sigma, gps=gps),
            gps=gps, EnWDF=float(np.linalg.norm(coefficients) / sigma),
            sigma=sigma, wave=wave, n_coeff=window, fs=fs,
            wt_index=index, wt_value=value, ifo=ifo))
    return pd.DataFrame(rows)


RAW_TRIGGER_COLUMNS = [
    "gps", "gpsStart", "gpsCentroid", "tSpread", "gpsPeak", "duration",
    "EnWDF", "sigma", "snrPeak",
    "freqMin", "freqMean", "freqMax", "freqPeak",
    "wave", "n_coeff", "fs", "wt_index", "wt_value", "ifo",
]

NCOEFF = 64
FS = 2048.0


def _trigger(rng, gps0, gps_peak, enwdf, ifo, freq_mean, freq_peak,
             freq_min, freq_max, snr_peak):
    n_nonzero = int(rng.integers(1, 5))
    index = np.sort(rng.choice(NCOEFF, size=n_nonzero, replace=False))
    return dict(
        gps=gps0, gpsStart=gps_peak - 0.05, gpsCentroid=gps_peak,
        tSpread=rng.uniform(0.005, 0.05), gpsPeak=gps_peak,
        duration=rng.uniform(0.05, 0.3),
        EnWDF=enwdf, sigma=1.0, snrPeak=snr_peak,
        freqMin=freq_min, freqMean=freq_mean, freqMax=freq_max, freqPeak=freq_peak,
        wave="BsplineC309", n_coeff=NCOEFF, fs=FS,
        wt_index=index.astype(np.uint16),
        wt_value=rng.normal(scale=enwdf, size=n_nonzero).astype(np.float32),
        ifo=ifo,
    )


def synth_raw_triggers(ifo: str, n_background: int, gps0: float, span_s: float,
                        seed: int = 0, burst_gps: float | None = None,
                        burst_snr: float = 15.0, burst_n: int = 6) -> pd.DataFrame:
    """`n_background` isolated low-SNR triggers uniformly spread across
    [gps0, gps0+span_s], plus (if `burst_gps` given) a tight burst of
    `burst_n` triggers around `burst_gps` with high EnWDF -- mimicking a
    real transient's burst of consecutive raw eventPE triggers.
    """
    rng = np.random.default_rng(seed)
    rows = [
        _trigger(rng, gps0, gp, rng.uniform(3.0, 6.0), ifo,
                 rng.uniform(60, 240), rng.uniform(60, 240), 50.0, 250.0,
                 rng.uniform(0.5, 3.0))
        for gp in gps0 + rng.uniform(0, span_s, n_background)
    ]
    if burst_gps is not None:
        rows += [
            _trigger(rng, gps0, burst_gps + dt, burst_snr + rng.uniform(-1, 1), ifo,
                     140.0 + rng.uniform(-5, 5), 140.0 + rng.uniform(-5, 5),
                     80.0, 200.0, rng.uniform(3, 6))
            for dt in rng.uniform(-0.01, 0.01, burst_n)
        ]
    df = pd.DataFrame(rows, columns=RAW_TRIGGER_COLUMNS)
    return df.sort_values("gpsPeak").reset_index(drop=True)
