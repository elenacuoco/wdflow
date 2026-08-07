"""Synthetic raw-trigger generators shared by the wdf.analysis test suite. No
WDF run required -- fabricates data matching the real eventPE-derived schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_TRIGGER_COLUMNS = [
    "gps", "gpsPeak", "duration", "EnWDF", "snrMean", "snrPeak",
    "freqMin", "freqMean", "freqMax", "freqPeak", "sigma", "wave", "ifo",
]


def synth_raw_triggers(ifo: str, n_background: int, gps0: float, span_s: float,
                        seed: int = 0, burst_gps: float | None = None,
                        burst_snr: float = 15.0, burst_n: int = 6) -> pd.DataFrame:
    """`n_background` isolated low-SNR triggers uniformly spread across
    [gps0, gps0+span_s], plus (if `burst_gps` given) a tight burst of
    `burst_n` triggers around `burst_gps` with high EnWDF -- mimicking a
    real transient's burst of consecutive raw eventPE triggers.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for gp in gps0 + rng.uniform(0, span_s, n_background):
        rows.append(dict(
            gps=gps0, gpsPeak=gp, duration=rng.uniform(0.05, 0.3),
            EnWDF=rng.uniform(3.0, 6.0), snrMean=rng.uniform(0.3, 1.0),
            snrPeak=rng.uniform(0.5, 3.0), freqMin=50.0, freqMean=rng.uniform(60, 240),
            freqMax=250.0, freqPeak=rng.uniform(60, 240), sigma=1.0, wave="BsplineC309", ifo=ifo,
        ))
    if burst_gps is not None:
        for dt in rng.uniform(-0.01, 0.01, burst_n):
            rows.append(dict(
                gps=gps0, gpsPeak=burst_gps + dt, duration=rng.uniform(0.05, 0.3),
                EnWDF=burst_snr + rng.uniform(-1, 1), snrMean=rng.uniform(1, 3),
                snrPeak=rng.uniform(3, 6), freqMin=80.0, freqMean=140.0 + rng.uniform(-5, 5),
                freqMax=200.0, freqPeak=140.0 + rng.uniform(-5, 5), sigma=1.0, wave="BsplineC309", ifo=ifo,
            ))
    df = pd.DataFrame(rows, columns=RAW_TRIGGER_COLUMNS)
    return df.sort_values("gpsPeak").reset_index(drop=True)
