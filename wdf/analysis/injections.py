"""Association of search results with known injections, and the efficiency,
recovery and false-alarm numbers that follow from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MATCH_COLUMNS = ["found", "candidate_index", "dt_s", "recovered_snr"]


def match_injections(candidates, injections, window_s=0.5, candidate_time="gpsPeak",
                     candidate_statistic="EnWDF", injection_time="gps"):
    """Associate each injection with the loudest candidate falling near it in time.

    An injection counts as found when at least one candidate lies within
    `window_s` of its time; among those, the loudest is the one recorded.
    Candidates may match more than one injection only if the injections
    themselves overlap in time.

    :type candidates: pandas.DataFrame
    :param candidates: search output, one row per candidate.
    :type injections: pandas.DataFrame
    :param injections: ground truth, one row per injection.
    :type window_s: float
    :param window_s: half-width of the association window, seconds.
    :type candidate_time: str
    :param candidate_time: column of `candidates` holding the candidate time.
    :type candidate_statistic: str
    :param candidate_statistic: column of `candidates` ranking candidates within the window.
    :type injection_time: str
    :param injection_time: column of `injections` holding the injection time.
    :return: pandas.DataFrame -- `injections` with `found`, `candidate_index`,
        `dt_s` and `recovered_snr` added.
    """
    out = injections.copy().reset_index(drop=True)
    for col, fill in zip(MATCH_COLUMNS, [False, -1, np.nan, np.nan]):
        out[col] = fill

    if candidates.empty:
        return out

    times = candidates[candidate_time].to_numpy(dtype=float)
    snrs = candidates[candidate_statistic].to_numpy(dtype=float)
    order = np.argsort(times)
    times_sorted, index_sorted = times[order], candidates.index.to_numpy()[order]

    for i, t_inj in enumerate(out[injection_time].to_numpy(dtype=float)):
        lo = np.searchsorted(times_sorted, t_inj - window_s, side="left")
        hi = np.searchsorted(times_sorted, t_inj + window_s, side="right")
        if hi <= lo:
            continue
        local = order[lo:hi]
        best = local[np.argmax(snrs[local])]
        out.at[i, "found"] = True
        out.at[i, "candidate_index"] = int(candidates.index.to_numpy()[best])
        out.at[i, "dt_s"] = float(times[best] - t_inj)
        out.at[i, "recovered_snr"] = float(snrs[best])
    return out


def false_alarms(candidates, injections, window_s=0.5, candidate_time="gpsPeak",
                 injection_time="gps"):
    """Candidates that lie near no injection.

    :type candidates: pandas.DataFrame
    :param candidates: search output, one row per candidate.
    :type injections: pandas.DataFrame
    :param injections: ground truth, one row per injection.
    :type window_s: float
    :param window_s: half-width of the association window, seconds.
    :type candidate_time: str
    :param candidate_time: column of `candidates` holding the candidate time.
    :type injection_time: str
    :param injection_time: column of `injections` holding the injection time.
    :return: pandas.DataFrame -- the subset of `candidates` with no injection
        within `window_s`.
    """
    if candidates.empty or injections.empty:
        return candidates.copy()
    inj = np.sort(injections[injection_time].to_numpy(dtype=float))
    times = candidates[candidate_time].to_numpy(dtype=float)
    idx = np.clip(np.searchsorted(inj, times), 1, len(inj) - 1)
    nearest = np.minimum(np.abs(times - inj[idx - 1]), np.abs(times - inj[idx]))
    if len(inj) == 1:
        nearest = np.abs(times - inj[0])
    return candidates[nearest > window_s].copy()


def efficiency(matched, bins=None, injected_snr_column="network_snr", group_column=None):
    """Fraction of injections recovered, in bins of injected signal-to-noise ratio.

    :type matched: pandas.DataFrame
    :param matched: output of `match_injections`.
    :type bins: numpy.ndarray | None
    :param bins: bin edges in SNR; ten equal bins over the data range if None.
    :type injected_snr_column: str
    :param injected_snr_column: column holding the injected optimal SNR, the
        ground truth the mock data set records -- not the search's own `EnWDF`.
    :type group_column: str | None
    :param group_column: if given, compute the curve separately per value of it.
    :return: pandas.DataFrame -- `snr_low`, `snr_high`, `snr_mid`, `n`, `n_found`,
        `efficiency`, plus `group_column` when grouping.
    """
    if bins is None:
        lo, hi = matched[injected_snr_column].min(), matched[injected_snr_column].max()
        bins = np.linspace(lo, hi, 11)
    bins = np.asarray(bins, dtype=float)

    groups = [(None, matched)] if group_column is None else list(matched.groupby(group_column))
    rows = []
    for name, frame in groups:
        idx = np.digitize(frame[injected_snr_column].to_numpy(dtype=float), bins) - 1
        for b in range(len(bins) - 1):
            sel = frame[idx == b]
            row = dict(snr_low=bins[b], snr_high=bins[b + 1],
                       snr_mid=0.5 * (bins[b] + bins[b + 1]),
                       n=len(sel), n_found=int(sel["found"].sum()),
                       efficiency=float(sel["found"].mean()) if len(sel) else np.nan)
            if group_column is not None:
                row[group_column] = name
            rows.append(row)
    return pd.DataFrame(rows)
