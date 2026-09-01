"""Association of search results with known injections, and the efficiency,
recovery and false-alarm numbers that follow from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MATCH_COLUMNS = ["found", "candidate_index", "dt_s", "recovered_snr"]


def candidate_spans(candidates, candidate_time="gpsPeak"):
    """The time each candidate covers, as a `(start, end)` pair of arrays.

    A candidate assembled from several analysis windows covers a stretch, and an
    injection belongs to it when it falls anywhere in that stretch: a chirp is
    recovered by the event that spans it, whose energy centroid sits well before
    the merger. A candidate with no recorded extent covers the single instant it
    reports, so one rule serves both.

    :type candidates: pandas.DataFrame
    :param candidates: search output, one row per candidate.
    :type candidate_time: str
    :param candidate_time: column holding the candidate time, used when the
        candidate records no extent.
    :return: tuple -- `(start, end)`, each an array over the candidates.
    """
    instant = pd.to_numeric(candidates[candidate_time], errors="coerce").to_numpy(float)
    if "gpsStart" not in candidates:
        return instant, instant

    start = pd.to_numeric(candidates["gpsStart"], errors="coerce").to_numpy(float)
    if "gpsEnd" in candidates:
        end = pd.to_numeric(candidates["gpsEnd"], errors="coerce").to_numpy(float)
    elif "duration" in candidates:
        end = start + pd.to_numeric(candidates["duration"], errors="coerce").to_numpy(float)
    else:
        end = start

    start = np.where(np.isfinite(start), start, instant)
    end = np.where(np.isfinite(end), end, start)
    return np.minimum(start, end), np.maximum(start, end)


def match_injections(candidates, injections, window_s=0.5, candidate_time="gpsPeak",
                     candidate_statistic="EnWDF", injection_time="gps"):
    """Associate each injection with the loudest candidate covering it in time.

    An injection counts as found when its time falls inside the stretch a
    candidate covers, widened by `window_s`; among those, the loudest is the one
    recorded. Matching against the candidate's own instant instead would miss a
    long signal recovered correctly, because the event's time sits where its
    energy is and a chirp carries most of its energy before the merger.

    :type candidates: pandas.DataFrame
    :param candidates: search output, one row per candidate.
    :type injections: pandas.DataFrame
    :param injections: ground truth, one row per injection.
    :type window_s: float
    :param window_s: how far outside its own extent a candidate still matches,
        seconds.
    :type candidate_time: str
    :param candidate_time: column of `candidates` holding the candidate time,
        used for `dt_s` and when a candidate records no extent.
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
    start, end = candidate_spans(candidates, candidate_time=candidate_time)

    order = np.argsort(start)
    start_sorted = start[order]
    longest = float(np.nanmax(end - start)) if len(start) else 0.0

    for i, t_inj in enumerate(out[injection_time].to_numpy(dtype=float)):
        # Candidates starting before the injection can still reach it, but no
        # further back than the longest extent any of them has.
        lo = np.searchsorted(start_sorted, t_inj - window_s - longest, side="left")
        hi = np.searchsorted(start_sorted, t_inj + window_s, side="right")
        if hi <= lo:
            continue
        local = order[lo:hi]
        local = local[end[local] >= t_inj - window_s]
        if not len(local):
            continue
        # A candidate whose statistic is not finite is not the loudest one:
        # `argmax` returns the first NaN it meets, which would credit the
        # injection to the one candidate that could not be measured and drop
        # the ones that could.
        measurable = local[np.isfinite(snrs[local])]
        if not len(measurable):
            continue
        best = measurable[np.argmax(snrs[measurable])]
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
    :param window_s: how far outside its own extent a candidate still counts as
        accounted for, seconds.
    :type candidate_time: str
    :param candidate_time: column of `candidates` holding the candidate time,
        used when a candidate records no extent.
    :type injection_time: str
    :param injection_time: column of `injections` holding the injection time.
    :return: pandas.DataFrame -- the subset of `candidates` covering no injection.
    """
    if candidates.empty or injections.empty:
        return candidates.copy()
    inj = np.sort(injections[injection_time].to_numpy(dtype=float))
    start, end = candidate_spans(candidates, candidate_time=candidate_time)

    # An injection anywhere in the widened extent makes the candidate accounted
    # for; the nearest one is the only one that can be.
    reaches = np.searchsorted(inj, start - window_s, side="left")
    reaches = np.clip(reaches, 0, len(inj) - 1)
    accounted = (inj[reaches] >= start - window_s) & (inj[reaches] <= end + window_s)
    return candidates[~accounted].copy()


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


def unclaimed_candidates(candidates, injections, window_s=0.5,
                         candidate_time="gpsPeak", injection_time="gps",
                         statistic=None, limit=None):
    """Candidates that no injection accounts for, loudest first.

    On simulated data every candidate above threshold should be either an
    injection or an accident, and the list this returns should be short and
    should look like the background. On real data it is where a real signal
    would appear: something the search found, that is not one of ours, and that
    the accidental rate does not comfortably explain.

    A candidate is claimed when its own time span, widened by `window_s`,
    contains any injection --- not merely when it is the loudest candidate an
    injection matched. The looser rule is the right one here: a second, weaker
    candidate on the same injection is still that injection's, and calling it
    unexplained would fill the list with our own signals.

    Nothing about the astrophysical origin of what remains is decided here. The
    list is what has to be looked at, against a catalogue and against the
    detectors' state, and it is deliberately short enough to look at.

    :type candidates: pandas.DataFrame
    :param candidates: search output, one row per candidate.
    :type injections: pandas.DataFrame
    :param injections: the injections made, whatever their strength: a weak
        injection that was recovered is still not an unexplained candidate.
    :type window_s: float
    :param window_s: how far outside its own extent a candidate still counts as
        covering an injection, seconds. It must be the tolerance the efficiency
        was measured with, or a candidate can be missing from both lists.
    :type candidate_time: str
    :param candidate_time: column holding the candidate time, used where a
        candidate records no extent.
    :type injection_time: str
    :param injection_time: column holding the injection time.
    :type statistic: str or None
    :param statistic: column to sort by, descending. The order the list is read
        in; None leaves the candidates in the order given.
    :type limit: int or None
    :param limit: keep only this many, after sorting. None keeps all.
    :return: pandas.DataFrame -- the rows of `candidates` that no injection
        covers, sorted and truncated as asked.
    :raises KeyError: if `statistic` is not a column of `candidates`.
    """
    if statistic is not None and statistic not in candidates:
        raise KeyError(f"{statistic!r} is not a column of the candidates")
    if candidates.empty:
        return candidates.copy()

    start, end = candidate_spans(candidates, candidate_time=candidate_time)
    if injections is None or injections.empty:
        claimed = np.zeros(len(candidates), dtype=bool)
    else:
        times = np.sort(
            pd.to_numeric(injections[injection_time], errors="coerce")
            .to_numpy(dtype=float))
        times = times[np.isfinite(times)]
        # An injection inside the widened span claims the candidate. Searching
        # the sorted injection times asks that of every candidate at once,
        # instead of pairing each candidate with every injection.
        first = np.searchsorted(times, start - float(window_s), side="left")
        last = np.searchsorted(times, end + float(window_s), side="right")
        claimed = last > first

    out = candidates.loc[~claimed]
    if statistic is not None:
        out = out.sort_values(statistic, ascending=False)
    return out.head(limit) if limit is not None else out
