"""Adapters from WDF's raw output (trigger files, eventPE objects) to the
plain pandas DataFrame schema the rest of wdfLib operates on.

Keeping this the only module that touches WDF-specific formats/types is what
lets clustering.py / coincidence.py / significance.py / roc.py / gnn.py stay
usable offline on already-saved trigger files, without a wdf/pytsa install.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from wdf.analysis.coefficients import (
    COEFFICIENT_FIELDS,
    META_FIELDS,
    coefficient_matrix,
    read_triggers,
)
from wdf.analysis.robust_events import stride_seconds

# Every column wdf.observers.SingleEventPrintFileObserver writes for a trigger.
TRIGGER_COLUMNS = list(META_FIELDS) + ["wave", "n_coeff", "fs"] + list(COEFFICIENT_FIELDS)


def triggers_from_files(paths: list[str], ifo: str) -> pd.DataFrame:
    """Concatenate one detector's WDF trigger files, tagging each row with `ifo`.

    Each row carries the `stride` of the run that produced it, read from the
    configuration written beside the file. The stride is what the search was
    told to advance by; inferring it from how far apart the triggers landed
    measures the transients instead, and a stretch where only every third
    window fired would report three times the truth.

    :type paths: list[str]
    :param paths: trigger files written by `TriggerWriter`.
    :type ifo: str
    :param ifo: detector name to tag the rows with.
    :return: pandas.DataFrame -- one row per trigger.
    :raises ValueError: if no paths are given, or a file is missing a column.
    :raises FileNotFoundError: if a file has no configuration beside it.
    """
    if not paths:
        raise ValueError(f"no trigger file paths given for ifo={ifo!r}")
    frames = [read_triggers(path) for path in paths]
    missing = set(TRIGGER_COLUMNS) - set(pd.concat(frames, ignore_index=True).columns)
    if missing:
        raise ValueError(f"trigger files missing expected columns {sorted(missing)}: {paths}")

    for path, frame in zip(paths, frames):
        frame["stride"] = stride_seconds(run_parameters(path))
    df = pd.concat(frames, ignore_index=True)
    df["ifo"] = ifo
    return df


def analysed_span(frames, time_column: str = "gpsStart"):
    """The stretch a search actually covered, read from the triggers it wrote.

    The span a configuration declares is not the span a search examines. The
    conditioning chain reads ahead of what it emits and stops before the end of
    the data it is given, and it needs a stretch at the start to fit its noise
    model, so the triggers begin after the declared start and end well before
    the declared end. Dividing a count by the declared span therefore
    understates every false-alarm rate, and counts injections placed past the
    last block as missed when nothing ever looked there.

    Given several detectors' triggers the result is their **intersection**, not
    their union: the livetime that divides a coincidence rate is the stretch
    every detector searched, since a coincidence cannot be formed where one of
    them was not looking.

    The span ends at the last block's start rather than its end. A block is
    short against any livetime worth quoting, and understating by one block is
    the safe direction: it never claims livetime that was not searched.

    :type frames: iterable of pandas.DataFrame
    :param frames: one trigger table per detector, as `triggers_from_files`
        returns.
    :type time_column: str
    :param time_column: the column holding each block's start time, seconds.
    :return: tuple[float, float] -- `(first, last)` GPS. The span is empty,
        with `last == first`, when the detectors have no stretch in common.
    :raises ValueError: if no frames are given, if one of them is empty, or if
        `time_column` is missing from any of them.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("no trigger frames given")

    firsts, lasts = [], []
    for k, frame in enumerate(frames):
        if time_column not in frame:
            raise ValueError(
                f"trigger frame {k} has no {time_column!r} column")
        if not len(frame):
            raise ValueError(
                f"trigger frame {k} is empty: a detector that wrote no trigger "
                "has no analysed span, and treating it as one would put a "
                "livetime under every rate that nothing measured")
        times = frame[time_column].to_numpy(dtype=float)
        firsts.append(float(np.nanmin(times)))
        lasts.append(float(np.nanmax(times)))

    first, last = max(firsts), min(lasts)
    return (first, max(last, first))


def covered_livetime_days(spans) -> float:
    """Total livetime of a set of spans, in days.

    The spans are those `analysed_span` returns, one per analysed stretch. They
    are summed rather than merged: separate stretches of an observing run are
    disjoint by construction, and a set that overlapped would be double-counting
    the same data whatever this did about it.

    :type spans: iterable of tuple[float, float]
    :param spans: `(first, last)` pairs in GPS seconds.
    :return: float -- their total length, days.
    :raises ValueError: if a span runs backwards.
    """
    total = 0.0
    for k, (first, last) in enumerate(spans):
        length = float(last) - float(first)
        if length < 0.0:
            raise ValueError(f"span {k} ends before it starts: ({first}, {last})")
        total += length
    return total / 86400.0


def run_parameters(trigger_path: str):
    """The run configuration a trigger file was produced under.

    A run searching several analysis window lengths writes one configuration
    per length, beside that length's trigger file in the same segment
    directory, and both file names carry the window length.

    :type trigger_path: str
    :param trigger_path: a trigger file written by `TriggerWriter`.
    :return: wdf.config.Parameters.Parameters -- the configuration used.
    :raises FileNotFoundError: if no matching configuration sits beside it.
    """
    from wdf.config.Parameters import Parameters

    match = re.search(r"-Win(\d+)-", os.path.basename(trigger_path))
    if match is None:
        raise FileNotFoundError(
            f"{trigger_path} does not name its analysis window, so the run "
            "configuration written beside it cannot be identified"
        )
    path = os.path.join(os.path.dirname(trigger_path),
                        "parametersUsed-Win%s.json" % match.group(1))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no run configuration at {path}")
    par = Parameters()
    par.load(path)
    return par


def load_triggers_dir(base_dir: str, ifo: str, pattern: str = "*.parquet") -> pd.DataFrame:
    """Glob a WDF output tree for trigger files and load them.

    :type base_dir: str
    :param base_dir: a WDF `outdir`/`run`/`ifo` tree.
    :type ifo: str
    :param ifo: detector name to tag the rows with.
    :type pattern: str
    :param pattern: file name pattern to match under `base_dir`.
    :return: pandas.DataFrame -- one row per trigger.
    """
    paths = sorted(glob.glob(os.path.join(base_dir, "**", pattern), recursive=True))
    return triggers_from_files(paths, ifo)


def clean_triggers(
    df: pd.DataFrame,
    snr_ceiling: float = 500.0,
    edge_guard_s: float = 5.0,
) -> pd.DataFrame:
    """Drop WDF numerical-artifact triggers before clustering/coincidence.

    Confirmed against real WDF output on a real detector trigger set: a
    small number of raw triggers carry `snrPeak` values many orders of
    magnitude above anything physical (e.g. ~1e21, vs. O(1-100) for real
    triggers) -- a WDF-internal numerical artifact, not a real high-SNR
    detection. Left unfiltered, these dominate clustering/coincidence/ROC
    ranking and hide genuine candidates. `snr_ceiling` guards against that,
    plus an edge guard dropping triggers within `edge_guard_s` of the
    analyzed segment's start/end (WDF's own whitening/AR-estimation warm-up
    and edge effects are least reliable there).
    """
    out = df[df["snrPeak"] < snr_ceiling].copy()
    if edge_guard_s > 0 and not out.empty:
        lo, hi = out["gps"].min() + edge_guard_s, out["gps"].max() - edge_guard_s
        out = out[(out["gpsPeak"] > lo) & (out["gpsPeak"] < hi)]
    return out.reset_index(drop=True)


def add_wavelet_energy_diagnostics(df, sigma_column="sigma"):
    """Add the coefficient energy and the statistic it implies, alongside `EnWDF`.

    `EnWDF_from_coeff` is what the search's own statistic should be: the
    coefficient norm on the noise scale. Keeping both makes the agreement
    visible rather than assumed.

    :type df: pandas.DataFrame
    :param df: triggers carrying the coefficient columns.
    :type sigma_column: str
    :param sigma_column: column holding each trigger's noise scale.
    :return: pandas.DataFrame -- `df` with `wavelet_l2`, `wavelet_energy`,
        `nActiveCoeff`, and, where the noise scale is available,
        `EnWDF_from_coeff` and `EnWDF_residual`.
    """
    out = df.copy()
    coefficients = coefficient_matrix(out).astype(float)
    coefficients = np.where(np.isfinite(coefficients), coefficients, 0.0)
    energy = np.einsum("ij,ij->i", coefficients, coefficients)

    out["wavelet_l2"] = np.sqrt(energy)
    out["wavelet_energy"] = energy
    out["nActiveCoeff"] = np.count_nonzero(coefficients, axis=1)

    if sigma_column in out.columns:
        sigma = out[sigma_column].to_numpy(dtype=float)
        valid = np.isfinite(sigma) & (sigma > np.finfo(float).tiny)
        statistic = np.full(len(out), np.nan)
        statistic[valid] = out["wavelet_l2"].to_numpy()[valid] / sigma[valid]
        out["EnWDF_from_coeff"] = statistic
        out["EnWDF_residual"] = out["EnWDF_from_coeff"] - out["EnWDF"]

    return out


def triggers_from_eventPE(events: list, ifo: str) -> pd.DataFrame:
    """Vectorise live `wdf.structures.eventPE.eventPE` objects into a frame.

    Collecting triggers through a custom Observer gives the same frame as
    reading them back from disk, so downstream code never sees eventPE.

    :type events: list
    :param events: the trigger records.
    :type ifo: str
    :param ifo: detector name to tag the rows with.
    :return: pandas.DataFrame -- the same schema `triggers_from_files` returns.
    """
    df = pd.DataFrame([ev.record() for ev in events], columns=TRIGGER_COLUMNS)
    df["ifo"] = ifo
    return df
