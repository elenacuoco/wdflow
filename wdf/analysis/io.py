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

# Every column wdf.observers.SingleEventPrintFileObserver writes for a trigger.
TRIGGER_COLUMNS = list(META_FIELDS) + ["wave", "n_coeff", "fs"] + list(COEFFICIENT_FIELDS)


def triggers_from_files(paths: list[str], ifo: str) -> pd.DataFrame:
    """Concatenate one detector's WDF trigger files, tagging each row with `ifo`.

    :type paths: list[str]
    :param paths: trigger files written by `TriggerWriter`.
    :type ifo: str
    :param ifo: detector name to tag the rows with.
    :return: pandas.DataFrame -- one row per trigger.
    :raises ValueError: if no paths are given, or a file is missing a column.
    """
    if not paths:
        raise ValueError(f"no trigger file paths given for ifo={ifo!r}")
    df = pd.concat([read_triggers(p) for p in paths], ignore_index=True)

    missing = set(TRIGGER_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"trigger files missing expected columns {sorted(missing)}: {paths}")
    df["ifo"] = ifo
    return df


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
