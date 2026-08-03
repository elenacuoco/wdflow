"""Adapters from WDF's raw output (trigger files, eventPE objects) to the
plain pandas DataFrame schema the rest of wdfLib operates on.

Keeping this the only module that touches WDF-specific formats/types is what
lets clustering.py / coincidence.py / significance.py / roc.py / gnn.py stay
usable offline on already-saved trigger files, without a wdf/pytsa install.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

# Columns written by wdf.observers.SingleEventPrintFileObserver for every
# trigger (fullPrint=0); wt*/rw* wavelet-coefficient columns are appended at
# fullPrint>=1/2 and are not required by wdfLib, so they aren't listed here.
TRIGGER_COLUMNS = [
    "gps", "gpsPeak", "duration", "EnWDF", "snrMean", "snrPeak",
    "freqMin", "freqMean", "freqMax", "freqPeak", "wave",
]

_READERS = {".parquet": pd.read_parquet, ".csv": pd.read_csv}


def _read_trigger_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(f"unsupported trigger file extension {ext!r}: {path}")
    return reader(path)


def triggers_from_csvs(csv_paths: list[str], ifo: str) -> pd.DataFrame:
    """Concatenate WDF trigger files (`.parquet` or, for older runs, `.csv`)
    for one detector into one DataFrame, tagging every row with the `ifo` it
    came from.
    """
    if not csv_paths:
        raise ValueError(f"no trigger file paths given for ifo={ifo!r}")
    frames = [_read_trigger_file(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    missing = set(TRIGGER_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"trigger files missing expected columns {sorted(missing)}: {csv_paths}")
    df["ifo"] = ifo
    return df


def load_triggers_dir(base_dir: str, ifo: str, pattern: str = "*.parquet") -> pd.DataFrame:
    """Convenience wrapper: glob `base_dir` recursively for trigger files and
    load them via `triggers_from_csvs`. `base_dir` is typically a WDF
    `outdir`/`run`/`ifo` output tree (e.g. `<outdir>/out/offLine/H1/**/*.parquet`).
    Pass `pattern="*.csv"` for output from a pre-Parquet WDF run.
    """
    paths = sorted(glob.glob(os.path.join(base_dir, "**", pattern), recursive=True))
    return triggers_from_csvs(paths, ifo)


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


def apply_wavelet_energy_snr(df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """Replaces `snrPeak`/`snrMean` with `wdfLib.wavelets.wavelet_energy_snr`'s
    energy-based statistic, computed directly from each trigger's own `wt*`
    columns. This is the SNR value wdfLib's downstream clustering,
    coincidence, and GNN scoring use (they all key off `snrPeak`/`snrMean`/
    `snrMax`), so calling this once, right after loading triggers, is what
    makes the energy-based statistic "the" SNR for the rest of the pipeline.

    Requires `wt0..wtN` columns (`fullPrint >= 1`). Run this *after*
    `clean_triggers` -- the numerical-artifact guard there checks the
    original `snrPeak`, which this function overwrites. The original values
    are kept under `snrPeak_legacy`/`snrMean_legacy` so nothing is silently
    lost.
    """
    from wdf.analysis.wavelets import wavelet_energy_snr

    wt_cols = sorted((c for c in df.columns if c.startswith("wt") and c[2:].isdigit()),
                      key=lambda c: int(c[2:]))
    if not wt_cols:
        raise ValueError("no wt* columns -- rerun WDF with fullPrint >= 1")

    snr_values = df[wt_cols].apply(
        lambda row: wavelet_energy_snr(row.to_numpy(dtype=float), sigma)["snr"], axis=1,
    )
    out = df.copy()
    out["snrPeak_legacy"] = out["snrPeak"]
    out["snrMean_legacy"] = out["snrMean"]
    out["snrPeak"] = snr_values
    out["snrMean"] = snr_values
    return out


def triggers_from_eventPE(events: list, ifo: str) -> pd.DataFrame:
    """Vectorize a list of live `wdf.structures.eventPE.eventPE` objects
    (e.g. collected via a custom Observer instead of round-tripping through
    CSV) into the same DataFrame schema as `triggers_from_csvs`.

    Only touches wdf types at the boundary; downstream wdfLib code never
    sees eventPE instances.
    """
    rows = [
        {col: getattr(ev, col) for col in TRIGGER_COLUMNS}
        for ev in events
    ]
    df = pd.DataFrame(rows, columns=TRIGGER_COLUMNS)
    df["ifo"] = ifo
    return df
