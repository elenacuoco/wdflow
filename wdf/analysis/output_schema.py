"""Trigger-schema guard for frames loaded from disk.

Trigger files written before the statistic was named consistently carry the WDF
statistic under `snrMax` (and, older still, `mSNR`). Everything downstream reads
`EnWDF`, so a frame is brought onto that name once, at the boundary, rather than
each consumer accepting several.
"""
from __future__ import annotations

import pandas as pd

LEGACY_STATISTIC_COLUMNS = ("snrMax", "mSNR")


def ensure_enwdf_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Return `frame` with the WDF statistic available as `EnWDF`.

    :type frame: pandas.DataFrame
    :param frame: triggers or events loaded from a trigger file.
    :return: pandas.DataFrame -- a copy carrying an `EnWDF` column.
    :raises KeyError: if neither `EnWDF` nor a legacy statistic column is
        present, which means the frame is not a WDF trigger table.
    """
    out = frame.copy()
    if "EnWDF" in out:
        return out

    for column in LEGACY_STATISTIC_COLUMNS:
        if column in out:
            out["EnWDF"] = pd.to_numeric(out[column], errors="coerce")
            return out

    raise KeyError(
        "no WDF statistic column: expected 'EnWDF', or "
        f"{' or '.join(repr(c) for c in LEGACY_STATISTIC_COLUMNS)} in a file "
        "written before the statistic was renamed"
    )
