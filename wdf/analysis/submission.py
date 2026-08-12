"""Writing a search's zero-lag triggers in the format a mock data challenge asks
for.

A challenge fixes the format so that every pipeline is read the same way, and
the format is not the one this pipeline works in: it wants one line per trigger
with a start and an end, a false-alarm rate in hertz, and a handful of optional
descriptors that must still occupy their columns. Translating between the two is
mechanical, and it is done here rather than in a notebook so that a submission
can be regenerated from a candidate table without anyone remembering the
conventions.

Two of those conventions carry a statement and not a formatting rule.

A false-alarm rate quoted below what the background can resolve is not a
measurement. A background of `T` seconds reaches `1/T` and no lower: asking for
one per ten years from three years of slides means asking for a third of an
accidental, and the honest answer is that the search cannot make the claim, not
a number rounded down to zero. `resolvable_threshold` refuses it.

An absent value is `-999` and never a blank or a zero. A blank shifts the
columns of every line after it, and a zero is a legitimate right ascension.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

# What the challenge asks for, in order. The first two match injections; the
# rest are read for diagnosis and must be present whether or not they are known.
SUBMISSION_COLUMNS = [
    "gps_start", "gps_end", "far_hz", "gps_peak",
    "frequency_start", "frequency_end", "frequency_peak",
    "right_ascension", "declination",
]

# The value standing for "this pipeline does not provide it".
MISSING = -999.0

SECONDS_PER_YEAR = 365.25 * 86400.0

_NAME = re.compile(r"^[a-z0-9-]+$")


def _clean(field: str, value: str) -> str:
    """One naming field, checked against the challenge's rules.

    :type field: str
    :param field: which field is being checked, for the error message.
    :type value: str
    :param value: the value given.
    :return: str -- the value, unchanged.
    :raises ValueError: if it is empty, upper case, or holds an underscore.
    """
    text = str(value)
    if not _NAME.match(text):
        raise ValueError(
            f"{field} must be lower-case letters, digits or hyphens and must "
            f"not hold an underscore, which separates the fields of the file "
            f"name itself; got {value!r}")
    return text


def resolvable_threshold(far_hz: float, livetime_s: float) -> float:
    """Check a false-alarm rate against the background that has to measure it.

    :type far_hz: float
    :param far_hz: the rate a submission is cut at, hertz.
    :type livetime_s: float
    :param livetime_s: background livetime the rate is read from, seconds.
    :return: float -- the expected number of accidentals above that rate.
    :raises ValueError: if fewer than one accidental is expected, in which case
        the background reaches no such rate and quoting one states more than
        was measured.
    """
    expected = float(far_hz) * float(livetime_s)
    if expected < 1.0:
        raise ValueError(
            f"a rate of {far_hz:.3e} Hz (one per "
            f"{1.0 / (far_hz * SECONDS_PER_YEAR):.1f} years) expects "
            f"{expected:.2f} accidentals in {livetime_s / SECONDS_PER_YEAR:.2f}"
            f" years of background: the background reaches no such rate, and a "
            f"submission cut there would quote a number nothing measured. "
            f"Accumulate at least {1.0 / far_hz / SECONDS_PER_YEAR:.1f} years, "
            f"or cut at {1.0 / livetime_s:.3e} Hz")
    return expected


def false_alarm_rate_hz(values, background, livetime_s) -> np.ndarray:
    """The rate at which the background reaches each candidate's statistic.

    Counted from the background itself and never extrapolated. A candidate above
    every accidental is given the rate of one such event over the whole
    livetime, which is an upper limit on its rate and is reported as the
    smallest rate the background can express rather than as zero.

    :param values: each candidate's ranking statistic, larger being more
        signal-like.
    :param background: the same statistic on an accidental population.
    :type livetime_s: float
    :param livetime_s: livetime that accidental population stands for, seconds.
    :return: numpy.ndarray -- one rate per candidate, hertz.
    """
    ranked = np.sort(np.asarray(background, dtype=float))
    values = np.asarray(values, dtype=float)
    above = ranked.size - np.searchsorted(ranked, values, side="left")
    return np.maximum(above, 1) / float(livetime_s)


def submission_triggers(candidates, background, livetime_s, statistic,
                        far_threshold_hz, columns=None) -> pd.DataFrame:
    """The zero-lag triggers a submission carries, in the challenge's columns.

    :type candidates: pandas.DataFrame
    :param candidates: the zero-lag candidates, one row each.
    :param background: the accidental values of `statistic`.
    :type livetime_s: float
    :param livetime_s: the accidental population's livetime, seconds.
    :type statistic: str
    :param statistic: the column the candidates are ranked on.
    :type far_threshold_hz: float
    :param far_threshold_hz: keep only candidates at or below this rate.
    :type columns: dict or None
    :param columns: where each submission column comes from in `candidates`.
        A name absent from `candidates`, or absent from this mapping, is
        written as `MISSING`.
    :return: pandas.DataFrame -- `SUBMISSION_COLUMNS`, ordered by rate.
    :raises ValueError: if the background cannot resolve the threshold, or the
        ranking statistic is missing.
    """
    if statistic not in candidates:
        raise KeyError(f"the candidates carry no {statistic!r} to rank on")
    resolvable_threshold(far_threshold_hz, livetime_s)

    rate = false_alarm_rate_hz(candidates[statistic].to_numpy(dtype=float),
                               background, livetime_s)
    keep = rate <= float(far_threshold_hz)
    rows = candidates.loc[keep].copy()

    mapping = dict(columns or {})
    mapping.setdefault("far_hz", None)
    out = pd.DataFrame(index=rows.index)
    for column in SUBMISSION_COLUMNS:
        source = mapping.get(column)
        if column == "far_hz":
            out[column] = rate[keep]
        elif source is not None and source in rows:
            out[column] = pd.to_numeric(rows[source],
                                        errors="coerce").to_numpy(dtype=float)
        else:
            out[column] = MISSING
        out[column] = out[column].fillna(MISSING)
    return out.sort_values("far_hz").reset_index(drop=True)


def submission_name(kind: str, mdc: str, dataset: str, pipeline: str,
                    date: str) -> str:
    """The file name the challenge requires, with its fields checked.

    :type kind: str
    :param kind: `triggers` or `segments`.
    :param mdc: the challenge's label, such as `o4b-2`.
    :param dataset: the dataset's name, such as `short-0`.
    :param pipeline: the search's name; no underscore, since the underscore
        separates the fields of the name itself.
    :param date: submission date as `YY-MM-DD`.
    :return: str -- the file name.
    :raises ValueError: if any field breaks the naming rules.
    """
    if kind not in ("triggers", "segments"):
        raise ValueError(f"kind must be triggers or segments, got {kind!r}")
    if not re.match(r"^\d{2}-\d{2}-\d{2}$", str(date)):
        raise ValueError(f"date must be YY-MM-DD, got {date!r}")
    fields = "_".join(_clean(name, value) for name, value in
                      (("mdc", mdc), ("dataset", dataset),
                       ("pipeline", pipeline)))
    return f"{kind}_{fields}_{date}.csv"


def write_submission(directory, triggers, segments, mdc, dataset, pipeline,
                     date) -> dict:
    """Write the trigger and segment files a submission consists of.

    The two names differ only in their first field, which is what pairs them:
    a submission whose files disagree describes triggers found in time nobody
    analysed.

    :type directory: str
    :param directory: where to write.
    :type triggers: pandas.DataFrame
    :param triggers: as `submission_triggers` returns.
    :param segments: `(start, end)` pairs of the analysed time, seconds.
    :param mdc: the challenge's label.
    :param dataset: the dataset's name.
    :param pipeline: the search's name.
    :param date: submission date as `YY-MM-DD`.
    :return: dict -- `{"triggers": path, "segments": path}`.
    :raises ValueError: if a trigger falls outside every analysed segment.
    """
    spans = np.asarray([(float(a), float(b)) for a, b in segments],
                       dtype=float).reshape(-1, 2)
    if len(triggers):
        start = triggers.gps_start.to_numpy(dtype=float)
        end = triggers.gps_end.to_numpy(dtype=float)
        inside = np.zeros(len(triggers), dtype=bool)
        for a, b in spans:
            inside |= (start >= a) & (end <= b)
        if not inside.all():
            raise ValueError(
                f"{int((~inside).sum())} of {len(triggers)} triggers fall "
                f"outside every analysed segment: the two files would describe "
                f"different data")

    os.makedirs(directory, exist_ok=True)
    paths = {}
    trigger_path = os.path.join(
        directory, submission_name("triggers", mdc, dataset, pipeline, date))
    triggers[SUBMISSION_COLUMNS].to_csv(trigger_path, index=False,
                                        header=False, float_format="%.6f")
    paths["triggers"] = trigger_path

    segment_path = os.path.join(
        directory, submission_name("segments", mdc, dataset, pipeline, date))
    pd.DataFrame(spans, columns=["gps_start", "gps_end"]).to_csv(
        segment_path, index=False, header=False, float_format="%.6f")
    paths["segments"] = segment_path
    return paths
