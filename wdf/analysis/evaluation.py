"""Comparing ranking statistics on the same footing.

A search is judged by how many real signals it keeps at a false-alarm rate one
is willing to tolerate, not by how many candidates it produces. Two statistics
are therefore compared by fixing that rate on the background, reading off the
threshold each one needs to reach it, and asking how many injections survive.

The split used to train a learned statistic belongs here too, because getting
it wrong is what makes such a comparison meaningless: a model scored on the
events it was fitted on reports its own memory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(frame: pd.DataFrame, time_column: str = "gps_candidate",
                   train_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Split a candidate list in two along time.

    Splitting at random is not enough when the rows are candidate edges of a
    graph: two edges can share a node, so the same event lands on both sides
    and the model is scored on something it has already seen. Cutting the
    segment in time keeps whole events together on one side.

    :type frame: pandas.DataFrame
    :param frame: candidates, one row each.
    :type time_column: str
    :param time_column: the column to split on.
    :type train_fraction: float
    :param train_fraction: fraction of the *span*, not of the rows, that goes
        to the training side. A quiet stretch therefore contributes few rows,
        which is the honest behaviour.
    :return: tuple[numpy.ndarray, numpy.ndarray] -- boolean train and test
        masks, over the rows of `frame` in their given order.
    :raises ValueError: if the column is missing or the span is degenerate.
    """
    if time_column not in frame:
        raise ValueError(f"no {time_column!r} column to split on")

    time = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(dtype=float)
    finite = time[np.isfinite(time)]
    if finite.size == 0 or finite.min() == finite.max():
        raise ValueError("the candidates span no time to split")

    boundary = finite.min() + float(np.clip(train_fraction, 0.0, 1.0)) * (
        finite.max() - finite.min())
    train = np.isfinite(time) & (time < boundary)
    return train, np.isfinite(time) & ~train


def threshold_at_far(background_scores, livetime_days: float,
                     far_per_day: float) -> float:
    """The threshold a statistic must exceed to reach a given false-alarm rate.

    :type background_scores: array-like
    :param background_scores: the statistic on background candidates.
    :type livetime_days: float
    :param livetime_days: background livetime the scores were collected over.
    :type far_per_day: float
    :param far_per_day: the tolerated rate.
    :return: float -- the threshold; `-inf` when the background is quieter than
        the requested rate even with no cut, and `nan` when the background
        livetime is too short to resolve the rate at all or when there is no
        background to read it from.
    """
    scores = np.asarray(background_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0 or livetime_days <= 0:
        # No background is not a quiet background: with -inf every candidate
        # would pass and the efficiency would read one, as a measurement.
        return float("nan")

    allowed = far_per_day * livetime_days
    if allowed >= scores.size:
        return float("-inf")
    if allowed < 1.0:
        # The requested rate allows less than one background event over this
        # livetime, so it cannot be resolved: any threshold above the loudest
        # background is consistent with it, and returning that maximum would
        # look like a measurement while being an artefact of the livetime --
        # worse, a statistic whose values pile up at its own maximum would then
        # score every tied candidate as recovered.
        return float("nan")

    # The k-th largest background score is exceeded by exactly k events.
    return float(np.sort(scores)[-int(np.floor(allowed))])


def efficiency_at_far(foreground_scores, background_scores, n_injections: int,
                      livetime_days: float, far_per_day: float) -> dict:
    """Fraction of injections recovered at a fixed false-alarm rate.

    :type foreground_scores: array-like
    :param foreground_scores: the statistic on the candidates that were matched
        to an injection, one per recovered injection.
    :type background_scores: array-like
    :param background_scores: the statistic on background candidates.
    :type n_injections: int
    :param n_injections: how many injections were made, which is the
        denominator -- an injection that produced no candidate at all counts
        against the efficiency and must not be silently dropped.
    :type livetime_days: float
    :param livetime_days: background livetime.
    :type far_per_day: float
    :param far_per_day: the tolerated rate.
    :return: dict -- `far_per_day`, `threshold`, `n_found`, `efficiency` and
        `measurable`, the last being False when the background livetime cannot
        resolve the requested rate, in which case the efficiency is `nan`
        rather than a number that would be read as a measurement.
    """
    threshold = threshold_at_far(background_scores, livetime_days, far_per_day)
    if not np.isfinite(threshold) and np.isnan(threshold):
        return dict(far_per_day=float(far_per_day), threshold=float("nan"),
                    n_found=0, efficiency=float("nan"), measurable=False)

    scores = np.asarray(foreground_scores, dtype=float)
    n_found = int(np.sum(np.isfinite(scores) & (scores >= threshold)))
    return dict(
        far_per_day=float(far_per_day),
        threshold=threshold,
        n_found=n_found,
        efficiency=float(n_found / n_injections) if n_injections else float("nan"),
        measurable=True,
    )


def compare_statistics(foreground: pd.DataFrame, background: pd.DataFrame,
                       statistics, n_injections: int, livetime_days: float,
                       far_targets=(1.0, 1.0 / 7.0)) -> pd.DataFrame:
    """Efficiency of several ranking statistics at the same false-alarm rates.

    Every statistic is read from both frames, so foreground and background are
    ranked on the same quantity -- the condition for the comparison to mean
    anything.

    :type foreground: pandas.DataFrame
    :param foreground: candidates matched to an injection, one row per
        recovered injection.
    :type background: pandas.DataFrame
    :param background: background candidates.
    :type statistics: iterable[str]
    :param statistics: the columns to compare.
    :type n_injections: int
    :param n_injections: total injections made.
    :type livetime_days: float
    :param livetime_days: background livetime.
    :type far_targets: iterable[float]
    :param far_targets: false-alarm rates, per day.
    :return: pandas.DataFrame -- one row per statistic and rate.
    :raises KeyError: if a statistic is missing from either frame.
    """
    rows = []
    for name in statistics:
        for frame, label in ((foreground, "foreground"), (background, "background")):
            if name not in frame:
                raise KeyError(f"{name!r} is not a column of the {label} candidates")
        for far in far_targets:
            row = efficiency_at_far(foreground[name], background[name],
                                    n_injections, livetime_days, far)
            row["statistic"] = name
            rows.append(row)
    return pd.DataFrame(rows)[
        ["statistic", "far_per_day", "threshold", "n_found", "efficiency",
         "measurable"]]
