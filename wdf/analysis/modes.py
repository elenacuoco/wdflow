"""Detection modes, compared on the rate their own background fixes.

A search can be read at several places along the pipeline: at the window it
found, at the event a detector assembled, or at the candidate a network agreed
on. Each of these is a different population of candidates with a different
background, so their thresholds are not interchangeable and the number of
candidates above a threshold says nothing on its own.

Comparing them means asking each mode the same question --- what fraction of the
injections it recovers at a stated false-alarm rate --- and letting each answer
it with its own background. A mode that produces more candidates then earns no
credit for them: they raise its own threshold. This is what makes the comparison
attribute a gain in sensitivity to the stage that produced it, rather than to
the population size.

The mode is a description of a population, not of an algorithm: what varies
between modes is which candidates are ranked and on what, so a new stage becomes
a new mode without changing anything here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wdf.analysis.evaluation import efficiency_at_far, threshold_at_far


@dataclass
class DetectionMode:
    """One population of candidates, and the statistic it is ranked on.

    :param name: label the mode is reported under.
    :param foreground: candidates matched to injections, one row per recovered
        injection; an injection recovered by no candidate is absent.
    :param background: candidates from data holding no injection, over
        `livetime_days`.
    :param statistic: column of both frames the mode is ranked on. Higher must
        mean more signal-like.
    :param n_injections: injections made, the denominator of the efficiency.
        Injections absent from `foreground` count as missed.
    :param livetime_days: background livetime the rate is measured over, days.
    """

    name: str
    foreground: pd.DataFrame
    background: pd.DataFrame
    statistic: str
    n_injections: int
    livetime_days: float

    def scores(self) -> tuple[np.ndarray, np.ndarray]:
        """The mode's foreground and background statistics.

        :return: tuple -- `(foreground, background)` arrays.
        :raises KeyError: if the statistic is missing from either frame.
        """
        for frame, label in ((self.foreground, "foreground"),
                             (self.background, "background")):
            if self.statistic not in frame:
                raise KeyError(
                    f"{self.name!r} is ranked on {self.statistic!r}, which the "
                    f"{label} candidates do not carry")
        return (self.foreground[self.statistic].to_numpy(dtype=float),
                self.background[self.statistic].to_numpy(dtype=float))


def compare_modes(modes, far_targets=(1.0, 0.1)) -> pd.DataFrame:
    """What each mode recovers at the same false-alarm rates.

    :type modes: iterable[DetectionMode]
    :param modes: the populations to compare.
    :type far_targets: iterable[float]
    :param far_targets: false-alarm rates, per day.
    :return: pandas.DataFrame -- one row per mode and rate, carrying
        `mode`, `statistic`, `far_per_day`, `threshold`, `n_candidates`,
        `n_found`, `efficiency` and `measurable`; `measurable` is False where
        the background is too short to reach the rate asked for, in which case
        the efficiency is an upper limit rather than a measurement.
    """
    rows = []
    for mode in modes:
        foreground, background = mode.scores()
        for far in far_targets:
            row = efficiency_at_far(foreground, background, mode.n_injections,
                                    mode.livetime_days, far)
            row.update(mode=mode.name, statistic=mode.statistic,
                       n_candidates=len(background))
            rows.append(row)
    return pd.DataFrame(rows)[
        ["mode", "statistic", "far_per_day", "threshold", "n_candidates",
         "n_found", "efficiency", "measurable"]]


def mode_roc(mode: DetectionMode, n_points: int = 60) -> pd.DataFrame:
    """A mode's efficiency as a function of its own false-alarm rate.

    The rates are read off the background itself rather than from a grid of
    thresholds, so every point is a rate the background actually reaches and
    none of them is extrapolated.

    :type mode: DetectionMode
    :param mode: the population to trace.
    :type n_points: int
    :param n_points: how many rates to sample.
    :return: pandas.DataFrame -- `far_per_day`, `threshold` and `efficiency`,
        ascending in rate; empty when the mode has no background.
    """
    foreground, background = mode.scores()
    if not len(background) or not len(foreground):
        return pd.DataFrame(
            columns=["far_per_day", "threshold", "ceiling", "efficiency"])

    # A non-finite score is not a candidate: it sorts before every real value
    # in a descending order, so leaving it in would give the lowest rates a
    # threshold of nan, an efficiency of zero, and a rate axis counting
    # candidates that do not exist.
    background = np.asarray(background, dtype=float)
    background = background[np.isfinite(background)]
    if not len(background):
        return pd.DataFrame(
            columns=["far_per_day", "threshold", "ceiling", "efficiency"])
    ranked = np.sort(background)[::-1]
    # The first point would be a threshold on the single loudest accidental,
    # which is a maximum and not a rate: on recorded strain the loudest
    # anything is a glitch, and a curve is not drawn through it.
    counts = np.unique(np.geomspace(1, len(ranked), n_points).astype(int))
    counts = counts[counts > 1]
    # The fraction of the injections some candidate matched at all. It is the
    # value the curve tends to as the threshold falls, so a curve saturating
    # there is limited by what the coincidence admitted and not by the ranking,
    # and no change of statistic moves it.
    ceiling = len(foreground) / max(mode.n_injections, 1)
    rows = []
    for count in counts:
        threshold = float(ranked[count - 1])
        # The rate the threshold realises, not the rank that produced it. A
        # statistic with an atom --- a pair sharing no tile scores exactly
        # zero --- maps a whole range of ranks to one threshold, and quoting
        # the rank would draw a shelf across rates the curve never had.
        rows.append(dict(
            far_per_day=(float(np.count_nonzero(ranked >= threshold))
                         / mode.livetime_days),
            threshold=threshold,
            ceiling=ceiling,
            efficiency=float(np.count_nonzero(foreground >= threshold))
            / max(mode.n_injections, 1)))
    return (pd.DataFrame(rows).drop_duplicates(subset="far_per_day")
            .sort_values("far_per_day").reset_index(drop=True))


def mode_threshold(mode: DetectionMode, far_per_day: float) -> float:
    """The statistic a mode's own background reaches at a stated rate.

    :type mode: DetectionMode
    :param mode: the population to read.
    :type far_per_day: float
    :param far_per_day: the rate the threshold is quoted at.
    :return: float -- the threshold on `mode.statistic`.
    """
    _, background = mode.scores()
    return threshold_at_far(background, mode.livetime_days, far_per_day)
