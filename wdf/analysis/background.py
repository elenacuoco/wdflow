"""Order statistics of a slid population, kept without the population.

A time-slide background is read for two things: the value a rate corresponds
to, and how far into the tail a candidate sits. Both are order statistics, and
an order statistic does not need the sample to be held --- only the part of it
that can still reach the top. A slide's candidates are therefore reduced as
they are formed and released, and what stays is bounded by the number of
values a rate can ask for rather than by the number of slides run.

Two structures do it. The largest `keep` values of every ranking are held
exactly, so a threshold at any rate whose count falls inside them is the same
number the whole population would have given. A histogram over the rest
answers the looser rates, where the count exceeds `keep`, at the resolution of
its bins. Both say which of the two they came from, so a reader is never left
to guess whether a threshold is exact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["BackgroundAccumulator"]


class BackgroundAccumulator:
    """Accumulate a slid population's order statistics slide by slide.

    :type statistics: sequence of str
    :param statistics: the ranking columns to accumulate. A column absent from
        a slide's table contributes nothing rather than raising: a background
        that never carried a ranking is empty for it, which
        :meth:`threshold` reports as `nan`.
    :type keep: int
    :param keep: how many of the largest values of each ranking to hold
        exactly. A threshold whose count is at most this is exact; above it the
        histogram answers. Memory is `keep` floats per ranking, plus the same
        again while a slide is merged.
    :type bins: int
    :param bins: bins of the histogram that covers the values below the kept
        tail.
    :type extras: sequence of str
    :param extras: columns carried alongside the kept tail, for what the tail
        is made of --- the slide it came from, the events it paired. They are
        held only for the rows that are kept.
    """

    def __init__(self, statistics, keep: int = 1_000_000, bins: int = 4096,
                 extras=()):
        self.statistics = [str(s) for s in statistics]
        self.keep = int(keep)
        self.n_bins = int(bins)
        self.extras = [str(e) for e in extras]
        self.n_slides = 0
        self.livetime_s = 0.0
        self.total = 0
        self._kept = {s: np.empty(0, dtype=float) for s in self.statistics}
        self._kept_extras = {s: {e: np.empty(0) for e in self.extras}
                             for s in self.statistics}
        self._edges = {}
        self._counts = {}
        self._below = {s: 0 for s in self.statistics}

    def add(self, candidates, livetime_s: float = 0.0) -> None:
        """Absorb one slide's candidates and let them go.

        :type candidates: pandas.DataFrame
        :param candidates: the accidental candidates of one displacement.
        :type livetime_s: float
        :param livetime_s: the livetime that displacement credits, seconds.
        """
        self.n_slides += 1
        self.livetime_s += float(livetime_s)
        if candidates is None or not len(candidates):
            return
        self.total += len(candidates)
        for statistic in self.statistics:
            if statistic not in candidates:
                continue
            values = np.asarray(candidates[statistic], dtype=float)
            finite = np.isfinite(values)
            values = values[finite]
            if not values.size:
                continue
            extras = {e: np.asarray(candidates[e])[finite]
                      for e in self.extras if e in candidates}
            self._absorb(statistic, values, extras)

    def _absorb(self, statistic, values, extras) -> None:
        """Merge one slide's values into the kept tail and the histogram."""
        kept = np.concatenate([self._kept[statistic], values])
        held = {e: np.concatenate([self._kept_extras[statistic].get(
                    e, np.empty(0, dtype=np.asarray(v).dtype)), v])
                for e, v in extras.items()}
        if kept.size > self.keep:
            # Partition rather than sort: the tail is what is kept and its
            # order does not matter until a threshold is read from it.
            cut = kept.size - self.keep
            order = np.argpartition(kept, cut)
            dropped, keep_index = order[:cut], order[cut:]
            self._histogram(statistic, kept[dropped])
            kept = kept[keep_index]
            held = {e: v[keep_index] for e, v in held.items()}
        self._kept[statistic] = kept
        self._kept_extras[statistic] = held

    def _histogram(self, statistic, values) -> None:
        """Bin the values that fall out of the kept tail."""
        if statistic not in self._edges:
            lo = float(np.min(values))
            hi = float(np.max(values))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0
            self._edges[statistic] = np.linspace(lo, hi, self.n_bins + 1)
            self._counts[statistic] = np.zeros(self.n_bins, dtype=np.int64)
        edges = self._edges[statistic]
        below = int(np.sum(values < edges[0]))
        self._below[statistic] += below
        inside = values[values >= edges[0]]
        if inside.size:
            index = np.clip(np.searchsorted(edges, inside, side="right") - 1,
                            0, self.n_bins - 1)
            np.add.at(self._counts[statistic], index, 1)

    @property
    def livetime_days(self) -> float:
        """The livetime the slides credited, days."""
        return self.livetime_s / 86400.0

    def threshold(self, statistic: str, far_per_day: float,
                  livetime_days: float | None = None):
        """The value a rate corresponds to, and whether it is exact.

        The threshold is the k-th largest accidental with k the whole part of
        the rate times the livetime, which is the same rule
        :func:`wdf.analysis.evaluation.threshold_at_far` applies to a held
        population.

        :type statistic: str
        :param statistic: the ranking to read.
        :type far_per_day: float
        :param far_per_day: the rate, per day.
        :type livetime_days: float or None
        :param livetime_days: the accidental livetime; the accumulated one when
            not given.
        :return: tuple[float, bool] -- the threshold and whether it came from
            the exactly kept tail. `nan` when the livetime cannot resolve the
            rate or the ranking was never seen.
        """
        livetime = (self.livetime_days if livetime_days is None
                    else float(livetime_days))
        k = int(far_per_day * livetime)
        kept = self._kept.get(statistic, np.empty(0))
        if k < 1 or not kept.size:
            return float("nan"), False
        if k <= kept.size:
            return float(np.sort(kept)[-k]), True
        counts = self._counts.get(statistic)
        if counts is None:
            return float("nan"), False
        # The kept tail holds the loudest; the histogram continues below it.
        remaining = k - kept.size
        cumulative = np.cumsum(counts[::-1])
        index = int(np.searchsorted(cumulative, remaining))
        edges = self._edges[statistic]
        if index >= counts.size:
            return float(edges[0]), False
        return float(edges[counts.size - index - 1]), False

    def tail(self, statistic: str) -> pd.DataFrame:
        """The kept tail of one ranking, with whatever was carried beside it.

        :param statistic: the ranking to read.
        :return: pandas.DataFrame -- one row per kept accidental, the ranking
            in a column of its name.
        """
        kept = self._kept.get(statistic, np.empty(0))
        frame = pd.DataFrame({statistic: kept})
        for name, values in self._kept_extras.get(statistic, {}).items():
            frame[name] = values
        return frame.sort_values(statistic, ascending=False,
                                 ignore_index=True)

    def summary(self) -> pd.DataFrame:
        """What was accumulated, one row per ranking."""
        rows = []
        for statistic in self.statistics:
            kept = self._kept.get(statistic, np.empty(0))
            rows.append(dict(statistic=statistic, accidentals=self.total,
                             kept=int(kept.size),
                             binned=int(self._counts.get(
                                 statistic, np.zeros(1)).sum()),
                             below_bins=int(self._below.get(statistic, 0)),
                             slides=self.n_slides,
                             livetime_days=self.livetime_days))
        return pd.DataFrame(rows)
