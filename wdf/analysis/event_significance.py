"""Making an event's statistic mean the same thing whatever its extent.

The statistic of an event is measured on the reconstruction stitched across the
blocks it spans, and for white noise the norm of that reconstruction grows with
the number of blocks: an accidental event covering many of them is louder than
an accidental event covering one, by construction and not because anything is
there. Ranking events on the raw statistic therefore ranks them partly by their
extent, and a grouping stage that assembles longer events raises its own
threshold by doing so.

The remedy is the one `wdf.analysis.scale` applies to the window length. The
statistic is mapped through the background distribution of events of the same
extent,

    S = -log P(L' >= L | H0, size),

which is exponential with unit rate whatever the extent, so a long event and a
short one are compared on what each is worth against its own noise. What the
grouping then earns is only the signal it accumulated, which is the quantity it
was introduced for.

The extent is binned rather than used exactly, because the background has to
have produced enough events of a size for a tail probability to be measurable
there. The bins are chosen from the background itself so that each holds at
least a stated number of events, rather than from a ladder written here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def size_bins(sizes, min_count: int = 200) -> np.ndarray:
    """Edges that pool event sizes until each bin is measurable.

    Sizes are pooled upward, so the smallest events --- which a search produces
    in quantity --- keep their own bins and the sparse tail is gathered into
    one. A bin holding a handful of background events cannot state a tail
    probability, and pooling is preferred to reporting one that cannot be
    measured.

    :param sizes: the extent of every background event, in blocks.
    :type min_count: int
    :param min_count: fewest background events a bin may hold.
    :return: numpy.ndarray -- ascending left edges, the first being the
        smallest size present; a size at or above the last edge is in the last
        bin.
    """
    sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
    if sizes.size == 0:
        return np.zeros(0, dtype=np.int64)

    present, counts = np.unique(sizes, return_counts=True)
    edges, running = [int(present[0])], 0
    for value, count in zip(present, counts):
        running += int(count)
        if running >= int(min_count) and value != present[-1]:
            edges.append(int(value) + 1)
            running = 0
    # A trailing bin that never reached the count is merged into the one below.
    if len(edges) > 1 and running < int(min_count):
        edges.pop()
    return np.array(edges, dtype=np.int64)


@dataclass
class EventCalibration:
    """The background distribution of an event statistic, by event extent.

    :param edges: bin edges over the extent, as `size_bins` returns.
    :param tables: sorted background statistics, one array per bin.
    :param statistic: the column the calibration was measured on.
    :param size_column: the column the extent is read from.
    """

    edges: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    tables: list = field(default_factory=list)
    statistic: str = "EnWDF"
    size_column: str = "n_triggers"

    @classmethod
    def fit(cls, background: pd.DataFrame, statistic: str = "EnWDF",
            size_column: str = "n_triggers",
            min_count: int = 200) -> "EventCalibration":
        """Measure the distribution from events built on injection-free data.

        The background must come from the same event-building pipeline as the
        foreground: a distribution measured on differently assembled events
        does not describe these.

        :type background: pandas.DataFrame
        :param background: events from data containing no signal.
        :type statistic: str
        :param statistic: the column to calibrate. Higher must mean more
            signal-like.
        :type size_column: str
        :param size_column: the column holding each event's extent in blocks.
        :type min_count: int
        :param min_count: fewest background events a bin may hold.
        :return: EventCalibration
        :raises ValueError: if the background is empty.
        :raises KeyError: if either column is missing.
        """
        for column in (statistic, size_column):
            if column not in background:
                raise KeyError(f"the background events carry no {column!r}")
        if background.empty:
            raise ValueError(
                "the background is empty: a significance is read from a "
                "measured distribution, and there is nothing to measure")

        sizes = background[size_column].to_numpy(dtype=np.int64)
        values = pd.to_numeric(background[statistic],
                               errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(values)
        sizes, values = sizes[keep], values[keep]

        edges = size_bins(sizes, min_count=min_count)
        index = np.clip(np.searchsorted(edges, sizes, side="right") - 1,
                        0, len(edges) - 1)
        tables = [np.sort(values[index == b]) for b in range(len(edges))]
        return cls(edges=edges, tables=tables, statistic=statistic,
                   size_column=size_column)

    def bin_of(self, sizes) -> np.ndarray:
        """Which calibration bin each extent falls in.

        :param sizes: event extents, in blocks.
        :return: numpy.ndarray -- one bin index per event.
        """
        sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
        if not len(self.edges):
            return np.zeros(sizes.shape, dtype=np.int64)
        return np.clip(np.searchsorted(self.edges, sizes, side="right") - 1,
                       0, len(self.edges) - 1)

    def significance(self, events: pd.DataFrame) -> np.ndarray:
        """`-log P(L' >= L | H0, size)` for every event, in nats.

        The tail probability uses the plug-in estimator `(above + 1)/(N + 1)`,
        which is bounded away from zero: an event louder than every background
        event of its extent is assigned the largest significance the background
        can support rather than an infinite one, and how large that is depends
        on how much background there is.

        :type events: pandas.DataFrame
        :param events: events to score, carrying the calibrated columns.
        :return: numpy.ndarray -- the significance of each event.
        :raises KeyError: if either column is missing.
        """
        for column in (self.statistic, self.size_column):
            if column not in events:
                raise KeyError(f"the events carry no {column!r}")

        values = pd.to_numeric(events[self.statistic],
                               errors="coerce").to_numpy(dtype=float)
        index = self.bin_of(events[self.size_column].to_numpy(dtype=np.int64))
        out = np.full(len(events), np.nan)
        for b, table in enumerate(self.tables):
            rows = np.flatnonzero((index == b) & np.isfinite(values))
            if rows.size == 0 or table.size == 0:
                continue
            above = table.size - np.searchsorted(table, values[rows], side="left")
            out[rows] = -np.log((above + 1.0) / (table.size + 1.0))
        return out
