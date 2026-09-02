"""Making an event's statistic mean the same thing whatever its extent.

The statistic of an event is the norm over the tiles it kept, and for white
noise that norm grows with how many tiles there are: an accidental event of
many tiles is louder than an accidental event of one, by construction and not
because anything is there. Ranking events on the raw statistic therefore ranks
them partly by their size, and a grouping stage that assembles larger events
raises its own threshold by doing so.

The remedy is the one `wdf.analysis.scale` applies to the window length. The
statistic is mapped through the background distribution of events of the same
extent,

    S = -log P(L' >= L | H0, size),

which is exponential with unit rate whatever the size, **for an event the
calibration was not fitted on**, so a large event and a small one are compared
on what each is worth against its own noise. What the grouping then earns is
only the signal it accumulated, which is the quantity it was introduced for.

An event the calibration does contain counts itself, and then the j-th largest
of a bin scores `log((N + 1) / (j + 1))` whatever the data: a ladder fixed by
the bin sizes, carrying no fluctuation, and never reaching the extrapolated
branch that candidates reach. A threshold read on such a sample states a rate
the sample cannot contradict. `out_of_sample_significance` is what a background
is scored with when a rate is to be read off it.

The conditioning is on the number of tiles and not on the number of blocks,
because the tiles are what the statistic sums and their count is what sets its
scale under the null. The block count is also a property of the analysis grid
-- at fixed physical duration it scales with the window length and the overlap
-- so conditioning on it would make an event's significance depend on where the
grid happened to start.

What the conditioning assumes is that a signal is no more likely than noise to
land in a bin: the evidence carried by the size itself is discarded, not
credited. Where a signal population prefers a size, that evidence is available
to a later stage and not to this one.

The extent is binned rather than used exactly, because the background has to
have produced enough events of a size for a tail probability to be measurable
there. The bins are chosen from the background itself so that each holds at
least a stated number of events, rather than from a ladder written here.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _sizes(column, name):
    """Event sizes as whole numbers, refusing anything that is not one.

    A non-finite value cast to `int64` becomes the smallest representable
    integer, which then forms a bin of its own below every real size and shifts
    every other event one bin up without raising anything.

    :param column: the size column.
    :type name: str
    :param name: its name, for the message.
    :return: numpy.ndarray of int64.
    :raises ValueError: if any value is not a finite whole number.
    """
    values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{int((~np.isfinite(values)).sum())} events carry no finite "
            f"{name!r}: an event's size is what its significance is "
            f"conditioned on, and there is no bin for a missing one")
    if not np.all(values == np.floor(values)):
        raise ValueError(f"{name!r} must be a whole number of tiles")
    return values.astype(np.int64)


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
    size_column: str = "n_pixels"
    tail_starts: list = field(default_factory=list)
    tail_scales: list = field(default_factory=list)
    tail_fraction: float = 0.1
    max_size: int = None

    @classmethod
    def fit(cls, background: pd.DataFrame, statistic: str = "EnWDF",
            size_column: str = "n_pixels", min_count: int = 200,
            tail_fraction: float = 0.1) -> "EventCalibration":
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
        :param size_column: the column holding each event's size, in
            tiles. A grid-dependent count, such as the number of blocks,
            makes the significance depend on the analysis grid.
        :type min_count: int
        :param min_count: fewest background events a bin may hold.
        :type tail_fraction: float
        :param tail_fraction: upper fraction of each bin the exponential tail
            is fitted on; at least ten events are used.
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

        sizes = _sizes(background[size_column], size_column)
        values = pd.to_numeric(background[statistic],
                               errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(values)
        if not keep.all():
            # The trials factor is the size of the sample the tail is read
            # from, so a background event whose statistic is missing has to be
            # visible rather than quietly absent from the denominator.
            warnings.warn(
                f"{int((~keep).sum())} of {len(keep)} background events carry "
                f"no finite {statistic!r} and are not calibrated on",
                RuntimeWarning, stacklevel=2)
        sizes, values = sizes[keep], values[keep]

        edges = size_bins(sizes, min_count=min_count)
        index = np.clip(np.searchsorted(edges, sizes, side="right") - 1,
                        0, len(edges) - 1)
        tables = [np.sort(values[index == b]) for b in range(len(edges))]

        # The empirical survival cannot fall below one count in the bin, so on
        # its own it caps the significance at log of the bin's size --- and a
        # threshold beyond the cap silently vetoes the whole extent class,
        # however loud its events. Each bin therefore carries an exponential
        # fitted to its own upper tail, and beyond the anchor the significance
        # continues along the measured slope instead of stopping. The scale is
        # the mean excess over the anchor, the maximum-likelihood estimate for
        # an exponential tail.
        tail_starts, tail_scales = [], []
        for table in tables:
            k = max(int(np.ceil(tail_fraction * table.size)), 10)
            k = min(k, table.size)
            if table.size == 0 or k < 2:
                tail_starts.append(np.nan)
                tail_scales.append(np.nan)
                continue
            anchor = float(table[-k])
            excess = table[-k:] - anchor
            # Conditioned on the k-th order statistic as the threshold, the
            # maximum-likelihood scale of an exponential tail divides the total
            # excess by k - 1: one of the k values is the anchor itself and
            # contributes nothing. Dividing by k understates the scale, which
            # steepens the extrapolation and overstates every significance
            # beyond the measurement.
            scale = float(excess.sum()) / float(k - 1)
            # A tail whose values are all equal measures no slope. Reporting
            # one anyway --- a scale at the smallest positive float --- sends
            # anything above the edge to 1e300 nats. There is nothing to
            # extrapolate along, and the empirical branch stands alone.
            tail_starts.append(anchor)
            tail_scales.append(scale if scale > 0.0 else np.nan)
        return cls(edges=edges, tables=tables, statistic=statistic,
                   size_column=size_column, tail_starts=tail_starts,
                   tail_scales=tail_scales, tail_fraction=tail_fraction,
                   max_size=int(sizes.max()) if sizes.size else None)

    def bin_of(self, sizes) -> np.ndarray:
        """Which calibration bin each extent falls in.

        The last bin is open above: the edges pool sizes upward, so it holds
        every extent at or beyond its own edge and an event longer than any the
        background produced belongs to it by construction. Below the first
        edge there is no such reading --- the background produced nothing that
        small --- and the bin is reported as -1 rather than as the first one.

        :param sizes: event extents, in blocks.
        :return: numpy.ndarray -- one bin index per event, -1 where the extent
            is below everything the background measured.
        """
        sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
        if not len(self.edges):
            return np.zeros(sizes.shape, dtype=np.int64)
        index = np.searchsorted(self.edges, sizes, side="right") - 1
        return np.where(sizes < self.edges[0], -1,
                        np.clip(index, 0, len(self.edges) - 1))

    def significance(self, events: pd.DataFrame) -> np.ndarray:
        """`-log P(L' >= L | H0, size)` for every event, in nats.

        Inside the measured range the tail probability is the plug-in
        estimator `(above + 1)/(N + 1)`, so the calibrated background is
        exponential with unit rate by construction. Beyond the largest value
        the bin measured it continues along the bin's own fitted exponential,

            S = log((N + 1) / 2) + (L - max) / scale,

        continuous at the edge and unbounded above: an event far louder than
        every background event of its extent is far more significant, by the
        slope its own background measured, rather than pinned at the largest
        value the empirical count can express --- a cap that would silently
        veto every extent class whose bin is smaller than the threshold
        demands. The extrapolation is a stated model of the tail, not a
        measurement, and the anchor records where the measurement ends.

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
        sizes = _sizes(events[self.size_column], self.size_column)
        # A size the background never reached is not in the last bin's
        # distribution, however the edges pool: the null scale keeps growing
        # with the size while the measured distribution stops, so the mapping
        # reads a tail that was not measured.
        if self.max_size is not None and np.any(sizes > self.max_size):
            warnings.warn(
                f"{int((sizes > self.max_size).sum())} events are larger than "
                f"any of the {self.max_size} the background produced; their "
                f"significance is read on the largest bin the background "
                f"measured and is an extrapolation in size as well as in "
                f"statistic", RuntimeWarning, stacklevel=2)
        index = self.bin_of(sizes)
        # An extent the background never produced leaves NaN: there is no
        # distribution to read the tail probability from, and the first bin's
        # is not it.
        out = np.full(len(events), np.nan)
        for b, table in enumerate(self.tables):
            # +inf is louder than anything measured and belongs at the top
            # of the mapping; NaN and -inf carry no reading and stay NaN.
            rows = np.flatnonzero((index == b)
                                  & (np.isfinite(values) | (values == np.inf)))
            if rows.size == 0 or table.size == 0:
                continue
            here = values[rows]
            above = table.size - np.searchsorted(table, here, side="left")
            empirical = -np.log((above + 1.0) / (table.size + 1.0))

            # The exponential continues the mapping only where the measurement
            # ends. Inside the sample the plug-in survival is kept untouched,
            # so the calibrated background stays exponential with unit rate by
            # construction; replacing the measured upper decade with a fitted
            # slope was found to bend the bulk on recorded noise, whose tail is
            # heavier than one exponential.
            scale = self.tail_scales[b] if b < len(self.tail_scales) else np.nan
            if np.isfinite(scale) and table.size:
                # Anchored at the plug-in value of the largest measured event,
                # so the mapping is continuous there; the half-count this keeps
                # at the edge is conservative by ln 2 and nothing more.
                edge = float(table[-1])
                base = -np.log(2.0 / (table.size + 1.0))
                beyond = here > edge
                empirical[beyond] = base + (here[beyond] - edge) / scale
            out[rows] = empirical
        return out


def out_of_sample_significance(background: pd.DataFrame, folds: int = 10,
                               **kwargs) -> np.ndarray:
    """Each background event scored by a calibration fitted without it.

    A calibration read on the sample it was fitted on scores the j-th largest
    event of a bin as `log((N + 1) / (j + 1))` whatever the data: the event
    counts itself, so the mapping is a ladder fixed by the bin sizes alone. A
    rate read off such a sample is not a measurement of that sample's tail --
    no member can reach the extrapolated branch, which a candidate reaches --
    and the false-alarm rate it states is lower than a fresh background of the
    same livetime gives.

    The background is cut into contiguous folds. Contiguous in the order given,
    because events close in time are the correlated ones and a fold's events
    must not be calibrated on their own neighbours; cutting at random would
    leave each event's neighbours in the fitting set and restore most of the
    self-scoring it is meant to remove.

    :type background: pandas.DataFrame
    :param background: events from data containing no signal, in the order
        they were produced.
    :type folds: int
    :param folds: how many pieces the background is cut into, at least two.
        Each calibration is fitted on a fraction `1 - 1/folds` of the sample,
        so these scores sit below a full-sample calibration's by about
        `log(folds / (folds - 1))` --- 0.105 nats at ten folds --- and a
        foreground scored on the full background is compared against them with
        that much in its favour.
    :param kwargs: passed to `EventCalibration.fit`.
    :return: numpy.ndarray -- one significance per row, positionally aligned
        with `background`.
    :raises ValueError: if `folds` is below two.
    """
    if int(folds) < 2:
        raise ValueError("a calibration cannot be held out of itself: "
                         "folds must be at least two")
    n = len(background)
    out = np.full(n, np.nan)
    if n == 0:
        return out

    cuts = np.linspace(0, n, int(folds) + 1).astype(int)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi <= lo:
            continue
        rest = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
        fitted = EventCalibration.fit(background.iloc[rest], **kwargs)
        out[lo:hi] = fitted.significance(background.iloc[lo:hi])
    return out


def significance_off_source(background: pd.DataFrame, events: pd.DataFrame,
                            folds: int = 10, time_column: str = "gpsStart",
                            **kwargs) -> np.ndarray:
    """Each event scored by a calibration fitted away from its own time.

    One stretch of data is all there is: the background is whichever part of it
    is taken to hold no signal, and a candidate lives in the same stretch. A
    calibration fitted on the whole of it has therefore seen the candidate's
    own time, and on recorded data a detector's noise at a given hour is not
    the noise of another hour --- one loud minute raises the tail the candidate
    beside it is then judged against.

    The stretch is cut into folds contiguous in time. An event is scored by the
    calibration fitted on the background of every other fold, so nothing is
    ever judged against its own neighbourhood, and the same rule serves a
    candidate and a background event alike.

    :type background: pandas.DataFrame
    :param background: the events taken to hold no signal, carrying
        `time_column`.
    :type events: pandas.DataFrame
    :param events: the events to score, carrying `time_column`.
    :type folds: int
    :param folds: how many stretches of equal duration the span is cut into.
    :type time_column: str
    :param time_column: the column holding each event's time, in GPS seconds.
    :param kwargs: passed to `EventCalibration.fit`.
    :return: numpy.ndarray -- one significance per row of `events`, NaN where
        the fold holds no background to calibrate on.
    :raises ValueError: if `folds` is below two.
    :raises KeyError: if either frame lacks `time_column`.
    """
    if int(folds) < 2:
        raise ValueError("a calibration cannot be held out of itself: "
                         "folds must be at least two")
    for name, frame in (("background", background), ("events", events)):
        if time_column not in frame:
            raise KeyError(f"the {name} events carry no {time_column!r}")

    out = np.full(len(events), np.nan)
    if background.empty or events.empty:
        return out

    when = pd.to_numeric(background[time_column],
                         errors="coerce").to_numpy(dtype=float)
    span = when[np.isfinite(when)]
    if not span.size:
        return out
    # Equal stretches of time, not equal counts of events: a quiet hour and a
    # loud one are the same amount of data, and the loud one must not be
    # allowed to become a fold of its own.
    cuts = np.linspace(span.min(), np.nextafter(span.max(), np.inf),
                       int(folds) + 1)
    of_background = np.clip(np.searchsorted(cuts, when, side="right") - 1,
                            0, int(folds) - 1)
    asked = pd.to_numeric(events[time_column],
                          errors="coerce").to_numpy(dtype=float)
    of_events = np.clip(np.searchsorted(cuts, asked, side="right") - 1,
                        0, int(folds) - 1)

    for fold in range(int(folds)):
        rows = np.flatnonzero(of_events == fold)
        if not rows.size:
            continue
        elsewhere = background.iloc[of_background != fold]
        if elsewhere.empty:
            continue
        out[rows] = EventCalibration.fit(elsewhere, **kwargs).significance(
            events.iloc[rows])
    return out
