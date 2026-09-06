"""Arrival times read off the reconstructions.

A trigger records two instants, the energy centroid and the centre of the
tile carrying its largest coefficient. Both are moments of the tiling: the
centroid follows how much of the transient survived thresholding in that
detector, and the tile centre cannot resolve better than the tile it sits on,
whose length is one over the upper edge of its own band. The reconstruction
carries the waveform at the sample, and two quantities are read off it here.

:func:`envelope_instant` gives one event its own instant: the peak of the
analytic envelope of its stitched reconstruction, sought within a stated width
of the instant the event was ranked on. It is a property of that event and of
nothing else, so a time slide carries it with the event and a difference of two
of them is a difference of two node quantities, measured once per event rather
than once per pair.

:func:`arrival_time_difference` gives a pair its arrival-time difference: the
lag that maximises the cross-correlation of the two events' stitched
reconstructions, over the lags the network's geometry allows, together with the
width the correlation peak declares. It measures on the morphology the two
detectors share, below the tile, and costs one correlation per pair.

Nothing here reaches back to the strain: a stitched series is a function of
the coefficients a trigger already carries (see
:func:`wdf.analysis.reconstruction.stitch`), so the estimators run wherever
the analysis runs and add nothing to the front end.
"""
from __future__ import annotations

import numpy as np

#: Half-width of the lag search, seconds. A bound is required: two events
#: admitted in coincidence overlap within the light travel time plus their
#: spreads, and an unbounded search would let a distant accidental overlap of
#: unrelated structure win the maximum. The value bounds what the estimator
#: can return, not what a signal can do, and it must comfortably exceed the
#: light travel time of the network it is used on.
MAX_LAG_S = 0.25


def arrival_time_difference(first, second, fs, max_lag_s=MAX_LAG_S):
    """The arrival-time difference of two reconstructed series, in seconds.

    Both series are placed on one absolute time grid before correlating, so
    the result is a difference of arrival times and not a sample offset; a
    common error in placing the pair cancels. The lag is read at the vertex of
    the parabola through the largest correlation sample and its two
    neighbours, so the difference is not quantised to the sampling interval,
    which on a two-detector baseline is a large fraction of the light travel
    time. The returned uncertainty is the one the pair itself declares: the
    half-width of the correlation peak above half its maximum, floored at one
    sample. That width is a candidate definition rather than an established
    one; coverage against known injections is what judges it.

    :type first: tuple
    :param first: ``(start, samples)`` of the first series. The start is on
        whatever clock the caller chose, and both series must be on the same
        one; `referred_to_instant` puts them on the one a time slide cannot
        move.
    :type second: tuple
    :param second: ``(start, samples)`` of the second series.
    :type fs: float
    :param fs: sampling frequency of both series, Hz.
    :type max_lag_s: float
    :param max_lag_s: half-width of the lag search, seconds; see `MAX_LAG_S`.
    :return: tuple -- ``(dt, sigma)`` in seconds. ``dt`` is positive when the
        first series arrives after the second.
    :raises ValueError: if either series is empty, if the sampling frequency
        is not positive, or if the two are placed further apart than the lag
        search covers, which no lag can close.
    """
    (first_start, a), (second_start, b) = first, second
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not len(a) or not len(b):
        raise ValueError("both series must carry samples")
    if not fs > 0:
        raise ValueError(f"sampling frequency must be positive, got {fs}")

    # Two supports placed further apart than the search may shift them can
    # never be brought into contact, so there is nothing to measure and the
    # grid below would lay out the whole gap once per pair. This is also where
    # a caller that placed the two series on different clocks --- one
    # displaced by a time slide, one not --- is caught, instead of being
    # answered with whatever lag sits at the edge of the search.
    gap = max(float(first_start) - (float(second_start) + len(b) / fs),
              float(second_start) - (float(first_start) + len(a) / fs), 0.0)
    if gap > max_lag_s:
        raise ValueError(
            f"the two series are placed {gap:.3f} s apart, beyond the "
            f"{max_lag_s:.3f} s the lag search covers, so no lag inside it "
            "brings them into contact; they are not on one clock")

    start = min(float(first_start), float(second_start))
    span = max(float(first_start) + len(a) / fs,
               float(second_start) + len(b) / fs) - start
    n = int(np.ceil(span * fs)) + 1
    on_grid_a, on_grid_b = np.zeros(n), np.zeros(n)
    offset_a = int(round((float(first_start) - start) * fs))
    offset_b = int(round((float(second_start) - start) * fs))
    on_grid_a[offset_a:offset_a + len(a)] = a
    on_grid_b[offset_b:offset_b + len(b)] = b

    # Only the lags the geometry allows are ever read, so only those are
    # formed. The whole correlation costs the product of the two lengths, and
    # an event assembled from many blocks is long: a pair of them would be
    # billions of products to keep a few hundred. Each lag is one inner
    # product over the samples that overlap at it, which is linear in the
    # length, so the cost follows the events and the search window rather than
    # the square of the events.
    reach = min(int(round(max_lag_s * fs)), n - 1)
    lags = np.arange(-reach, reach + 1)
    correlation = np.empty(lags.size)
    for position, lag in enumerate(lags):
        # `np.correlate`'s convention: the value at lag L is the sum over the
        # samples where the first series, advanced by L, meets the second.
        if lag >= 0:
            correlation[position] = np.dot(on_grid_a[lag:], on_grid_b[:n - lag])
        else:
            correlation[position] = np.dot(on_grid_a[:n + lag], on_grid_b[-lag:])
    correlation = np.abs(correlation)

    best = int(np.argmax(correlation))
    above = correlation >= 0.5 * correlation[best]
    low = best
    while low > 0 and above[low - 1]:
        low -= 1
    high = best
    while high < len(correlation) - 1 and above[high + 1]:
        high += 1

    # The correlation is sampled on the grid, so its maximum sample is not its
    # maximum: taking the lag of that sample quantises every arrival-time
    # difference to 1/fs, which on a two-detector baseline is a large fraction
    # of the light travel time and is what the spread of the residuals would
    # then be measuring. The peak of a band-limited correlation is smooth and
    # locally quadratic, so the three samples about the largest one place its
    # vertex. The offset is bounded by half a sample by construction: a larger
    # one would mean a neighbour was the maximum.
    offset = 0.0
    if 0 < best < len(correlation) - 1:
        left, peak, right = correlation[best - 1], correlation[best], correlation[best + 1]
        curvature = left - 2.0 * peak + right
        # Negative curvature is what makes the stationary point a maximum. A
        # flat or upward triple is a plateau or a tie between two lags, where
        # the parabola has no vertex to report and the sample lag stands.
        if curvature < 0.0:
            offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))

    dt = (float(lags[best]) + offset) / fs
    sigma = max((high - low) / 2.0, 1.0) / fs
    return dt, sigma


def referred_to_instant(series, events, instant="gpsPeak"):
    """Each event's waveform, placed relative to its own instant.

    A reconstruction is inverted on absolute time, while the instant an event
    reports is a column of a catalogue that a time slide displaces. The
    difference of the two is where the waveform sits inside its own event,
    which no displacement changes, and it is what `arrival_time_difference`
    has to be given: a pair correlated with one series on absolute time and
    the other's instant displaced differs by the whole displacement, and the
    lag search cannot reach across it.

    Referring the series once, before any slide, is what makes the correction
    a property of the pair of events rather than of the slide that happened to
    form it, so it can be measured once and reused.

    :type series: dict
    :param series: ``{cluster_id: (gps_start, samples)}``, as
        :func:`wdf.analysis.reconstruction.stitch` returns them.
    :type events: pandas.DataFrame
    :param events: the events those reconstructions belong to, carrying
        ``cluster_id`` and the instant column, at their undisplaced times.
    :type instant: str
    :param instant: the column each waveform is referred to.
    :return: dict -- ``{cluster_id: (offset_s, samples)}``, the offset being
        the start of the waveform measured from the event's instant.
    :raises KeyError: if the events do not carry `instant` or ``cluster_id``.
    :raises ValueError: if a reconstruction belongs to no event given.
    """
    at = dict(zip(events["cluster_id"].astype(int),
                  events[instant].astype(float)))
    missing = set(int(label) for label in series) - set(at)
    if missing:
        raise ValueError(
            f"reconstruction for cluster {min(missing)} belongs to no event "
            "given; the offset is measured from the event's own instant")
    return {int(label): (float(start) - at[int(label)], samples)
            for label, (start, samples) in series.items()}


def envelope_instant(series, around, fs, width_s):
    """One event's instant, read on its own reconstruction.

    The instant a trigger reports is the centre of the tile carrying its
    largest coefficient, and in a dyadic transform a tile lasts one over the
    upper edge of its own band --- a few milliseconds high in the band, tens of
    milliseconds low in it. Where the loudest coefficient sits low the tile is
    longer than the light travel time of the network and the difference of two
    such instants carries no geometry. The reconstruction carries the waveform
    at the sample, and the peak of its analytic envelope is where the event's
    amplitude is largest, on the grid the samples are on.

    The search is bounded to `width_s` about the tile instant. An event
    assembled from many blocks spreads its energy over its whole extent, and
    the envelope of a long transient peaks where that energy happened to
    concentrate rather than where the event arrived; restricting the search to
    the block the event was ranked on keeps the instant on the feature the
    search selected it for.

    This is a property of one event and of nothing else, so a time slide
    carries it with the event and a difference of two of them is a difference
    of two node quantities. It costs one pass over the samples of one block,
    once per event, and nothing per pair.

    :type series: tuple
    :param series: ``(gps_start, samples)``, as
        :func:`wdf.analysis.reconstruction.stitch` returns it.
    :type around: float
    :param around: the instant to search about, on the same clock as
        `gps_start`; the tile instant of the block the event is ranked on.
    :type fs: float
    :param fs: sampling frequency of the series, Hz.
    :type width_s: float
    :param width_s: full width of the search about `around`, seconds. The
        length of one analysis block is the value this was measured with; a
        wider one lets a long transient's own energy pull the instant away.
    :return: float -- the instant, on the clock `gps_start` is on, or NaN when
        the series carries nothing to read it from.
    :raises ValueError: if the sampling frequency or the width is not positive.
    """
    if not fs > 0:
        raise ValueError(f"sampling frequency must be positive, got {fs}")
    if not width_s > 0:
        raise ValueError(f"the search width must be positive, got {width_s}")

    start, samples = series
    x = np.asarray(samples, dtype=float)
    if x.size < 2 or not np.any(x):
        return float("nan")

    first = int(np.ceil((float(around) - 0.5 * width_s - float(start)) * fs))
    last = int(np.floor((float(around) + 0.5 * width_s - float(start)) * fs))
    first, last = max(first, 0), min(last, x.size - 1)
    if last < first:
        # The block the event is ranked on lies outside the samples the
        # reconstruction covers, which is not a reading of anything.
        return float("nan")

    # The analytic signal by its definition: the spectrum with the negative
    # frequencies removed and the positive ones doubled. Done here rather than
    # through scipy so that the window actually transformed is the one read.
    piece = x[first:last + 1]
    spectrum = np.fft.fft(piece)
    n = piece.size
    weight = np.zeros(n)
    weight[0] = 1.0
    if n % 2 == 0:
        weight[1:n // 2] = 2.0
        weight[n // 2] = 1.0
    else:
        weight[1:(n + 1) // 2] = 2.0
    envelope = np.abs(np.fft.ifft(spectrum * weight))
    return float(start) + (first + int(np.argmax(envelope))) / fs


def envelope_instants(series, events, fs, width_s, instant="gpsPeak"):
    """Each event's instant, read on its own reconstruction.

    :type series: dict
    :param series: ``{cluster_id: (gps_start, samples)}``.
    :type events: pandas.DataFrame
    :param events: the events those reconstructions belong to, carrying
        ``cluster_id`` and the `instant` column.
    :type fs: float
    :param fs: sampling frequency of the series, Hz.
    :type width_s: float
    :param width_s: full width of the search about each event's `instant`.
    :type instant: str
    :param instant: the column the search is centred on.
    :return: numpy.ndarray -- one instant per row of `events`, NaN where the
        event has no reconstruction to read.
    :raises KeyError: if `events` carries no `instant` or ``cluster_id``.
    """
    labels = events["cluster_id"].astype(int).to_numpy()
    around = events[instant].astype(float).to_numpy()
    out = np.full(len(events), np.nan)
    for row, (label, centre) in enumerate(zip(labels, around)):
        held = series.get(int(label))
        if held is None or not np.isfinite(centre):
            continue
        out[row] = envelope_instant(held, centre, fs, width_s)
    return out
