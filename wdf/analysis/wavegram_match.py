"""Whether the second detector holds a similar wavegram, at an allowed time.

An event is its parameters and its wavegram. What makes two events a
coincidence is not that both detectors were loud, nor that two individual
coefficients share a sign: it is that the second detector holds *a similar
map* at a time the light travel time allows. This module measures that.

The two maps are laid on one absolute time grid and one is slid against the
other through every displacement the coincidence admits. What is reported is
the largest agreement found and the displacement that found it. Both come from
the same pass, so the arrival-time difference of a pair is not a separate
estimate to be reconciled with its agreement --- it is where the agreement is.

Sliding is what makes this a coincidence test rather than a comparison of two
pictures. Two maps of one source agree at one displacement and nowhere else;
two maps of unrelated noise agree at no displacement, and allowing the slide
costs only the trials factor of the displacements tried, which the accidental
background measures like every other cost.

Three properties the measure has by construction, and they are the reasons for
its form:

- **It is taken on the map, not on a pair of tiles.** A coefficient is a
  projection onto one basis function; two detectors may resolve one transient
  onto different bases and with a phase of their own, so the sign of a single
  product carries little and its magnitude carries the detector's amplitude
  rather than the pair's agreement. A map holds the transient's shape across
  the band and across time, which is what survives both.
- **It is normalised.** The agreement is a cosine between the two maps, so it
  is bounded by one whatever either event's loudness, and a pair that is merely
  loud cannot reach it. Loudness is measured elsewhere and belongs in the
  ranking beside this, not inside it.
- **It is taken on magnitudes.** Two detectors can respond to one source with
  opposite sign, and they resolve one transient onto different basis functions
  and at an arrival-time difference finer than a bin, so their coefficients
  disagree cell by cell even where the morphologies are identical. A cosine
  between signed maps then measures that phase difference rather than the
  shape, and taking its magnitude afterwards does not undo it: the
  cancellation is inside the sum, not on the total. The maps are therefore
  compared as magnitudes. The sign is not discarded from the analysis --- the
  coherent energy is a signed sum whose magnitude is taken once at the end ---
  only from the question of whether two maps have the same shape.

Each map may be rendered on its own event's instant, which costs nothing as
long as the displacement between the two anchors is carried alongside and the
lags searched are the ones that displacement leaves admissible. Anchoring
without carrying it is what would make the comparison invariant to the
displacement, and throw away the one quantity the coincidence exists to test:
`correlation_profiles` therefore takes the anchor difference as `offset_s`,
searches the map lags that reach the admitted absolute displacements, and
reports the lag axis of the map, so that the absolute displacement of the pair
is `offset_s` plus the reported lag.

The bin is what limits everything downstream. A bin coarser than the light
travel time cannot resolve the delay at all, whatever the lag axis is; a bin
coarser than a tile sums signed coefficients of one detector against each other
before the two detectors are ever compared, so an oscillating transient
cancels against itself and the bands whose tiles are shortest are the ones that
lose most. Neither is a bin the caller may choose for convenience.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import correlation_lags, fftconvolve


#: The width of a time bin, seconds, as a fraction of the shortest tile the
#: band ladder holds. A bin coarser than the light travel time cannot place a
#: pair on the sky; one much finer costs displacements without resolving
#: anything, since a tile's own duration is what limits the map's resolution.
BIN_PER_TILE = 1.0


def render(cloud, bands, first, last, bin_seconds):
    """One event's wavegram on a stated absolute time grid.

    :param cloud: the event's tiles, as `wdf.analysis.detector_graph.event_tiles`
        returns them: ``(t_lo, t_hi, f_lo, f_hi, energy, amplitude)``, times in
        absolute GPS seconds and amplitudes on the block's noise scale.
    :param bands: the band edges to render on, shape ``(n_bands, 2)``. They are
        a property of the analysis window and of the rate, so the two detectors
        share them and a map of one is comparable with a map of the other.
    :type first: float
    :param first: left edge of the grid, absolute GPS seconds.
    :type last: float
    :param last: right edge of the grid, absolute GPS seconds.
    :type bin_seconds: float
    :param bin_seconds: width of a time bin, seconds.
    :return: numpy.ndarray -- shape ``(n_bands, n_bins)``, the signed
        coefficient density on the tile's complete time support, and zero
        where the event placed none. Tiles wholly outside ``[first, last)``
        are discarded. A tile that intersects that interval but does not
        belong to ``bands`` is an error rather than being silently discarded.
        ``bands`` must be the band scale of one window length.
    :raises ValueError: if the grid is empty or the bin width is not positive.
    """
    if not bin_seconds > 0.0:
        raise ValueError(f"a time bin must be positive, got {bin_seconds!r}")
    bands = np.asarray(bands, dtype=float)
    if bands.ndim != 2 or bands.shape[1] != 2 or not len(bands):
        raise ValueError("bands must be a non-empty array of shape (n_bands, 2)")
    if np.any(np.diff(bands[:, 0]) < 0.0):
        raise ValueError("bands must be ordered by increasing lower edge")
    # Subtracting large absolute GPS values can leave an exact bin boundary a
    # few ulps to either side of the integer it represents. Snap only at that
    # floating-point scale: otherwise `ceil` can invent a column and a tile can
    # occupy two bins when its support is exactly one bin wide.
    time_roundoff = 8.0 * np.spacing(max(abs(float(first)), abs(float(last))))
    n_bins = int(np.ceil(((last - first) - time_roundoff) / bin_seconds))
    if n_bins < 1:
        raise ValueError(
            f"the grid from {first!r} to {last!r} holds no bin of "
            f"{bin_seconds!r} s")
    fields = tuple(np.asarray(field) for field in cloud)
    if len(fields) != 6 or any(field.ndim != 1 for field in fields):
        raise ValueError("a tile cloud must contain six one-dimensional arrays")
    if len({len(field) for field in fields}) != 1:
        raise ValueError("the six tile arrays must have the same length")
    t_lo, t_hi, f_lo, f_hi, _, amplitude = fields
    grid = np.zeros((len(bands), n_bins))
    if not len(amplitude):
        return grid

    duration = t_hi - t_lo
    if np.any(~np.isfinite(duration) | (duration <= 0.0)):
        raise ValueError("every tile must have finite, positive time support")

    # The grid cells are half-open intervals. A tile contributes to every cell
    # its support intersects; the exclusive stop makes a tile ending exactly on
    # a cell edge contribute only to the cells on its left. Dividing by the
    # square root of the duration represents the coefficient as a constant L2
    # density over its support, instead of rewarding a long tile for occupying
    # more cells.
    start = np.floor((t_lo - first + time_roundoff) / bin_seconds).astype(int)
    stop = np.ceil((t_hi - first - time_roundoff) / bin_seconds).astype(int)
    in_time = (stop > 0) & (start < n_bins)

    row = np.searchsorted(bands[:, 0], f_lo, side="right") - 1
    clipped_row = np.clip(row, 0, max(len(bands) - 1, 0))
    in_band = ((row >= 0) & (row < len(bands))
               & np.isclose(bands[clipped_row, 0], f_lo)
               & np.isclose(bands[clipped_row, 1], f_hi))
    missing = in_time & ~in_band
    if missing.any():
        at = int(np.flatnonzero(missing)[0])
        raise ValueError(
            f"tile {at} intersects the time grid but its band "
            f"({f_lo[at]!r}, {f_hi[at]!r}) is absent from bands; bands must "
            "be the scale of one window length")

    inside = in_time & in_band
    if not inside.any():
        return grid

    start = np.clip(start[inside], 0, n_bins)
    stop = np.clip(stop[inside], 0, n_bins)
    weight = amplitude[inside] / np.sqrt(duration[inside])

    # Range additions followed by one cumulative sum render the complete
    # support without visiting every cell of every tile in Python.
    difference = np.zeros((len(bands), n_bins + 1))
    np.add.at(difference, (row[inside], start), weight)
    np.add.at(difference, (row[inside], stop), -weight)
    return np.cumsum(difference[:, :-1], axis=1)


def correlation_profile(left, right, max_shift_s, bin_seconds):
    """Return the per-band cosine profile of the magnitudes, on admitted lags.

    :param left: one event's map, shape ``(n_bands, n_bins)``.
    :param right: the other event's map, with the same shape.
    :param max_shift_s: largest admitted displacement, seconds.
    :param bin_seconds: duration of one time bin, seconds.
    :return: tuple -- ``(profile, lags, norm_left, norm_right)`` where the
        profile has shape ``(n_bands, n_admitted_lags)`` and is non-negative,
        the maps having been compared as magnitudes.
    :raises ValueError: if the maps have different shapes or invalid timing.
    """
    if not np.isfinite(bin_seconds) or bin_seconds <= 0.0:
        raise ValueError(f"a time bin must be positive, got {bin_seconds!r}")
    if not np.isfinite(max_shift_s) or max_shift_s < 0.0:
        raise ValueError(
            f"the maximum shift must be finite and non-negative, got "
            f"{max_shift_s!r}")
    if left.shape != right.shape:
        raise ValueError(
            f"the two maps are {left.shape} and {right.shape}; they must be "
            f"rendered on one grid to be compared")
    # Shape is compared on magnitudes; see the module docstring.
    left, right = np.abs(left), np.abs(right)
    norm_left = np.linalg.norm(left)
    norm_right = np.linalg.norm(right)
    lag = correlation_lags(left.shape[1], right.shape[1], mode="full")
    admitted = np.abs(lag) <= max_shift_s / bin_seconds
    if not admitted.any():
        return (np.zeros((left.shape[0], 0)), lag[admitted] * bin_seconds,
                norm_left, norm_right)
    overlap = fftconvolve(left, right[:, ::-1], mode="full", axes=1)
    denominator = norm_left * norm_right
    if denominator == 0.0:
        denominator = 1.0
    return (overlap[:, admitted] / denominator,
            lag[admitted] * bin_seconds, norm_left, norm_right)



def correlation_profiles(left, right, i, j, max_shift_s, bin_seconds,
                         offset_s=0.0):
    """Profiles of many event pairs, on magnitudes, without pairwise matrices.

    :param left: maps with shape ``(n_events, n_bands, n_bins)``.
    :param right: maps with the same shape.
    :param i: indices into `left`, one per edge.
    :param j: indices into `right`, one per edge.
    :param max_shift_s: admitted absolute shift for each edge, seconds.
    :param bin_seconds: map-bin duration, seconds.
    :param offset_s: the anchor difference ``t_left - t_right`` of each edge,
        seconds, so that a map lag ``L`` places the pair at the absolute
        displacement ``offset_s + L``. Admission is tested on that sum.
    :return: tuple -- ``(profiles, lags, residual)``. `profiles` has shape
        ``(n_edges, n_bands, n_lags)`` and an edge's entries at the lags it
        does not admit, or at which the two maps overlap in nothing, are zero.
        `lags` is one axis for every edge, in seconds, spanning the
        displacements the widest tolerance admits. `residual` is one value per
        edge, the part of its anchor difference finer than a bin: the absolute
        displacement of edge ``e`` at lag ``lags[m]`` is
        ``residual[e] + lags[m]``.
    """
    # Shape is compared on magnitudes; see the module docstring.
    left, right = np.abs(np.asarray(left)), np.abs(np.asarray(right))
    i, j = np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64)
    max_shift_s = np.asarray(max_shift_s, dtype=float)
    offset_s = np.asarray(offset_s, dtype=float)
    if max_shift_s.ndim == 0:
        max_shift_s = np.full(len(i), float(max_shift_s))
    if offset_s.ndim == 0:
        offset_s = np.full(len(i), float(offset_s))
    if left.ndim != 3 or right.shape != left.shape:
        raise ValueError("maps must be equally shaped 3-D arrays")
    if (i.shape != j.shape or i.ndim != 1
            or max_shift_s.shape != i.shape or offset_s.shape != i.shape):
        raise ValueError("pair indices and shifts must be equally shaped vectors")
    if not np.isfinite(bin_seconds) or bin_seconds <= 0.0:
        raise ValueError(f"a time bin must be positive, got {bin_seconds!r}")
    if np.any(~np.isfinite(max_shift_s)) or np.any(max_shift_s < 0.0):
        raise ValueError("every admitted shift must be finite and non-negative")
    if np.any(~np.isfinite(offset_s)):
        raise ValueError("every absolute time offset must be finite")
    n_bins, n_bands = left.shape[2], left.shape[1]
    # A map lag of L places the pair at the absolute displacement
    # ``offset_s + L``, so an axis of +-max_shift_s alone cannot reach zero
    # displacement for a pair whose two anchors are already a tolerance apart.
    # Widening the common axis to reach them would make every pair pay for the
    # lags one distant pair needs, and the work would grow with how far apart
    # anchors happen to be rather than with what is measured. The anchor
    # difference is therefore split: a whole number of bins, applied as a shift
    # of the map, and what the rounding leaves over. Pairs sharing a whole part
    # share their shifts, so the axis searched is the same for all of them ---
    # the displacements the tolerance admits, widened by the half bin the
    # rounding can leave --- whatever the anchors are.
    anchor = np.round(offset_s / bin_seconds).astype(np.int64)
    residual = offset_s - anchor * bin_seconds
    max_lag = int(np.ceil((max_shift_s.max(initial=0.0) + 0.5 * bin_seconds)
                          / bin_seconds))
    span = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
    lags = span.astype(float) * bin_seconds
    profiles = np.zeros((len(i), n_bands, len(lags)), dtype=float)
    if not len(i) or not len(lags):
        return profiles, lags, residual
    # The norm of a map is a property of the event, not of the edge it is used
    # in. Taking it per edge gathers one map per pair, which at a background's
    # pair count is the memory wall of this stage and buys nothing.
    denominator = (np.linalg.norm(left.reshape(len(left), -1), axis=1)[i]
                   * np.linalg.norm(right.reshape(len(right), -1), axis=1)[j])
    denominator[denominator == 0.0] = 1.0
    # Gathered once per group of pairs sharing a shift, and reduced in numpy.
    # `paired_dot` sends its matrices to the device on every call, which is the
    # right trade when one call reduces a whole pair set against them and the
    # wrong one here: a shift changes the slice, so no device copy survives to
    # the next lag, and each reduction touches a handful of pairs of a matrix
    # of gigabytes. Measured on this stage's shape the transfer is three orders
    # of magnitude more than the arithmetic it carries.
    order = np.argsort(anchor, kind="mergesort")
    sorted_anchor = anchor[order]
    breaks = np.flatnonzero(np.r_[True, sorted_anchor[1:] != sorted_anchor[:-1]])
    for group in np.split(order, breaks[1:]):
        whole = int(anchor[group[0]])
        gathered_left = left[i[group]]
        gathered_right = right[j[group]]
        scale = denominator[group][:, None]
        for position, step in enumerate(span):
            shift = int(step) - whole
            if abs(shift) >= n_bins:
                continue
            if shift >= 0:
                cut_left, cut_right = slice(shift, None), slice(None, n_bins - shift)
            else:
                cut_left, cut_right = slice(None, n_bins + shift), slice(-shift, None)
            admitted = (np.abs(residual[group] + lags[position])
                        <= max_shift_s[group])
            if not admitted.any():
                continue
            # Reduced on the whole group and selected afterwards. Selecting
            # first copies both gathered blocks at every lag, and a copy of a
            # map per pair is what this stage cannot afford; the slice is a
            # view, and the arithmetic it carries is a fraction of the copy it
            # would take to narrow it.
            overlap = np.einsum("gbt,gbt->gb",
                                gathered_left[:, :, cut_left],
                                gathered_right[:, :, cut_right])
            profiles[group[admitted], :, position] = (overlap[admitted]
                                                      / scale[admitted])
    return profiles, lags, residual


def flatten_windows(windows):
    """Every event's window in one array, with where each one begins.

    The windows differ in width, so a list of them can only be gathered by
    iterating over it in Python. Laying them end to end turns the gather into
    an indexing operation, which is what a stage asked for one window per pair
    at every displacement of a time slide needs.

    :param windows: one rendered map per event, shape ``(n_bands, w_e)``.
    :return: tuple -- ``(flat, offsets, widths)``. `flat` is
        ``(sum_e w_e, n_bands)``, event `e` occupying the rows
        ``offsets[e] : offsets[e] + widths[e]``, transposed so that one
        window's bins are contiguous.
    """
    n_bands = windows[0].shape[0] if len(windows) else 1
    widths = np.array([w.shape[1] for w in windows], dtype=np.int64)
    flat = (np.concatenate([np.asarray(w, dtype=float).T for w in windows])
            if len(windows) else np.zeros((0, n_bands)))
    offsets = np.zeros(len(widths), dtype=np.int64)
    if len(widths):
        offsets[1:] = np.cumsum(widths)[:-1]
    return flat, offsets, widths


def compare_on_pair_grids(windows, half, bin_seconds, i, j, search, offset_s,
                          carry_lags, flat_windows=None):
    """Compare each pair on a grid its own two events fix, not the catalogue's.

    An event is rendered once, on a window centred on its own instant and no
    wider than the event reaches. What two events need to be compared on is
    then a property of that pair --- the longer of the two, widened by the
    displacements to be searched --- and of nothing else. Rendering every event
    on the widest event in the run makes a transient lasting minutes set the
    grid of every millisecond-long one, which at a bin of a millisecond is not
    merely wasteful: the array does not exist.

    Pairs are handled in groups sharing a grid, so that a grid is built once
    per scale rather than once per pair and no loop runs over pairs. The scales
    are powers of two, which is the ladder the transform already works on.

    :param windows: one rendered map per event, shape ``(n_bands, 2h+1)`` with
        the event's instant in the middle bin; widths may differ.
    :param half: the half-width `h` of each event's window, in bins.
    :type bin_seconds: float
    :param bin_seconds: width of a time bin, seconds.
    :param i: left event index for each pair.
    :param j: right event index for each pair.
    :param search: largest displacement to search for each pair, seconds.
    :param offset_s: the difference of the two events' instants, seconds.
    :param carry_lags: the lag axis the returned profiles are placed on. It is
        common to every pair, so it is what a fixed-size feature can be read
        on; the maximum is taken over each pair's own search and may fall
        outside it.
    :param flat_windows: what `flatten_windows` returned for `windows`, when
        the caller formed it once; it is formed here otherwise. It is what
        keeps the padding from being a loop over events.
    :return: tuple -- ``(profiles, match, displacement, measured)``. `profiles`
        has shape ``(n_pairs, n_bands, len(carry_lags))``. `match` is the
        largest agreement found over the pair's own search, `displacement`
        where it was found in seconds, or not-a-number where the pair was
        compared at no displacement at all, and `measured` says which.
    """
    n_bands = windows[0].shape[0] if len(windows) else 1
    i, j = np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64)
    search = np.asarray(search, dtype=float)
    offset_s = np.asarray(offset_s, dtype=float)
    carry_lags = np.asarray(carry_lags, dtype=float)
    profiles = np.zeros((len(i), n_bands, len(carry_lags)))
    match = np.zeros(len(i))
    displacement = np.full(len(i), np.nan)
    measured = np.zeros(len(i), dtype=bool)
    if not len(i):
        return profiles, match, displacement, measured
    half = np.asarray(half, dtype=np.int64)
    flat, offsets, widths = (flatten_windows(windows) if flat_windows is None
                             else flat_windows)
    reach = np.ceil(search / bin_seconds).astype(np.int64)
    need = np.maximum(np.maximum(half[i], half[j]) + reach, 1)
    scale = np.ceil(np.log2(need)).astype(np.int64)
    for step in np.unique(scale):
        block = np.flatnonzero(scale == step)
        reference = int(2 ** int(step))
        # Only the events this scale's pairs are made of, padded into the one
        # grid they share. Padding the whole catalogue at every scale would
        # bring the longest transient's width back onto every event, which is
        # the thing this exists to avoid; both events of a pair fit by
        # construction, the scale being at least the wider of the two.
        events = np.unique(np.concatenate([i[block], j[block]]))
        grid = np.zeros((len(events), n_bands, 2 * reference + 1))
        # Placed by indexing rather than one event at a time: the destination
        # column of every bin of every window is an arithmetic sequence
        # restarting at each event, so the whole scatter is one assignment.
        lengths = widths[events]
        total = int(lengths.sum())
        if total:
            row_start = np.zeros(len(lengths), dtype=np.int64)
            row_start[1:] = np.cumsum(lengths)[:-1]
            within = np.arange(total) - np.repeat(row_start, lengths)
            rows = np.repeat(np.arange(len(events)), lengths)
            columns = np.repeat(reference - half[events], lengths) + within
            source = np.repeat(offsets[events], lengths) + within
            grid[rows, :, columns] = flat[source]
        left = np.searchsorted(events, i[block])
        right = np.searchsorted(events, j[block])
        found, lags, residual = correlation_profiles(
            grid, grid, left, right, search[block], bin_seconds,
            offset_s=offset_s[block])
        # The maximum is over the displacements this pair was compared at, and
        # a pair compared at none has no displacement to report: the first
        # point of an axis is not a measurement.
        present = np.any(found != 0.0, axis=1)
        somewhere = present.any(axis=1)
        reduced = found.sum(axis=1)
        at = np.argmax(np.where(present, np.abs(reduced), -np.inf), axis=1)
        match[block] = np.where(
            somewhere, np.abs(reduced[np.arange(len(block)), at]), 0.0)
        displacement[block] = np.where(somewhere, residual + lags[at], np.nan)
        measured[block] = somewhere
        # The part of the profile every pair can be read on, whatever grid it
        # was compared on, so that it is a feature of fixed size.
        shared = np.flatnonzero(np.isin(np.round(lags, 12),
                                        np.round(carry_lags, 12)))
        if len(shared):
            into = np.searchsorted(np.round(carry_lags, 12),
                                   np.round(lags[shared], 12))
            piece = profiles[block]
            piece[:, :, into] = found[:, :, shared]
            profiles[block] = piece
    return profiles, match, displacement, measured

def absolute_profiles(clouds, bands, i, j, max_shift_s, bin_seconds,
                      displacement=None):
    """Compute signed wavegram profiles on each pair's absolute time grid.

    The tile supports are rendered before correlation, so an event displacement
    moves its representation as well as its catalogue times. This deliberately
    uses one pair-local map at a time: it avoids a dense event-by-event matrix
    while retaining the absolute timing needed by a time-slide background.

    :param clouds: one absolute tile cloud per event.
    :param bands: shared frequency bands, shape ``(n_bands, 2)``.
    :param i: left event index for each pair.
    :param j: right event index for each pair.
    :param max_shift_s: admitted lag per pair, seconds.
    :param bin_seconds: time-bin width, seconds.
    :param displacement: event displacement from the prepared coordinates.
    :return: tuple -- ``(profiles, lags)`` with shape
        ``(n_pairs, n_bands, n_lags)``. The lag axis is common and entries
        outside each pair's admitted lag are zero.
    """
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    limits = np.asarray(max_shift_s, dtype=float)
    if limits.ndim == 0:
        limits = np.full(len(i), float(limits))
    shifts = (np.zeros(len(clouds)) if displacement is None
              else np.asarray(displacement, dtype=float))
    if i.shape != j.shape or limits.shape != i.shape:
        raise ValueError("pair indices and tolerances must be equally shaped")
    if shifts.shape != (len(clouds),) or not np.all(np.isfinite(shifts)):
        raise ValueError("displacement must contain one finite value per cloud")
    if not len(i):
        return np.zeros((0, len(bands), 0)), np.zeros(0)
    maximum = float(np.max(limits))
    lag_count = int(np.floor(maximum / bin_seconds))
    lags = np.arange(-lag_count, lag_count + 1, dtype=float) * bin_seconds
    profiles = np.zeros((len(i), len(bands), len(lags)), dtype=float)
    for pair, (left_index, right_index) in enumerate(zip(i, j)):
        if clouds[left_index] is None or clouds[right_index] is None:
            continue
        left_cloud = list(np.asarray(field, dtype=float) for field in clouds[left_index])
        right_cloud = list(np.asarray(field, dtype=float) for field in clouds[right_index])
        left_cloud[0] += shifts[left_index]
        left_cloud[1] += shifts[left_index]
        right_cloud[0] += shifts[right_index]
        right_cloud[1] += shifts[right_index]
        if not len(left_cloud[5]) or not len(right_cloud[5]):
            continue
        first = min(left_cloud[0].min(), right_cloud[0].min())
        last = max(left_cloud[1].max(), right_cloud[1].max())
        left_map = render(left_cloud, bands, first, last, bin_seconds)
        right_map = render(right_cloud, bands, first, last, bin_seconds)
        pair_profile, pair_lags, _, _ = correlation_profile(
            left_map, right_map, maximum, bin_seconds)
        common = np.flatnonzero(np.isin(lags, pair_lags))
        source = np.flatnonzero(np.isin(pair_lags, lags[common]))
        common = common[np.abs(lags[common]) <= limits[pair]]
        source = source[np.abs(pair_lags[source]) <= limits[pair]]
        profiles[pair][:, common] = pair_profile[:, source]
    return profiles, lags


def agreement(left, right, max_shift_s, bin_seconds):
    """How alike two wavegrams are, at the displacement that aligns them best.

    The cosine between the two maps is formed at every displacement the
    coincidence admits, and the largest magnitude is reported with the
    displacement it was found at. The maps must be rendered on one grid, by
    :func:`render` with the same `first`, `last`, `bands` and `bin_seconds`,
    or they are not comparable.

    :param left: one event's map, shape ``(n_bands, n_bins)``.
    :param right: the other event's map, same shape.
    :type max_shift_s: float
    :param max_shift_s: the largest displacement to try, seconds, in either
        direction. It is what the coincidence admits: the light travel time
        widened by what each event declares its timing is worth. Every
        displacement tried is a trial, and the accidental background pays for
        it, so this is the tolerance and not a number chosen for convenience.
    :type bin_seconds: float
    :param bin_seconds: width of a time bin, seconds, as the maps were
        rendered with.
    :return: tuple -- ``(agreement, shift_s, polarity)``. The agreement is the
        magnitude of the cosine, in ``[0, 1]``, and is zero where either map is
        empty or every admitted overlap is at round-off level. Lags outside
        ``[-max_shift_s, max_shift_s]`` are discarded. ``shift_s = t_left -
        t_right`` is the displacement applied to the right map; `polarity` is
        the sign of the cosine there, which is physical --- two detectors can
        respond to one source with opposite sign.
    :raises ValueError: if the two maps are not the same shape.
    """
    profile, lags, norm_left, norm_right = correlation_profile(
        left, right, max_shift_s, bin_seconds)
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0, 0.0, 0.0
    if not len(lags):
        return 0.0, 0.0, 0.0
    overlap = profile.sum(axis=0)
    at = int(np.argmax(np.abs(overlap)))
    if abs(overlap[at]) <= 1e-12:
        return 0.0, 0.0, 0.0
    cosine = overlap[at]
    return abs(cosine), lags[at], float(np.sign(cosine))


def pair_agreement(left_cloud, right_cloud, bands, max_shift_s,
                   bin_seconds=None):
    """The agreement of two events, from their tiles.

    Renders both on the grid their two extents span and reports what
    :func:`agreement` finds. The full correlation supplies the zero padding
    needed while sliding, so the rendered grid itself needs no shift padding.

    :param left_cloud: one event's tiles.
    :param right_cloud: the other event's tiles.
    :param bands: the band edges both maps are rendered on, shape
        ``(n_bands, 2)``.
    :type max_shift_s: float
    :param max_shift_s: the largest displacement to try, seconds.
    :param bin_seconds: width of a time bin, seconds, or None to take the
        shortest tile the ladder holds --- one over the upper edge of the
        highest band, scaled by `BIN_PER_TILE`.
    :return: tuple -- ``(agreement, shift_s, polarity)``, as
        :func:`agreement` returns them; ``(0.0, 0.0, 0.0)`` where either event
        has no tile.
    """
    if not len(left_cloud[5]) or not len(right_cloud[5]):
        return 0.0, 0.0, 0.0
    bands = np.asarray(bands, dtype=float)
    if bin_seconds is None:
        bin_seconds = BIN_PER_TILE / float(np.max(bands[:, 1]))
    first = min(left_cloud[0].min(), right_cloud[0].min())
    last = max(left_cloud[1].max(), right_cloud[1].max())
    left = render(left_cloud, bands, first, last, bin_seconds)
    right = render(right_cloud, bands, first, last, bin_seconds)
    return agreement(left, right, max_shift_s, bin_seconds)
 