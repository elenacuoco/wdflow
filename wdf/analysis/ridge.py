"""The track an event leaves across the plane, as numbers.

An event is a set of tiles. Where those tiles line up into a track --- one
frequency per instant, moving smoothly --- the event looks like a transient that
sweeps; where they scatter, it looks like noise that happened to be grouped. The
eye reads that difference off a time-frequency picture immediately, and this
module is what lets a statistic read it too.

Nothing here decides whether a candidate exists. The admissibility of a pair is
geometry and physics --- the light travel time, the overlap of bands and
supports --- and stays where it is. These descriptors rank candidates that rule
has already admitted, which is where a statement about morphology belongs: a
prior about shape in the admission would make the injections a description of
the search instead of a check on it.

For the same reason the descriptors are **symmetric in the direction of the
sweep**. A track that falls scores exactly as one that rises. Preferring rising
frequency would encode a compact binary, and an unmodelled search that does that
has stopped being unmodelled.

Frequency is treated in its logarithm throughout, the coordinate the dyadic
tiling is uniform in, so a slope is octaves per second and means the same thing
at every frequency.
"""
from __future__ import annotations

import numpy as np

EPS = np.finfo(float).tiny

RIDGE_FEATURES = ["ridge_occupancy", "ridge_slope", "ridge_scatter",
                  "ridge_monotonicity", "ridge_continuity"]


def event_ridge(t_lo, t_hi, f_lo, f_hi, energy, n_bins: int = 32):
    """One tile per time bin, the loudest, as a track across the plane.

    Bins the tiles by the centre of their time support and keeps, per bin, the
    tile carrying the most energy. Bins no tile falls in are left `nan` rather
    than interpolated, so a gap in the track stays a gap and the descriptors
    below can see it.

    :param t_lo: start of each tile, seconds.
    :param t_hi: end of each tile, seconds.
    :param f_lo: lower band edge of each tile, Hz.
    :param f_hi: upper band edge of each tile, Hz.
    :param energy: energy carried by each tile, on the noise scale.
    :type n_bins: int
    :param n_bins: how many time bins the event's extent is divided into.
    :return: tuple -- `(time, log_frequency, energy)`, one entry per bin, with
        `nan` where no tile fell.
    """
    t_lo = np.asarray(t_lo, dtype=float).reshape(-1)
    t_hi = np.asarray(t_hi, dtype=float).reshape(-1)
    f_lo = np.asarray(f_lo, dtype=float).reshape(-1)
    f_hi = np.asarray(f_hi, dtype=float).reshape(-1)
    energy = np.asarray(energy, dtype=float).reshape(-1)

    empty = (np.full(int(n_bins), np.nan),) * 3
    if t_lo.size == 0 or not np.isfinite(energy).any():
        return empty

    start, stop = float(t_lo.min()), float(t_hi.max())
    if not (stop > start):
        return empty

    centre = 0.5 * (t_lo + t_hi)
    # The geometric centre of a band, which is its middle in log frequency.
    # The coarsest tile starts at zero frequency, which has no logarithm and no
    # geometric centre: it is represented by half its upper edge, the
    # convention `wavelets.tile_frequency` owns. A floor of EPS instead would
    # place it 500 octaves below the band and drag every moment with it.
    log_f = np.where(f_lo > 0.0,
                     0.5 * (np.log(np.maximum(f_lo, EPS))
                            + np.log(np.maximum(f_hi, EPS))),
                     np.log(np.maximum(0.5 * f_hi, EPS)))

    edges = np.linspace(start, stop, int(n_bins) + 1)
    index = np.clip(np.digitize(centre, edges) - 1, 0, int(n_bins) - 1)

    # The loudest tile of each bin, by a reduction rather than a pass per bin.
    order = np.lexsort((energy, index))
    last = np.ones(order.size, dtype=bool)
    last[:-1] = index[order][1:] != index[order][:-1]
    winner = order[last]

    time = np.full(int(n_bins), np.nan)
    frequency = np.full(int(n_bins), np.nan)
    loudness = np.full(int(n_bins), np.nan)
    time[index[winner]] = centre[winner]
    frequency[index[winner]] = log_f[winner]
    loudness[index[winner]] = energy[winner]
    return time, frequency, loudness



def ridge_track(time, log_frequency, bin_centres=None):
    """The ridge's frequency in every bin, with the gaps filled in.

    `event_ridge` leaves a bin no tile fell in as `nan`, which is the
    measurement. A track, though, is one frequency per instant, and the
    consumers that ask where the event is at a given time need a value in every
    bin. This fills the holes by a straight line in log frequency between the
    occupied bins on either side, and holds the nearest occupied value beyond
    the first and the last of them.

    Linear interpolation between neighbours and a hold outside them is the only
    rule that reads the same forwards and backwards in time: reversing the bins
    reverses the result. It carries no preferred sweep direction, no slope
    prior and no smoothing length, so nothing about the shape of the source
    enters here --- that stays with the descriptors.

    Which bins were measured and which were filled is returned beside the
    track, because a filled bin is not evidence and no statistic may count it
    as one.

    :param time: the ridge's times, as `event_ridge` returns them; unused for
        the interpolation, which runs on `bin_centres`.
    :param log_frequency: the ridge's frequencies, in nats of log frequency,
        `nan` in the bins no tile fell in.
    :param bin_centres: the time each bin stands for, or None to interpolate
        against the bin index. The bins of `event_ridge` are uniform in time,
        so the two give the same track; pass the centres when the track is to
        be read at absolute times.
    :return: tuple -- `(track, measured)`, one entry per bin. `track` is the
        gap-filled log frequency, `nan` only when no bin at all was occupied;
        `measured` is True where a tile was actually found and False where the
        value was filled in.
    """
    log_frequency = np.asarray(log_frequency, dtype=float).reshape(-1)
    n = log_frequency.size
    measured = np.isfinite(log_frequency)
    x = (np.arange(n, dtype=float) if bin_centres is None
         else np.asarray(bin_centres, dtype=float).reshape(-1))
    if not measured.any() or x.size != n:
        return np.full(n, np.nan), measured
    track = np.interp(x, x[measured], log_frequency[measured])
    return track, measured


def ridge_members(t_lo, t_hi, f_lo, f_hi, track, times) -> np.ndarray:
    """The tiles the track passes through.

    A tile is a member when the track's frequency at the centre of its time
    support falls inside that tile's own band. The band is the corridor: there
    is no width to choose and no tolerance to tune, so the selection is fixed
    by the tiling the search already uses and by nothing else.

    The coarsest band starts at zero frequency, which has no logarithm; its
    lower edge is read as half its upper edge, the convention
    `wavelets.tile_frequency` owns and `pixel_graph.cluster_events` already
    applies to the same edge.

    The mask is over the tiles in the order they were given, which is what
    `wdf.analysis.pixel_graph.cluster_events` takes as `labels` once it is cast
    to an integer label per tile.

    :param t_lo: start of each tile, seconds.
    :param t_hi: end of each tile, seconds.
    :param f_lo: lower band edge of each tile, Hz.
    :param f_hi: upper band edge of each tile, Hz.
    :param track: log frequency per bin, as `ridge_track` returns it.
    :param times: the time each bin stands for, seconds.
    :return: numpy.ndarray -- boolean, one entry per tile.
    """
    t_lo = np.asarray(t_lo, dtype=float).reshape(-1)
    t_hi = np.asarray(t_hi, dtype=float).reshape(-1)
    f_lo = np.asarray(f_lo, dtype=float).reshape(-1)
    f_hi = np.asarray(f_hi, dtype=float).reshape(-1)
    track = np.asarray(track, dtype=float).reshape(-1)
    times = np.asarray(times, dtype=float).reshape(-1)

    here = np.isfinite(track) & np.isfinite(times)
    if t_lo.size == 0 or not here.any():
        return np.zeros(t_lo.size, dtype=bool)

    order = np.argsort(times[here], kind="mergesort")
    abscissa, ordinate = times[here][order], track[here][order]
    at = np.interp(0.5 * (t_lo + t_hi), abscissa, ordinate)

    band_lo = np.where(f_lo > 0.0, f_lo, 0.5 * f_hi)
    low = np.log(np.maximum(band_lo, EPS))
    high = np.log(np.maximum(f_hi, EPS))
    return (at >= low) & (at <= high)


def ridge_features(time, log_frequency, energy, measured=None) -> dict:
    """How much of a track the ridge is, and how it moves.

    :param time: the ridge's times, as `event_ridge` returns them.
    :param log_frequency: its frequencies, in nats of log frequency.
    :param energy: the energy of the tile chosen in each bin.
    :param measured: boolean per bin saying which bins hold a tile that was
        actually found, as `ridge_track` returns beside a gap-filled track, or
        None when every finite bin is a measurement. A filled bin carries no
        evidence, so it is never counted in the occupancy; passing a track
        without its mask would report an occupancy of one for any event.
    :return: dict -- the entries of `RIDGE_FEATURES`:

        `ridge_occupancy`
            fraction of the bins that hold a tile at all, counted on the
            measured bins alone. A track is continuous in time; a scatter of
            tiles is not.
        `ridge_slope`
            octaves per second, from an energy-weighted straight-line fit in
            log frequency. Signed, so its magnitude is the sweep rate and its
            sign the direction, which nothing downstream is obliged to prefer.
        `ridge_scatter`
            octaves of residual about that line, energy weighted. Small where
            the tiles lie on a track, large where they do not.
        `ridge_monotonicity`
            how one-directional the steps are, from 0 when they alternate to 1
            when they all go the same way. Computed on the dominant direction,
            so a falling track and a rising one score alike.
        `ridge_continuity`
            median absolute step in octaves between consecutive occupied bins.
            A track moves a little at a time; noise jumps octaves.

        `ridge_occupancy` is always defined --- with no tiles at all it is
        zero, which is a measurement. The rest are `nan` where too few bins
        are occupied to define them.
    """
    time = np.asarray(time, dtype=float).reshape(-1)
    log_frequency = np.asarray(log_frequency, dtype=float).reshape(-1)
    energy = np.asarray(energy, dtype=float).reshape(-1)

    out = {name: float("nan") for name in RIDGE_FEATURES}
    here = np.isfinite(time) & np.isfinite(log_frequency) & np.isfinite(energy)
    occupied = here if measured is None else (
        here & np.asarray(measured, dtype=bool).reshape(-1))
    out["ridge_occupancy"] = (float(occupied.mean()) if occupied.size
                              else float("nan"))
    if here.sum() < 2:
        return out

    t, octave, w = time[here], log_frequency[here] / np.log(2.0), energy[here]
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        w = np.ones_like(t)

    step = np.diff(octave)
    out["ridge_continuity"] = float(np.median(np.abs(step)))
    rising = float((step > 0).sum())
    falling = float((step < 0).sum())
    moved = rising + falling
    # Symmetric by construction: the dominant direction is whichever it is.
    out["ridge_monotonicity"] = (float(max(rising, falling) / moved * 2.0 - 1.0)
                                 if moved else float("nan"))

    if here.sum() >= 3 and np.ptp(t) > 0:
        centre_t = np.average(t, weights=w)
        centre_f = np.average(octave, weights=w)
        variance = np.average((t - centre_t) ** 2, weights=w)
        if variance > 0:
            slope = np.average((t - centre_t) * (octave - centre_f),
                               weights=w) / variance
            residual = octave - (centre_f + slope * (t - centre_t))
            out["ridge_slope"] = float(slope)
            out["ridge_scatter"] = float(
                np.sqrt(np.average(residual ** 2, weights=w)))
    return out


def event_ridge_features(t_lo, t_hi, f_lo, f_hi, energy,
                         n_bins: int = 32) -> dict:
    """The ridge descriptors of one event, from its tiles.

    :param t_lo: start of each tile, seconds.
    :param t_hi: end of each tile, seconds.
    :param f_lo: lower band edge of each tile, Hz.
    :param f_hi: upper band edge of each tile, Hz.
    :param energy: energy carried by each tile, on the noise scale.
    :type n_bins: int
    :param n_bins: how many time bins the event's extent is divided into.
    :return: dict -- the entries of `RIDGE_FEATURES`.
    """
    return ridge_features(*event_ridge(t_lo, t_hi, f_lo, f_hi, energy, n_bins))
