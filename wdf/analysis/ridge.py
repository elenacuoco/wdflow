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


def ridge_features(time, log_frequency, energy) -> dict:
    """How much of a track the ridge is, and how it moves.

    :param time: the ridge's times, as `event_ridge` returns them.
    :param log_frequency: its frequencies, in nats of log frequency.
    :param energy: the energy of the tile chosen in each bin.
    :return: dict -- the entries of `RIDGE_FEATURES`:

        `ridge_occupancy`
            fraction of the bins that hold a tile at all. A track is continuous
            in time; a scatter of tiles is not.
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
    out["ridge_occupancy"] = float(here.mean()) if here.size else float("nan")
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
