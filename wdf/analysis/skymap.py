"""Where on the sky a coincidence puts the source, from its arrival times.

A network of `n` detectors measures `n - 1` independent arrival-time
differences, and each one fixes the angle between the source and the baseline
joining a pair of sites. Two detectors therefore give a ring rather than a
point; a third crosses that ring with another and leaves a pair of patches
reflected through the plane of the network; each further site cuts further.

What the region is worth depends entirely on the uncertainty put into it, so the
uncertainty is not chosen here. Each event declares its own timing spread, and
the spread on a difference is the two combined. A region drawn from anything
else is a picture rather than a measurement.

This is a statement about the pipeline's timing, read on the sky, and the check
it has to pass is coverage: over many injections the true position must fall
inside the region of stated credibility about that often. A region that is too
narrow fails it, and one that is too wide passes it while saying nothing.

The sky is in equatorial coordinates, so the map depends on the time through the
Earth's orientation, and the time is taken from the arrivals themselves.
"""
from __future__ import annotations

import numpy as np

from wdf.analysis.detectors import DETECTOR_VERTEX, SPEED_OF_LIGHT

# Days per sidereal day, and the Greenwich sidereal time at the GPS epoch. The
# rotation is what turns a baseline fixed in the Earth into one fixed on the sky.
#
# The constant is fixed against a reference sidereal time across the whole span
# of GPS dates rather than taken at the epoch alone, so that no offset survives
# the decades between that epoch and an observing run. An error here does not
# look like an error: it rotates every sky map by one angle and leaves every
# internal consistency check passing, while biasing every arrival-time
# prediction the map is built from.
_GPS_EPOCH_GMST_RAD = 1.8272328613543891
_SIDEREAL_RATE = 2.0 * np.pi * 1.0027379093508615 / 86400.0


def greenwich_sidereal_angle(gps) -> np.ndarray:
    """The Earth's rotation angle, from GPS seconds.

    A uniform rotation, which is what a *mean* sidereal time is: nutation and
    the equation of the equinoxes are not modelled. Against a reference
    implementation the residual stays below a milliradian over the whole GPS
    era, which is some tens of microseconds on the longest baseline between
    ground-based detectors.

    :param gps: GPS time, seconds; scalar or array.
    :return: numpy.ndarray -- the angle, radians, wrapped to [0, 2 pi).
    """
    gps = np.asarray(gps, dtype=float)
    return np.mod(_GPS_EPOCH_GMST_RAD + _SIDEREAL_RATE * gps, 2.0 * np.pi)


def source_direction(ra, dec, gps) -> np.ndarray:
    """Unit vectors towards a sky position, in the Earth-fixed frame.

    :param ra: right ascension, radians.
    :param dec: declination, radians.
    :param gps: GPS time the Earth's orientation is taken at, seconds.
    :return: numpy.ndarray -- shape `(..., 3)`, one unit vector per position.
    """
    hour_angle = np.asarray(ra, dtype=float) - greenwich_sidereal_angle(gps)
    dec = np.asarray(dec, dtype=float)
    return np.stack([np.cos(dec) * np.cos(hour_angle),
                     np.cos(dec) * np.sin(hour_angle),
                     np.sin(dec)], axis=-1)


def arrival_delay(ra, dec, gps, ifo_a: str, ifo_b: str) -> np.ndarray:
    """Arrival-time difference a source at `(ra, dec)` would produce.

    Positive when the signal reaches `ifo_a` first, matching the sign of
    `t_a - t_b` measured on the events.

    :param ra: right ascension, radians.
    :param dec: declination, radians.
    :param gps: GPS time, seconds.
    :type ifo_a: str
    :param ifo_a: the detector the difference is measured from.
    :type ifo_b: str
    :param ifo_b: the other detector.
    :return: numpy.ndarray -- the delay, seconds, broadcast over the positions.
    :raises KeyError: if either detector's position is unknown.
    """
    baseline = (np.asarray(DETECTOR_VERTEX[str(ifo_b).upper()], dtype=float)
                - np.asarray(DETECTOR_VERTEX[str(ifo_a).upper()], dtype=float))
    return source_direction(ra, dec, gps) @ baseline / SPEED_OF_LIGHT


def sky_grid(n_ra: int = 360, n_dec: int = 180):
    """A grid over the sky, uniform in right ascension and in sine declination.

    Uniform in the sine so that every cell subtends the same solid angle, which
    is what makes a sum over the grid an integral over the sky.

    :type n_ra: int
    :param n_ra: cells in right ascension.
    :type n_dec: int
    :param n_dec: cells in declination.
    :return: tuple -- `(ra, dec)`, each of shape `(n_dec, n_ra)`, radians.
    """
    ra = np.linspace(0.0, 2.0 * np.pi, int(n_ra), endpoint=False)
    dec = np.arcsin(np.linspace(-1.0, 1.0, int(n_dec)))
    return np.meshgrid(ra, dec)


def localise(arrivals, spreads, n_ra: int = 360, n_dec: int = 180):
    """The sky positions consistent with a set of measured arrival times.

    A network of `n` detectors measures `n - 1` independent arrival-time
    differences, taken here against the detector that recorded the earliest
    time. Every sky position is weighted by how well the differences it would
    produce agree with the measured ones, each on its own scale:

        L(ra, dec) = exp( -1/2 sum_i [ dt_i(ra, dec) - dt_i ]^2 / sigma_i^2 ),

    normalised to unit sum over the grid. With two detectors this is a ring,
    because one number cannot say more than an angle to the baseline; with three
    it is where two rings cross, and the ambiguity left is the reflection through
    the plane of the network.

    The absolute times do not enter: only their differences do, so a common
    error in placing the event moves nothing.

    :type arrivals: dict
    :param arrivals: `{ifo: gps}`, one arrival time per detector, seconds.
    :type spreads: dict
    :param spreads: `{ifo: sigma}`, each detector's timing uncertainty in
        seconds. These must be the spreads the events declare rather than
        values chosen to make the region look small; the uncertainty on a
        difference is the two combined in quadrature.
    :type n_ra: int
    :param n_ra: cells in right ascension.
    :type n_dec: int
    :param n_dec: cells in declination.
    :return: tuple -- `(ra, dec, weight)`, each of shape `(n_dec, n_ra)`; the
        weight sums to one over the grid.
    :raises ValueError: if fewer than two detectors are given, or if any spread
        is not positive.
    :raises KeyError: if a detector's position is unknown.
    """
    times = {str(name).upper(): float(value) for name, value in arrivals.items()}
    sigma = {str(name).upper(): float(value) for name, value in spreads.items()}
    ifos = sorted(times)
    if len(ifos) < 2:
        raise ValueError("a sky position needs at least two detectors, "
                         f"got {ifos}")
    missing = [name for name in ifos if name not in sigma]
    if missing:
        raise KeyError(f"no timing spread given for {missing}")
    for name in ifos:
        if not np.isfinite(sigma[name]) or sigma[name] <= 0.0:
            raise ValueError(f"the timing spread of {name} must be positive, "
                             f"got {sigma[name]}")

    # The earliest arrival is the reference, so every measured difference is
    # positive and the residuals below are of one sign convention throughout.
    reference = min(ifos, key=lambda name: times[name])
    gps = times[reference]

    ra, dec = sky_grid(n_ra, n_dec)
    chi_square = np.zeros(ra.shape)
    for name in ifos:
        if name == reference:
            continue
        measured = times[name] - times[reference]
        expected = arrival_delay(ra, dec, gps, name, reference)
        combined = np.hypot(sigma[name], sigma[reference])
        chi_square += ((expected - measured) / combined) ** 2

    weight = np.exp(-0.5 * chi_square)
    total = weight.sum()
    return ra, dec, weight / total if total > 0 else weight


def credible_area(weight, level: float = 0.9) -> float:
    """Solid angle of the smallest region holding `level` of the weight.

    :param weight: a normalised sky map, as `localise` returns.
    :type level: float
    :param level: the fraction of the weight the region must hold.
    :return: float -- the area in square degrees.
    """
    weight = np.asarray(weight, dtype=float)
    ordered = np.sort(weight.reshape(-1))[::-1]
    inside = np.searchsorted(np.cumsum(ordered), float(level)) + 1
    cell = 4.0 * np.pi / weight.size
    return float(inside * cell * (180.0 / np.pi) ** 2)


def contains(ra_true, dec_true, ra, dec, weight, level: float = 0.9) -> bool:
    """Whether the true position falls in the `level` credible region.

    Over many injections the fraction for which this is true is the check the
    map has to pass: a map whose regions are too narrow fails it, and one whose
    regions are too wide passes it while saying nothing.

    :param ra_true: the source's right ascension, radians.
    :param dec_true: the source's declination, radians.
    :param ra: the grid's right ascension, as `localise` returns.
    :param dec: the grid's declination.
    :param weight: the normalised sky map.
    :type level: float
    :param level: the credible level.
    :return: bool
    """
    weight = np.asarray(weight, dtype=float)
    cell = np.argmin((ra - float(ra_true)) ** 2 + (dec - float(dec_true)) ** 2)
    ordered = np.sort(weight.reshape(-1))[::-1]
    cut = ordered[min(np.searchsorted(np.cumsum(ordered), float(level)),
                      ordered.size - 1)]
    return bool(weight.reshape(-1)[cell] >= cut)
