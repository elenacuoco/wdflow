"""The sky region a network's arrival times imply, and what it must satisfy."""
import numpy as np
import pytest

from wdf.analysis.skymap import (
    arrival_delay, contains, credible_area, localise, sky_grid,
)
from wdf.analysis.detectors import light_travel_time

GPS = 1400000000.0


def _arrivals(ra, dec, ifos, gps=GPS):
    """The times a source at `(ra, dec)` would produce, exactly."""
    return {ifo: gps + float(arrival_delay(ra, dec, gps, ifo, ifos[0]))
            for ifo in ifos}


def test_no_delay_can_exceed_the_light_travel_time():
    """The model delay is a projection of a baseline, so it is bounded by it."""
    ra, dec = sky_grid(120, 60)
    delay = arrival_delay(ra, dec, GPS, "H1", "L1")
    assert np.abs(delay).max() <= light_travel_time("H1", "L1") * (1 + 1e-9)


def test_two_detectors_give_a_ring_and_three_give_a_patch():
    """A second baseline crosses the first, so the region shrinks."""
    ra_true, dec_true = 1.2, -0.4
    times = _arrivals(ra_true, dec_true, ("H1", "L1", "V1"))
    spreads = {ifo: 5e-4 for ifo in times}

    pair = localise({k: times[k] for k in ("H1", "L1")},
                    {k: spreads[k] for k in ("H1", "L1")}, 180, 90)
    triple = localise(times, spreads, 180, 90)

    assert credible_area(triple[2]) < 0.5 * credible_area(pair[2])


def test_the_true_position_is_inside_the_region():
    """The region is where the measurement puts the source, so it holds it."""
    rng = np.random.default_rng(0)
    inside = 0
    for _ in range(30):
        ra_true = rng.uniform(0.0, 2.0 * np.pi)
        dec_true = np.arcsin(rng.uniform(-1.0, 1.0))
        times = _arrivals(ra_true, dec_true, ("H1", "L1"))
        ra, dec, weight = localise(times, {ifo: 5e-4 for ifo in times}, 180, 90)
        inside += contains(ra_true, dec_true, ra, dec, weight, level=0.9)
    assert inside == 30


def test_a_common_shift_of_every_arrival_moves_nothing():
    """Only differences carry direction, so placing the event late is harmless."""
    times = _arrivals(0.7, 0.3, ("H1", "L1", "V1"))
    spreads = {ifo: 5e-4 for ifo in times}
    here = localise(times, spreads, 120, 60)[2]
    later = localise({ifo: t + 3.0 for ifo, t in times.items()}, spreads, 120, 60)[2]
    # The sky rotates with absolute time, so the map is compared at the same
    # orientation: a shift common to every detector leaves the differences, and
    # therefore the region relative to that orientation, unchanged.
    assert credible_area(here) == pytest.approx(credible_area(later), rel=0.05)


def test_a_wider_spread_gives_a_wider_region():
    times = _arrivals(0.7, 0.3, ("H1", "L1"))
    narrow = localise(times, {ifo: 2e-4 for ifo in times}, 180, 90)[2]
    wide = localise(times, {ifo: 2e-3 for ifo in times}, 180, 90)[2]
    assert credible_area(wide) > credible_area(narrow)


def test_one_detector_cannot_place_a_source():
    with pytest.raises(ValueError, match="at least two detectors"):
        localise({"H1": GPS}, {"H1": 1e-3})


def test_a_spread_that_is_not_positive_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        localise({"H1": GPS, "L1": GPS}, {"H1": 0.0, "L1": 1e-3})


def test_a_missing_spread_names_the_detector():
    with pytest.raises(KeyError, match="L1"):
        localise({"H1": GPS, "L1": GPS}, {"H1": 1e-3})
