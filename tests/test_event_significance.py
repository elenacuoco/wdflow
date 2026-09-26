"""An event's statistic, made to mean the same thing whatever its extent."""
import numpy as np
import pandas as pd
import pytest

from wdf.analysis.event_significance import EventCalibration, size_bins


def _background(rng, n=20000):
    """Events whose statistic grows with their extent, as noise makes it.

    The stitched norm over `k` blocks of white noise grows like the square root
    of `k`, which is the whole reason the calibration exists.
    """
    size = rng.integers(1, 25, size=n)
    value = np.sqrt(size) * rng.chisquare(df=8, size=n) / 8.0
    return pd.DataFrame(dict(n_pixels=size, EnWDF=value))


def test_bins_hold_enough_background_to_measure_a_tail():
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 40, size=5000)
    edges = size_bins(sizes, min_count=500)
    index = np.clip(np.searchsorted(edges, sizes, side="right") - 1, 0, len(edges) - 1)
    counts = np.bincount(index, minlength=len(edges))
    assert (counts >= 500).all()


def test_the_calibrated_statistic_no_longer_favours_long_events():
    """The point of the calibration, stated as the property it must have."""
    rng = np.random.default_rng(1)
    background = _background(rng)
    other = _background(rng)
    calibration = EventCalibration.fit(background, min_count=500)

    raw = other.groupby("n_pixels").EnWDF.median()
    scored = other.assign(S=calibration.significance(other))
    calibrated = scored.groupby("n_pixels").S.median()

    # Before, the median grows with the extent; after, it does not.
    assert raw.iloc[-1] > 2.0 * raw.iloc[0]
    spread = calibrated.max() - calibrated.min()
    assert spread < 0.5, f"significance still depends on extent: {calibrated}"


def test_the_background_significance_is_exponential_with_unit_rate():
    """Which is what makes a threshold on it a statement about a rate."""
    rng = np.random.default_rng(2)
    calibration = EventCalibration.fit(_background(rng), min_count=500)
    scored = calibration.significance(_background(rng))
    scored = scored[np.isfinite(scored)]

    assert np.mean(scored) == pytest.approx(1.0, abs=0.1)
    assert np.mean(scored > 3.0) == pytest.approx(np.exp(-3.0), abs=0.01)


def test_a_signal_keeps_its_advantage_after_calibration():
    """Calibrating removes the extent, not the excess over the noise."""
    rng = np.random.default_rng(3)
    background = _background(rng)
    calibration = EventCalibration.fit(background, min_count=500)

    loud = pd.DataFrame(dict(n_pixels=[1, 16],
                             EnWDF=[np.sqrt(1) * 6.0, np.sqrt(16) * 6.0]))
    scored = calibration.significance(loud)
    assert np.isfinite(scored).all()
    # The same excess over its own noise, at both extents.
    assert scored[0] == pytest.approx(scored[1], rel=0.25)


def test_an_empty_background_is_refused():
    with pytest.raises(ValueError, match="nothing to measure"):
        EventCalibration.fit(pd.DataFrame(dict(n_pixels=[], EnWDF=[])))


def test_a_missing_column_names_itself():
    with pytest.raises(KeyError, match="n_pixels"):
        EventCalibration.fit(pd.DataFrame(dict(EnWDF=[1.0])))


def test_a_loud_event_is_not_pinned_at_the_bin_ceiling():
    """The empirical survival caps at log of the bin's size; the tail must not.

    A threshold tighter than a bin's ceiling silently vetoes the whole extent
    class, however loud its events --- which is how the calibrated statistic
    once lost every multi-block injection while the raw statistic kept them.
    An event far above everything its bin measured must therefore score far
    above the ceiling, along the slope the bin's own tail fixed.
    """
    rng = np.random.default_rng(7)
    background = pd.DataFrame({
        "EnWDF": np.concatenate([rng.exponential(1.0, 3000) + 5.0,
                                 rng.exponential(2.0, 300) + 7.0]),
        "n_pixels": np.concatenate([np.ones(3000, dtype=int),
                                      np.full(300, 4, dtype=int)]),
    })
    calibration = EventCalibration.fit(background, statistic="EnWDF")

    ceiling = np.log(301.0)
    loud = pd.DataFrame({"EnWDF": [60.0], "n_pixels": [4]})
    significance = float(calibration.significance(loud)[0])
    assert significance > 2.0 * ceiling

    # Continuity at the edge of the measurement, and exactness inside it:
    # within the sample the mapping must stay the plug-in survival, so a
    # calibrated background is exponential with unit rate by construction.
    table = calibration.tables[calibration.bin_of([4])[0]]
    edge = float(table[-1])
    near = pd.DataFrame({"EnWDF": [edge - 1e-9, edge + 1e-9],
                         "n_pixels": [4, 4]})
    below, above = calibration.significance(near)
    assert abs(above - below) < 0.1

    scored = calibration.significance(background)
    inside = np.isfinite(scored)
    assert abs(np.mean(scored[inside] > 3.0) - np.exp(-3)) < 0.01


def test_the_raw_threshold_is_where_the_calibrated_one_is_reached():
    """`statistic_at` inverts the calibration, bin by bin and past its edge."""
    from wdf.analysis.event_significance import EventCalibration

    rng = np.random.default_rng(7)
    calibration = EventCalibration.fit(_background(rng), min_count=500)
    for wanted in (1.0, 3.0, 6.0, 12.0):
        raw = calibration.statistic_at(wanted)
        at = pd.DataFrame(dict(n_pixels=calibration.edges, EnWDF=raw))
        above = at.assign(EnWDF=np.nextafter(raw, np.inf))
        assert (calibration.significance(at) <= wanted + 1e-9).all()
        assert (calibration.significance(above) >= wanted - 1e-9).all()


def test_a_large_accidental_event_needs_a_larger_raw_statistic():
    """The size-dependent threshold rises with the extent, as noise does."""
    from wdf.analysis.event_significance import EventCalibration

    rng = np.random.default_rng(8)
    calibration = EventCalibration.fit(_background(rng), min_count=500)
    raw = calibration.statistic_at(3.0)
    assert raw[-1] > 2.0 * raw[0]


def test_the_matched_threshold_passes_as_much_noise_as_the_reference():
    """exp(-S*) of an exponential background is the reference's share."""
    from wdf.analysis.event_significance import (EventCalibration,
                                                 rate_matched_significance)

    rng = np.random.default_rng(9)
    calibration = EventCalibration.fit(_background(rng), min_count=500)
    fresh = _background(rng)
    threshold = rate_matched_significance(len(fresh), 1000)
    passed = np.count_nonzero(calibration.significance(fresh) >= threshold)
    assert passed == pytest.approx(1000, rel=0.1)


def test_the_matched_threshold_at_its_limits():
    from wdf.analysis.event_significance import rate_matched_significance

    assert rate_matched_significance(100, 100) == 0.0
    assert rate_matched_significance(100, 400) == 0.0
    assert rate_matched_significance(100, 0) == np.inf
    assert rate_matched_significance(1000, 10) == pytest.approx(np.log(100.0))
    with pytest.raises(ValueError):
        rate_matched_significance(0, 1)
