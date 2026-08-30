"""The arrival-time difference estimator, against constructions with a known
answer: the lag is exact where the shift is exact, the sign follows the
convention stated, absolute placement cancels, and the bound is respected."""
import numpy as np
import pytest

from wdf.analysis.timing import arrival_time_difference


FS = 2048.0


def _pulse(n=400, centre=200, width=6.0):
    t = np.arange(n)
    return np.exp(-0.5 * ((t - centre) / width) ** 2) * np.sin(0.7 * t)


def test_recovers_a_known_shift_with_positive_sign():
    # The first series carries the pulse seven samples later, so it arrives
    # after the second and dt must be positive by the stated convention.
    dt, _ = arrival_time_difference(
        (0.0, np.roll(_pulse(), 7)), (0.0, _pulse()), FS)
    assert dt == pytest.approx(7.0 / FS, abs=0.5 / FS)


def test_absolute_placement_cancels():
    # Moving both series by the same start time changes nothing, and a start
    # offset is equivalent to a sample shift: the difference is what counts.
    shift_s = 5.0 / FS
    dt, _ = arrival_time_difference(
        (1e9 + shift_s, _pulse()), (1e9, _pulse()), FS)
    assert dt == pytest.approx(shift_s, abs=0.5 / FS)


def test_sigma_is_floored_at_one_sample():
    _, sigma = arrival_time_difference((0.0, _pulse()), (0.0, _pulse()), FS)
    assert sigma >= 1.0 / FS


def test_lag_search_is_bounded():
    # A shift beyond the bound cannot be returned: the estimator answers
    # inside its stated window rather than chasing a distant overlap.
    far = int(0.2 * FS)
    dt, _ = arrival_time_difference(
        (0.0, np.roll(_pulse(2048, 1024), far)), (0.0, _pulse(2048, 1024)),
        FS, max_lag_s=0.05)
    assert abs(dt) <= 0.05 + 1.0 / FS


def test_empty_series_is_refused():
    with pytest.raises(ValueError):
        arrival_time_difference((0.0, np.array([])), (0.0, _pulse()), FS)
