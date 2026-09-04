import numpy as np
import pytest

from wdf.analysis.timing_prior import TimingPrior


def test_the_ratio_prefers_a_pair_that_used_little_of_its_tolerance():
    """Under a signal the arrival-time difference is concentrated; under an
    accidental it is flat across whatever window the pair was allowed. The
    ratio of the two therefore rises towards zero difference, and it is that
    rise --- not a chosen penalty --- that ranks one pair above another."""
    rng = np.random.default_rng(0)
    accidental = rng.uniform(-1.0, 1.0, 200_000)
    signal = np.clip(rng.normal(0.0, 0.12, 20_000), -1.0, 1.0)

    prior = TimingPrior.fit(accidental, signal)
    near, far = prior.score([0.0, 0.9])
    assert near > far
    # A flat accidental population and a concentrated signal one: the middle
    # of the range is worth more than a uniform density, the edge less.
    assert near > 0.0 > far


def test_a_prior_needs_both_populations():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="one is empty"):
        TimingPrior.fit(rng.uniform(-1, 1, 100), [])


def test_a_bin_no_signal_reached_is_floored_and_not_infinite():
    """A bin no injection landed in is a bin the measurement did not reach.
    Sending the ratio to minus infinity there would rank a candidate on one
    absent count."""
    rng = np.random.default_rng(2)
    prior = TimingPrior.fit(rng.uniform(-1.0, 1.0, 50_000),
                            np.zeros(500))
    assert np.all(np.isfinite(prior.log_ratio))
    assert np.isfinite(prior.score([-0.95, 0.0, 0.95])).all()


def test_a_difference_that_is_not_a_number_is_worth_nothing():
    """A pair whose instants could not be read carries no evidence either way,
    which is zero in log units and not a penalty."""
    rng = np.random.default_rng(3)
    prior = TimingPrior.fit(rng.uniform(-1, 1, 10_000),
                            rng.normal(0, 0.1, 1_000))
    assert prior.score([np.nan])[0] == 0.0


def test_two_runs_estimate_on_the_same_bins():
    """The fraction of its tolerance a pair consumed is bounded by one either
    side, so the range is fixed rather than taken from the data and two priors
    can be compared bin by bin."""
    rng = np.random.default_rng(4)
    wide = TimingPrior.fit(rng.uniform(-1, 1, 5_000), rng.normal(0, 0.3, 500))
    tight = TimingPrior.fit(rng.uniform(-0.2, 0.2, 5_000),
                            rng.normal(0, 0.02, 500))
    assert np.array_equal(wide.edges, tight.edges)
