"""The noise scale read where the transient is not."""
import numpy as np
import pandas as pd
import pytest

from wdf.analysis.scale import local_noise_scale, on_local_scale


def stretch(n=400, sigma=2.0):
    return pd.DataFrame(dict(gps=np.arange(n) * 0.234,
                             sigma=np.full(n, sigma),
                             EnWDF=np.full(n, 10.0)))


def test_a_block_the_transient_inflated_is_read_on_its_neighbours():
    triggers = stretch()
    triggers.loc[200, "sigma"] = 6.0
    assert local_noise_scale(triggers)[200] == pytest.approx(2.0)
    # The statistic divides by the scale, so restoring the block's own and
    # dividing by the local one is exact.
    assert on_local_scale(triggers)[200] == pytest.approx(30.0)


def test_stationary_noise_is_left_exactly_alone():
    triggers = stretch()
    np.testing.assert_allclose(local_noise_scale(triggers),
                               triggers.sigma.to_numpy())
    np.testing.assert_allclose(on_local_scale(triggers),
                               triggers.EnWDF.to_numpy())


def test_a_block_with_no_neighbours_keeps_what_it_measured():
    """Where the stage has nothing to read it does nothing."""
    alone = pd.DataFrame(dict(gps=[0.0], sigma=[3.0], EnWDF=[7.0]))
    assert local_noise_scale(alone)[0] == pytest.approx(3.0)
    assert on_local_scale(alone)[0] == pytest.approx(7.0)


def test_the_median_follows_a_drifting_detector():
    n = 400
    drift = np.linspace(1.0, 3.0, n)
    triggers = pd.DataFrame(dict(gps=np.arange(n) * 0.234, sigma=drift,
                                 EnWDF=np.full(n, 10.0)))
    local = local_noise_scale(triggers, neighbours=41)
    # Away from the ends the median of a linear ramp is the centre value.
    np.testing.assert_allclose(local[100:300], drift[100:300], rtol=1e-12)


def test_rows_out_of_time_order_keep_their_place():
    triggers = stretch(n=101)
    triggers.loc[50, "sigma"] = 9.0
    shuffled = triggers.iloc[np.random.default_rng(0).permutation(101)]
    local = local_noise_scale(shuffled)
    loud = int(np.flatnonzero(shuffled.index.to_numpy() == 50)[0])
    assert local[loud] == pytest.approx(2.0)


def test_a_scale_that_is_not_a_scale_is_not_used():
    triggers = stretch(n=41)
    triggers.loc[:9, "sigma"] = [np.nan] * 5 + [0.0] * 5
    local = local_noise_scale(triggers)
    assert np.all(local[10:] == pytest.approx(2.0))
    # A block whose own estimate failed still sits in noise the neighbours
    # measured, so it takes theirs.
    assert np.all(local[:10] == pytest.approx(2.0))
    # Its statistic, though, was formed by dividing by a scale that does not
    # exist, and no rescaling recovers it.
    assert np.all(~np.isfinite(on_local_scale(triggers)[:10]))


def test_the_columns_it_needs_are_named():
    with pytest.raises(KeyError, match="sigma"):
        local_noise_scale(pd.DataFrame(dict(gps=[0.0])))
    with pytest.raises(ValueError, match="fewer than one"):
        local_noise_scale(stretch(n=3), neighbours=0)
