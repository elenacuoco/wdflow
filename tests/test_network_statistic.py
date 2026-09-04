import numpy as np
import pandas as pd
import pytest

from wdf.analysis.network_statistic import CoherentRanking


def _pairs(n, lag_s, fraction, amplitude, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "network_morphology": np.full(n, amplitude, dtype=float),
        "block_coherent_fraction": np.full(n, fraction, dtype=float),
        "block_coherent_dt": rng.normal(0.0, lag_s, n),
    })


TRAVEL = 0.01


def _fitted(seed=0):
    accidental = _pairs(20_000, 5 * TRAVEL, 0.3, 4.0, seed)
    signal = _pairs(4_000, 0.3 * TRAVEL, 0.8, 9.0, seed + 1)
    return CoherentRanking.fit(accidental, signal, travel_time_s=TRAVEL), \
        accidental, signal


def test_a_pair_that_arrived_together_outranks_one_that_did_not():
    """The timing term is measured, not chosen: under a signal the shared
    tiles arrive together, under an accidental their lag is spread over
    whatever the extents allowed. Two pairs of equal coherent energy are then
    separated by how far apart their shared structure arrived."""
    ranking, _, _ = _fitted()
    together = _pairs(1, 0.0, 0.8, 9.0)
    together["block_coherent_dt"] = 0.0
    apart = together.copy()
    apart["block_coherent_dt"] = 4 * TRAVEL
    assert ranking.score(together)[0] > ranking.score(apart)[0]


def test_at_zero_weight_it_is_the_coherent_energy_and_nothing_else():
    """A stage must reduce to the stage before it: with the timing given no
    weight the ranking is the coherent energy the analysis releases today,
    exactly and not nearly."""
    ranking, _, _ = _fitted()
    ranking.timing_weight = 0.0
    table = _pairs(64, 2 * TRAVEL, 0.5, 7.0)
    assert np.allclose(ranking.score(table),
                       table.network_morphology.to_numpy() ** 2)


def test_a_pair_sharing_no_tile_is_ranked_below_every_pair_that_measured():
    """No shared tile is no coherent energy and no lag. The pair keeps its
    place in the population --- it is not cut --- and is scored at the least
    the prior measured, so nothing that measured something ranks below it."""
    ranking, _, _ = _fitted()
    nothing = _pairs(1, 0.0, 0.0, 0.0)
    nothing["block_coherent_dt"] = 0.0
    measured = _pairs(1, 0.0, 0.4, 3.0)
    measured["block_coherent_dt"] = 3 * TRAVEL
    assert ranking.score(nothing)[0] < ranking.score(measured)[0]
    assert np.isnan(ranking.lag_fraction(nothing)[0])


def test_the_lag_is_expressed_in_light_travel_times():
    ranking, _, _ = _fitted()
    table = _pairs(3, 0.0, 0.5, 5.0)
    table["block_coherent_dt"] = [0.0, TRAVEL, -2 * TRAVEL]
    assert np.allclose(ranking.lag_fraction(table), [0.0, 1.0, -2.0])


def test_a_baseline_must_be_a_positive_time():
    accidental = _pairs(100, TRAVEL, 0.3, 4.0)
    with pytest.raises(ValueError, match="light travel times"):
        CoherentRanking.fit(accidental, accidental, travel_time_s=0.0)


def test_a_population_without_the_columns_is_refused():
    ranking, _, _ = _fitted()
    with pytest.raises(ValueError, match="rebuild the network graph"):
        ranking.score(pd.DataFrame({"network_morphology": [1.0]}))


def test_attaching_writes_one_single_precision_column():
    ranking, _, _ = _fitted()
    table = _pairs(8, TRAVEL, 0.5, 6.0)
    ranking.attach(table)
    assert table["network_coherent_timed"].dtype == np.float32
    assert np.allclose(table["network_coherent_timed"].to_numpy(dtype=float),
                       ranking.score(table), rtol=1e-5)
