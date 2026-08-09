import numpy as np
import pandas as pd
import pytest

from wdf.analysis.baseline import (
    BASELINE_FEATURES,
    GraphCoincidenceFinder,
    LogisticCoincidenceBaseline,
)


def _table(n, seed=0, separable=True):
    rng = np.random.default_rng(seed)
    labels = (rng.uniform(0, 1, n) > 0.6).astype(float)
    shift = labels if separable else 0.0
    return pd.DataFrame({
        "candidate_id": np.arange(n),
        "gps_candidate": 1000.0 + np.sort(rng.uniform(0, 100, n)),
        "dt_s": rng.normal(0, 0.05, n) - 0.0,
        "wavegram_similarity": rng.uniform(0, 1, n) * 0.5 + 0.5 * shift,
        "frequency_overlap": rng.uniform(0, 1, n),
        "time_overlap": rng.uniform(0, 1, n),
        "log_energy_ratio": rng.normal(0, 1, n),
        "network_min_enwdf": rng.uniform(3, 10, n) + 5.0 * shift,
        "network_enwdf": rng.uniform(5, 20, n),
        "wavegram_similarity_aligned": rng.uniform(0, 1, n) * 0.5 + 0.5 * shift,
        "wavegram_lag_bins": rng.integers(-3, 4, n).astype(float),
        "wavegram_overlap": rng.uniform(0, 30, n) + 20.0 * shift,
        "wavegram_overlap_aligned": rng.uniform(0, 30, n) + 20.0 * shift,
        "energy_band_overlap": rng.uniform(0, 1, n),
        "dt_over_tolerance": rng.uniform(0, 1, n),
        "network_correlation": rng.uniform(0, 1, n) * 0.5 + 0.4 * shift,
        "coherent_statistic": rng.uniform(0, 20, n) + 25.0 * shift,
    }), labels


def test_the_baseline_ranks_separable_candidates():
    table, labels = _table(400)
    scored = LogisticCoincidenceBaseline().fit(table, labels).score(table)
    real = scored.loc[labels == 1, "baseline_logit"].mean()
    accidental = scored.loc[labels == 0, "baseline_logit"].mean()
    assert real > accidental


def test_the_baseline_uses_the_edge_s_own_physical_features():
    assert "dt_s" in BASELINE_FEATURES
    assert "frequency_overlap" in BASELINE_FEATURES
    assert "time_overlap" in BASELINE_FEATURES
    assert "log_energy_ratio" in BASELINE_FEATURES


def test_a_missing_feature_is_reported_rather_than_ignored():
    table, labels = _table(50)
    with pytest.raises(KeyError, match="time_overlap"):
        LogisticCoincidenceBaseline().fit(table.drop(columns=["time_overlap"]), labels)


def test_one_class_cannot_be_fitted():
    table, _ = _table(50)
    with pytest.raises(ValueError, match="single class"):
        LogisticCoincidenceBaseline().fit(table, np.ones(len(table)))


def test_scoring_before_fitting_is_refused():
    table, _ = _table(10)
    with pytest.raises(RuntimeError, match="not been fitted"):
        LogisticCoincidenceBaseline().score(table)


def test_an_empty_candidate_table_scores_to_nothing():
    table, labels = _table(80)
    fitted = LogisticCoincidenceBaseline().fit(table, labels)
    assert fitted.score(table.iloc[0:0]).empty


def test_the_finder_returns_nothing_when_a_detector_is_empty():
    class _Scorer:
        def score(self, table):
            return table

    finder = GraphCoincidenceFinder(builder=None, scorer=_Scorer(), coefficients={})
    empty = pd.DataFrame(columns=["gpsPeak"])
    assert finder.find({"H1": empty, "L1": empty}).empty
