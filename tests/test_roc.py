import numpy as np

from wdf.analysis.roc import ROCCurve


def test_perfect_separation_gives_auc_near_one():
    positives = np.array([10.0, 12.0, 15.0])
    background = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
    roc = ROCCurve(positives, background)
    assert roc.auc() > 0.95


def test_identical_distributions_give_auc_near_half():
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, 500)
    roc = ROCCurve(scores[:250], scores[250:])
    assert 0.3 < roc.auc() < 0.7


def test_curve_columns_and_range():
    roc = ROCCurve(np.array([5.0, 6.0]), np.array([1.0, 2.0, 3.0]))
    c = roc.curve(n_thresholds=50)
    assert set(["threshold", "tpr", "fpr"]).issubset(c.columns)
    assert (c["tpr"] >= 0).all() and (c["tpr"] <= 1).all()
    assert (c["fpr"] >= 0).all() and (c["fpr"] <= 1).all()
