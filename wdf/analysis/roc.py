"""ROC curve for a coincidence detection statistic (classical `network_enwdf`
or GNN `gnn_logit`), built from known-event candidate scores (TRUE class)
against time-slide background candidate scores (FALSE class, from
significance.BackgroundEstimator).

CAVEAT: with a single continuous real-data segment there are only a handful
of known positives at most -- both the curve and its AUC carry large
uncertainty at this sample size, and the "positive" events were themselves
selected because other pipelines already confirmed them, so this measures
whether the statistic ranks confirmed events above this segment's own
accidental background, not sensitivity to a representative population.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ROCCurve:
    def __init__(self, positive_scores: np.ndarray, background_scores: np.ndarray):
        self.positive_scores = np.asarray(positive_scores, dtype=float)
        self.background_scores = np.asarray(background_scores, dtype=float)
        if self.positive_scores.size == 0 or self.background_scores.size == 0:
            raise ValueError("ROCCurve needs at least one positive and one background score")

    @classmethod
    def from_results(
        cls,
        known_event_candidates: pd.DataFrame,
        background: pd.DataFrame,
        score_col: str = "network_enwdf",
    ) -> "ROCCurve":
        return cls(
            known_event_candidates[score_col].to_numpy(),
            background[score_col].to_numpy(),
        )

    def curve(self, n_thresholds: int = 200) -> pd.DataFrame:
        """Sweep thresholds spanning both score distributions. TPR = fraction
        of known positives with score >= threshold. FPR = fraction of
        background time-slide candidates with score >= threshold (a
        background "false alarm rate per slide-trial", not a classic
        frame-by-frame FPR -- there is no well-defined total "negative
        frame count" for a trigger-coincidence pipeline the way there is
        for a per-sample classifier).
        """
        lo = min(self.positive_scores.min(), self.background_scores.min())
        hi = max(self.positive_scores.max(), self.background_scores.max())
        thresholds = np.linspace(lo, hi, n_thresholds)
        tpr = np.array([(self.positive_scores >= t).mean() for t in thresholds])
        fpr = np.array([(self.background_scores >= t).mean() for t in thresholds])
        # Both tpr(threshold) and fpr(threshold) are non-increasing in threshold
        # (>=threshold count only shrinks as threshold grows), so sweeping
        # thresholds from high to low already yields tpr/fpr both
        # non-decreasing -- a proper monotonic staircase. (Re-sorting by fpr
        # ascending instead would leave tie-breaking among equal-fpr
        # thresholds to argsort's stability, which can silently pick the
        # *lowest* tpr in a tied-fpr band and undercount the AUC.)
        return pd.DataFrame({
            "threshold": thresholds[::-1],
            "tpr": tpr[::-1],
            "fpr": fpr[::-1],
        })

    def auc(self) -> float:
        """Trapezoidal AUC over curve(). Report alongside an explicit small-N
        caveat -- do not treat this as a calibrated sensitivity estimate."""
        c = self.curve()
        return float(np.trapezoid(c["tpr"], c["fpr"]))

    def plot(self, ax=None, label: str | None = None, savepath: str | None = None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        c = self.curve()
        auc = self.auc()
        lbl = label or "ROC"
        ax.plot(c["fpr"], c["tpr"], label=f"{lbl} (AUC={auc:.3f}, n_pos={self.positive_scores.size})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("FPR (background slide-trials >= threshold)")
        ax.set_ylabel("TPR (known events >= threshold)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=8)
        if savepath:
            ax.figure.savefig(savepath, dpi=150, bbox_inches="tight")
        return ax
