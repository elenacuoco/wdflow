"""A deterministic ranking of the same candidates the learned one ranks.

Before a learned statistic is adopted it has to beat something honest. The
baseline here is a logistic regression on the candidate edge's own physical
features -- the arrival-time difference, the shared fraction of band and of time
support, the agreement between the two wavegrams, the log ratio of the two
energies and the two single-detector statistics -- fitted on the same edges, and
ranked and calibrated through the same time-slide background.

It requires no torch, which is the point: the comparison is between the graph's
physics and what a learned combiner adds on top of it. If the network does not
beat this at a fixed false alarm rate, the network is not buying anything.

Both this and the learned scorer expose `find`, so
`wdf.analysis.robust_events.TimeSlideFAR` produces a background for either
without knowing which it holds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wdf.analysis.network_graph import EDGE_FEATURES, TriggerGraphBuilder

# The two single-detector statistics belong in the ranking as well: a pair of
# loud events and a pair of quiet ones with the same geometry are not equally
# likely to be astrophysical.
BASELINE_FEATURES = list(EDGE_FEATURES) + ["network_min_enwdf", "network_enwdf"]


class LogisticCoincidenceBaseline:
    """Logistic regression on a candidate edge's physical features.

    :param features: the candidate-table columns to rank on.
    """

    def __init__(self, features: list[str] | None = None):
        self.features = list(features) if features else list(BASELINE_FEATURES)
        self.model = None

    def _design(self, table: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.features if name not in table.columns]
        if missing:
            raise KeyError(f"the candidate table carries no {missing}")
        design = table[self.features].to_numpy(dtype=float)
        return np.where(np.isfinite(design), design, 0.0)

    def fit(self, table: pd.DataFrame, labels) -> "LogisticCoincidenceBaseline":
        """Fit on a scored candidate table.

        :type table: pandas.DataFrame
        :param table: candidate edges, as `TriggerGraph.candidate_table` gives.
        :param labels: 1 where the candidate is a real coincidence, 0 otherwise.
        :return: self
        :raises ValueError: if the labels carry only one class.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        labels = np.asarray(labels, dtype=float)
        if len(np.unique(labels)) < 2:
            raise ValueError(
                "the labels carry a single class, so nothing separates the "
                "candidates and no ranking can be fitted"
            )
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ).fit(self._design(table), labels)
        return self

    def score(self, table: pd.DataFrame) -> pd.DataFrame:
        """Add the baseline's ranking statistic to a candidate table.

        :type table: pandas.DataFrame
        :param table: candidate edges.
        :return: pandas.DataFrame -- `table` with `baseline_logit` and
            `baseline_score`.
        :raises RuntimeError: if called before `fit`.
        """
        if self.model is None:
            raise RuntimeError("the baseline has not been fitted")
        out = table.copy()
        if out.empty:
            out["baseline_logit"] = []
            out["baseline_score"] = []
            return out
        design = self._design(out)
        out["baseline_logit"] = self.model.decision_function(design)
        out["baseline_score"] = self.model.predict_proba(design)[:, 1]
        return out


class GraphCoincidenceFinder:
    """A finder that builds the graph and scores it, for the FAR machinery.

    `wdf.analysis.robust_events.TimeSlideFAR` asks a finder for the candidates
    of each time slide. Wrapping a scorer this way gives the learned and the
    deterministic statistic a time-slide background measured exactly as the
    classical one's is, which is the only calibration of a ranking statistic
    that means anything.

    :param builder: the graph builder.
    :param scorer: anything with `score(table_or_graph)`.
    :param coefficients: ``{ifo: {cluster_id: EventWavegram}}``, the assembly
        map.
    :param on_graph: True when the scorer takes the graph itself (the learned
        scorer), False when it takes the candidate table (the baseline).
    :param comparison: the same events rendered for the cross-detector
        comparison; the assembly map when None.
    """

    def __init__(self, builder: TriggerGraphBuilder, scorer, coefficients,
                 on_graph: bool = False, comparison=None, prepared=None):
        self.builder = builder
        self.scorer = scorer
        self.coefficients = coefficients
        self.on_graph = on_graph
        self.comparison = comparison
        self.prepared = prepared

    def find(self, events_by_ifo: dict) -> pd.DataFrame:
        """Score every physically admissible candidate of these events.

        :type events_by_ifo: dict[str, pandas.DataFrame]
        :param events_by_ifo: each detector's events.
        :return: pandas.DataFrame -- the scored candidate table.
        """
        if any(frame.empty for frame in events_by_ifo.values()):
            return pd.DataFrame()
        if self.prepared is None:
            graph = self.builder.build(
                events_by_ifo,
                {ifo: self.coefficients[ifo] for ifo in events_by_ifo},
                comparison=None if self.comparison is None else
                {ifo: self.comparison[ifo] for ifo in events_by_ifo},
            )
        else:
            graph = self.builder.build_from_prepared(events_by_ifo, self.prepared)
        if not len(graph.cross_edges):
            # The empty table, with its columns. A DataFrame with no rows and no
            # columns reports a missing background as a missing statistic, which
            # sends the reader looking for the wrong fault.
            return graph.candidate_table()
        return (self.scorer.score(graph) if self.on_graph
                else self.scorer.score(graph.candidate_table()))
