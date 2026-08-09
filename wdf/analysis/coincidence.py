"""Classical time-window multi-detector coincidence between per-IFO clustered
WDF events (TriggerClusterer.clustered_events output).

No coincidence code of any kind exists in wdf/p4TSA itself -- WDF produces
per-detector triggers only, with no cross-detector timing test built in.
This module is that first real cross-detector test: visually overlaying two
detectors' triggers is not the same as actually testing whether they
coincide within a physically justified window.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from wdf.analysis.detectors import DETECTOR_VERTEX, light_travel_time

# Derived from the detectors' positions rather than tabulated: a baseline is a
# distance, and a table of times has to be extended by hand for every pair.
LIGHT_TRAVEL_TIME_S = {
    frozenset((a, b)): light_travel_time(a, b)
    for i, a in enumerate(DETECTOR_VERTEX) for b in list(DETECTOR_VERTEX)[i + 1:]
}

CANDIDATE_COLUMNS_BASE = [
    "candidate_id", "gps_candidate", "ifos_involved", "dt_s",
    "network_enwdf", "n_ifos", "n_candidate_matches",
]

NETWORK_CANDIDATE_COLUMNS_BASE = [
    "candidate_id", "gps_candidate", "ifos_involved",
    "network_enwdf", "n_ifos", "n_triggers_in_candidate",
]


class CoincidenceFinder:
    """Classical coincidence: matches per-IFO clustered events that fall
    within light-travel-time + timing-jitter of each other.
    """

    def __init__(
        self,
        ifo_pairs: list[tuple[str, str]] | None = None,
        timing_jitter_s: float = 0.01,
        network_stat: str = "quadrature_sum",
    ):
        """
        :type ifo_pairs: list[tuple[str, str]] | None
        :param ifo_pairs: detector pairs to test for `find` (default: every pairwise
            combination of the IFOs present in the `clustered` dict passed to `find`).
        :type timing_jitter_s: float
        :param timing_jitter_s: margin added on top of light-travel-time in
            `coincidence_window` (see that method for why).
        :type network_stat: str
        :param network_stat: how to combine per-detector `EnWDF` into one candidate
            `network_enwdf` -- one of "quadrature_sum", "min", "mean".
        :raises ValueError: if `network_stat` is not one of the three above.
        """
        if network_stat not in ("quadrature_sum", "min", "mean"):
            raise ValueError(f"unknown network_stat {network_stat!r}")
        self.ifo_pairs = ifo_pairs
        self.timing_jitter_s = timing_jitter_s
        self.network_stat = network_stat

    def coincidence_window(self, ifo_a: str, ifo_b: str) -> float:
        """window_s = light_travel_time(ifo_a, ifo_b) + 2 * timing_jitter_s.

        The jitter margin matters because WDF's own trigger peak is not a
        matched-filter timing measurement -- it is the center of the
        analysis window that fired, not a sub-sample-accurate arrival time --
        so light-travel-time alone would be too tight a window.

        :type ifo_a: str
        :param ifo_a: first detector's name (e.g. "H1").
        :type ifo_b: str
        :param ifo_b: second detector's name (e.g. "L1").
        :return: float -- coincidence window in seconds.
        :raises KeyError: if no light-travel-time baseline is known for this
            detector pair (see `LIGHT_TRAVEL_TIME_S`).
        """
        key = frozenset((ifo_a, ifo_b))
        try:
            ltt = LIGHT_TRAVEL_TIME_S[key]
        except KeyError:
            raise KeyError(
                f"no light-travel-time baseline known for {ifo_a}-{ifo_b}; "
                f"add it to LIGHT_TRAVEL_TIME_S"
            )
        return ltt + 2 * self.timing_jitter_s

    def _network_enwdf(self, statistics: list[float]) -> float:
        arr = np.asarray(statistics, dtype=float)
        if self.network_stat == "quadrature_sum":
            return float(np.sqrt(np.sum(arr ** 2)))
        if self.network_stat == "min":
            return float(arr.min())
        return float(arr.mean())

    def find(self, clustered: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Input: {ifo: clustered_events_df} (TriggerClusterer output per
        detector). Output: one row per coincident candidate -- unmatched
        (single-detector-only) clusters are NOT included; they remain
        visible only in the per-IFO clustered_events DataFrames.

        :type clustered: dict[str, pandas.DataFrame]
        :param clustered: {ifo: TriggerClusterer.clustered_events output for that
            detector}.
        :return: pandas.DataFrame -- one row per pairwise coincidence candidate,
            columns per `CANDIDATE_COLUMNS_BASE` plus per-IFO `EnWDF_<ifo>`/
            `freqMean_<ifo>`/`cluster_id_<ifo>` columns for each pair tested; empty
            (with `CANDIDATE_COLUMNS_BASE` columns only) if nothing coincides.
        """
        ifos = list(clustered.keys())
        pairs = self.ifo_pairs or list(combinations(ifos, 2))

        candidates = []
        for ifo_a, ifo_b in pairs:
            df_a, df_b = clustered.get(ifo_a), clustered.get(ifo_b)
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                continue
            window_s = self.coincidence_window(ifo_a, ifo_b)
            for _, row_a in df_a.iterrows():
                dt = (df_b["gpsPeak"] - row_a["gpsPeak"]).to_numpy()
                in_window = np.abs(dt) <= window_s
                n_matches = int(in_window.sum())
                if n_matches == 0:
                    continue
                # greedy nearest-|dt| match; ambiguity surfaced via
                # n_candidate_matches rather than silently resolved
                best_idx = np.argmin(np.abs(np.where(in_window, dt, np.inf)))
                row_b = df_b.iloc[best_idx]
                enwdf_a, enwdf_b = float(row_a["EnWDF"]), float(row_b["EnWDF"])
                network_enwdf = self._network_enwdf([enwdf_a, enwdf_b])
                gps_candidate = float(
                    np.average([row_a["gpsPeak"], row_b["gpsPeak"]],
                               weights=[enwdf_a, enwdf_b])
                )
                candidates.append({
                    "gps_candidate": gps_candidate,
                    "ifos_involved": f"{ifo_a},{ifo_b}",
                    "dt_s": float(row_a["gpsPeak"] - row_b["gpsPeak"]),
                    f"EnWDF_{ifo_a}": enwdf_a,
                    f"EnWDF_{ifo_b}": enwdf_b,
                    f"freqMean_{ifo_a}": float(row_a["freqMean"]),
                    f"freqMean_{ifo_b}": float(row_b["freqMean"]),
                    "network_enwdf": network_enwdf,
                    "n_ifos": 2,
                    "n_candidate_matches": n_matches,
                    f"cluster_id_{ifo_a}": row_a["cluster_id"],
                    f"cluster_id_{ifo_b}": row_b["cluster_id"],
                })

        out = pd.DataFrame(candidates)
        if out.empty:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS_BASE)
        out.insert(0, "candidate_id", range(len(out)))
        return out

    def find_network(self, clustered: dict[str, pd.DataFrame], min_ifos: int = 2) -> pd.DataFrame:
        """General N-detector coincidence: connected components of the graph
        where two per-IFO clustered events are linked if they fall within
        `coincidence_window` of each other, for every IFO pair with a known
        `LIGHT_TRAVEL_TIME_S` baseline. Each connected component becomes one
        candidate, spanning however many distinct IFOs its members touch --
        the same density-chaining principle `TriggerClusterer` already uses
        within a single detector, applied across detectors instead. With
        exactly two detectors this reduces to the same matches `find` finds
        for a genuine coincidence, but scales to three or more IFOs without
        a per-network-size branch. Candidates spanning fewer than `min_ifos`
        distinct detectors are dropped (e.g. a lone unmatched cluster).

        :type clustered: dict[str, pandas.DataFrame]
        :param clustered: {ifo: TriggerClusterer.clustered_events output for that
            detector}.
        :type min_ifos: int
        :param min_ifos: minimum number of distinct detectors a connected
            component must span to be kept as a candidate.
        :return: pandas.DataFrame -- one row per connected-component candidate,
            columns per `NETWORK_CANDIDATE_COLUMNS_BASE` plus per-involved-IFO
            `EnWDF_<ifo>`/`freqMean_<ifo>`/`cluster_id_<ifo>` columns, sorted by
            `gps_candidate`; empty (with `NETWORK_CANDIDATE_COLUMNS_BASE` columns
            only) if nothing qualifies.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        nodes = [
            (ifo, row)
            for ifo, df in clustered.items()
            if df is not None and not df.empty
            for _, row in df.iterrows()
        ]
        n = len(nodes)
        if n == 0:
            return pd.DataFrame(columns=NETWORK_CANDIDATE_COLUMNS_BASE)

        edge_i, edge_j = [], []
        for i in range(n):
            ifo_i, row_i = nodes[i]
            for j in range(i + 1, n):
                ifo_j, row_j = nodes[j]
                if ifo_i == ifo_j:
                    continue
                try:
                    window_s = self.coincidence_window(ifo_i, ifo_j)
                except KeyError:
                    continue
                if abs(row_i["gpsPeak"] - row_j["gpsPeak"]) <= window_s:
                    edge_i.append(i)
                    edge_j.append(j)

        adjacency = csr_matrix(
            (np.ones(len(edge_i)), (edge_i, edge_j)), shape=(n, n),
        )
        n_components, labels = connected_components(adjacency, directed=False)

        candidates = []
        for comp_id in range(n_components):
            members = [nodes[i] for i in range(n) if labels[i] == comp_id]
            involved_ifos = sorted(set(ifo for ifo, _ in members))
            if len(involved_ifos) < min_ifos:
                continue
            statistics = [float(row["EnWDF"]) for _, row in members]
            gps_candidate = float(
                np.average([row["gpsPeak"] for _, row in members], weights=statistics)
            )
            row_out = {
                "gps_candidate": gps_candidate,
                "ifos_involved": ",".join(involved_ifos),
                "network_enwdf": self._network_enwdf(statistics),
                "n_ifos": len(involved_ifos),
                "n_triggers_in_candidate": len(members),
            }
            for ifo in involved_ifos:
                # if two clusters from the same IFO end up chained into one
                # component (e.g. via a third IFO), keep the louder one
                loudest = max(
                    (row for f, row in members if f == ifo),
                    key=lambda row: row["EnWDF"],
                )
                row_out[f"EnWDF_{ifo}"] = float(loudest["EnWDF"])
                row_out[f"freqMean_{ifo}"] = float(loudest["freqMean"])
                row_out[f"cluster_id_{ifo}"] = loudest["cluster_id"]
            candidates.append(row_out)

        out = pd.DataFrame(candidates)
        if out.empty:
            return pd.DataFrame(columns=NETWORK_CANDIDATE_COLUMNS_BASE)
        out = out.sort_values("gps_candidate").reset_index(drop=True)
        out.insert(0, "candidate_id", range(len(out)))
        return out
