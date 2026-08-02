"""Classical time-window multi-detector coincidence between per-IFO clustered
WDF events (TriggerClusterer.clustered_events output).

No coincidence code of any kind exists in wdf/p4TSA -- TANDEM_4's
`plot_glitchgram` overlays H1/L1 triggers visually but never actually tests
whether they coincide. This module is the first real cross-detector test.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

# Fixed light-travel-time baselines between LIGO/Virgo/KAGRA sites, seconds.
# H1-L1 (~10.0 ms) matches the baseline TANDEM_4's plot_data_wdf_overlay
# already quotes. Extend this table only when a 3rd IFO is actually
# exercised -- not guessed ahead of need.
LIGHT_TRAVEL_TIME_S = {
    frozenset(("H1", "L1")): 0.010,
}

CANDIDATE_COLUMNS_BASE = [
    "candidate_id", "gps_candidate", "ifos_involved", "dt_s",
    "network_snr", "n_ifos", "n_candidate_matches",
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
        if network_stat not in ("quadrature_sum", "min", "mean"):
            raise ValueError(f"unknown network_stat {network_stat!r}")
        self.ifo_pairs = ifo_pairs
        self.timing_jitter_s = timing_jitter_s
        self.network_stat = network_stat

    def coincidence_window(self, ifo_a: str, ifo_b: str) -> float:
        """window_s = light_travel_time(ifo_a, ifo_b) + 2 * timing_jitter_s.

        The jitter margin matters because WDF's own trigger peak is not a
        matched-filter timing measurement (TANDEM_4's plot_data_wdf_overlay
        flags this explicitly) -- light-travel-time alone would be too
        tight a window.
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

    def _network_snr(self, snrs: list[float]) -> float:
        arr = np.asarray(snrs, dtype=float)
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
                dt = (df_b["gpsMax"] - row_a["gpsMax"]).to_numpy()
                in_window = np.abs(dt) <= window_s
                n_matches = int(in_window.sum())
                if n_matches == 0:
                    continue
                # greedy nearest-|dt| match; ambiguity surfaced via
                # n_candidate_matches rather than silently resolved
                best_idx = np.argmin(np.abs(np.where(in_window, dt, np.inf)))
                row_b = df_b.iloc[best_idx]
                snr_a, snr_b = float(row_a["snrMax"]), float(row_b["snrMax"])
                network_snr = self._network_snr([snr_a, snr_b])
                weights = np.array([snr_a, snr_b])
                gps_candidate = float(
                    np.average([row_a["gpsMax"], row_b["gpsMax"]], weights=weights)
                )
                candidates.append({
                    "gps_candidate": gps_candidate,
                    "ifos_involved": f"{ifo_a},{ifo_b}",
                    "dt_s": float(row_a["gpsMax"] - row_b["gpsMax"]),
                    f"snr_{ifo_a}": snr_a,
                    f"snr_{ifo_b}": snr_b,
                    f"freq_{ifo_a}": float(row_a["freqMean"]),
                    f"freq_{ifo_b}": float(row_b["freqMean"]),
                    "network_snr": network_snr,
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
