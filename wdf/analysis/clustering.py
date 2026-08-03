"""Time-frequency-SNR clustering of raw per-window WDF triggers into
candidate events.

wdf.structures.ClusteredEvent exists upstream but is a plain data holder --
no clustering algorithm populates it anywhere in wdf/p4TSA. A real
astrophysical (or glitch) transient typically registers as a burst of many
overlapping/consecutive raw eventPE triggers (one per analysis window that
crosses WDF's internal wavelet threshold); TriggerClusterer groups those
raw triggers back into single candidate events.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

CLUSTER_COL = "cluster_id"

CLUSTERED_EVENT_COLUMNS = [
    "cluster_id", "is_noise", "trigger_index", "ifo", "gpsStart", "gpsMax", "snrMean", "snrMax",
    "freqMean", "freqMax", "freqMin", "duration", "wave",
    "n_triggers", "gps_span_s",
]


class TriggerClusterer:
    """Groups raw per-window WDF triggers from a SINGLE detector that likely
    belong to the same underlying transient into clustered events.

    Cross-detector matching is CoincidenceFinder's job, not this class's --
    keeping per-IFO clustering and cross-IFO coincidence as separate steps
    mirrors how a burst of raw triggers in one detector says nothing on its
    own about whether a real astrophysical signal is present.
    """

    def __init__(
        self,
        method: str = "dbscan",
        time_eps_s: float = 0.5,
        freq_eps_hz: float = 50.0,
        min_samples: int = 2,
        snr_weight: float = 0.0,
        time_col: str = "gpsPeak",
        freq_col: str = "freqPeak",
        snr_col: str = "snrPeak",
    ):
        """
        :type method: str
        :param method: "dbscan" (sklearn `DBSCAN`) or "greedy" (sklearn-free
            peak-merge baseline/cross-check, see `_greedy_labels`).
        :type time_eps_s: float
        :param time_eps_s: time-axis clustering radius, seconds.
        :type freq_eps_hz: float
        :param freq_eps_hz: frequency-axis clustering radius, Hz.
        :type min_samples: int
        :param min_samples: minimum members for a group to count as a real cluster
            rather than noise/singletons (same semantics as `sklearn.cluster.DBSCAN`'s
            `min_samples`, applied identically in the greedy method too).
        :type snr_weight: float
        :param snr_weight: if > 0, adds `snr_weight * log10(snr_col)` as a third
            clustering feature alongside time/frequency (0 disables it).
        :type time_col: str
        :param time_col: trigger column used as the time-axis clustering feature.
        :type freq_col: str
        :param freq_col: trigger column used as the frequency-axis clustering feature.
        :type snr_col: str
        :param snr_col: trigger column used to rank cluster members (peak selection
            in `clustered_events`) and as the optional SNR clustering feature.
        :raises ValueError: if `method` is not "dbscan" or "greedy".
        """
        if method not in ("dbscan", "greedy"):
            raise ValueError(f"method must be 'dbscan' or 'greedy', got {method!r}")
        self.method = method
        self.time_eps_s = time_eps_s
        self.freq_eps_hz = freq_eps_hz
        self.min_samples = min_samples
        self.snr_weight = snr_weight
        self.time_col = time_col
        self.freq_col = freq_col
        self.snr_col = snr_col

    def fit_predict(self, triggers: pd.DataFrame) -> pd.DataFrame:
        """Input: raw single-IFO trigger DataFrame (>=1 unique `ifo` value;
        raises if more than one is present -- clustering is per-detector).

        Output: the same DataFrame with an added integer `cluster_id` column.
        DBSCAN noise points (`-1`) are kept, not dropped, so isolated
        sub-threshold raw triggers stay visible downstream as singletons.

        :type triggers: pandas.DataFrame
        :param triggers: raw per-window triggers for exactly one detector.
        :return: pandas.DataFrame -- `triggers` (row order/index reset) with an added
            int `cluster_id` column (`-1` = noise/singleton, `>= 0` = a real cluster).
        :raises ValueError: if `triggers` contains more than one distinct `ifo` value.
        """
        if triggers.empty:
            out = triggers.copy()
            out[CLUSTER_COL] = pd.Series(dtype=int)
            return out
        if "ifo" in triggers.columns and triggers["ifo"].nunique() > 1:
            raise ValueError(
                "TriggerClusterer.fit_predict expects a single detector's triggers, "
                f"got ifo values {sorted(triggers['ifo'].unique())}"
            )

        out = triggers.reset_index(drop=True).copy()
        if self.method == "dbscan":
            labels = self._dbscan_labels(out)
        else:
            labels = self._greedy_labels(out)
        out[CLUSTER_COL] = labels
        return out

    def _dbscan_labels(self, df: pd.DataFrame) -> np.ndarray:
        t = df[self.time_col].to_numpy(dtype=float) / self.time_eps_s
        f = df[self.freq_col].to_numpy(dtype=float) / self.freq_eps_hz
        features = [t, f]
        if self.snr_weight > 0:
            snr = np.log10(np.clip(df[self.snr_col].to_numpy(dtype=float), 1e-12, None))
            features.append(self.snr_weight * snr)
        X = np.column_stack(features)
        labels = DBSCAN(eps=1.0, min_samples=self.min_samples).fit_predict(X)
        return labels

    def _greedy_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Greedy peak-merge clustering, generalized to 2-D (time and
        frequency, vs. a 1-D time-only merge): merge a trigger into an
        existing open cluster if it is within [time_eps_s, freq_eps_hz] of
        that cluster's most recent member; sklearn-free baseline/cross-check
        against `_dbscan_labels` above.
        """
        order = np.argsort(df[self.time_col].to_numpy())
        labels = np.full(len(df), -1, dtype=int)
        open_clusters: list[dict] = []  # [{"id", "t", "f", "n"}]
        next_id = 0
        for i in order:
            t = df[self.time_col].iat[i]
            f = df[self.freq_col].iat[i]
            match = None
            for c in open_clusters:
                if abs(t - c["t"]) <= self.time_eps_s and abs(f - c["f"]) <= self.freq_eps_hz:
                    match = c
                    break
            if match is None:
                labels[i] = -1
                open_clusters.append({"id": None, "t": t, "f": f, "members": [i]})
            else:
                match["members"].append(i)
                match["t"], match["f"] = t, f  # advance the reference point

        # Assign real cluster ids only to groups with >= min_samples members;
        # smaller groups fall back to -1 (noise/singleton), matching DBSCAN's
        # min_samples semantics for a fair head-to-head comparison.
        for group in open_clusters:
            if len(group["members"]) >= self.min_samples:
                for i in group["members"]:
                    labels[i] = next_id
                next_id += 1
            else:
                for i in group["members"]:
                    labels[i] = -1
        return labels

    def clustered_events(self, labeled_triggers: pd.DataFrame) -> pd.DataFrame:
        """Input: output of fit_predict (has `cluster_id`).

        Output: one row per cluster -- real clusters (`cluster_id >= 0`) are
        aggregated across their member triggers; DBSCAN/greedy noise points
        (`cluster_id == -1`) are each kept as their own singleton "cluster"
        rather than dropped, so sub-threshold isolated candidates remain
        visible to CoincidenceFinder.

        `cluster_id` here is the same int64 fit_predict uses (`-1` for
        noise, not a string escape) -- `labeled[labeled.cluster_id ==
        row.cluster_id]` recovers a real cluster's exact member triggers by
        matching directly against fit_predict's own output. It can't do that
        for noise rows on its own, since every noise trigger in
        fit_predict's output shares the same `-1`: use `trigger_index`
        (fit_predict output's row position) to recover a specific noise
        row's one source trigger instead, e.g.
        `labeled.iloc[[row.trigger_index]]`. `is_noise` (`cluster_id == -1`)
        is provided as a readability convenience for that branch.
        """
        if labeled_triggers.empty:
            return pd.DataFrame(columns=CLUSTERED_EVENT_COLUMNS)

        df = labeled_triggers.reset_index(drop=True).copy()
        noise_mask = df[CLUSTER_COL] == -1
        # Internal-only grouping key, unique per row for noise triggers (so
        # groupby doesn't collapse every noise point into one row) -- not
        # the same thing as the *output* cluster_id below, which stays the
        # real int64 label (or -1) fit_predict produced.
        group_key = df[CLUSTER_COL].astype(str).to_numpy(dtype=object)
        group_key[noise_mask.to_numpy()] = [f"n{i}" for i in df.index[noise_mask]]
        df["_group"] = group_key
        df["_row_index"] = df.index

        rows = []
        for gid, g in df.groupby("_group", sort=False):
            peak = g.loc[g[self.snr_col].idxmax()]
            n = len(g)
            gps_span_s = float(g[self.time_col].max() - g[self.time_col].min())
            is_noise = bool(g[CLUSTER_COL].iloc[0] == -1)
            rows.append(dict(
                cluster_id=int(g[CLUSTER_COL].iloc[0]),
                is_noise=is_noise,
                trigger_index=int(g["_row_index"].iloc[0]) if is_noise else -1,
                ifo=g["ifo"].iloc[0] if "ifo" in g.columns else None,
                gpsStart=float(g["gps"].min()) if "gps" in g.columns else float(g[self.time_col].min()),
                gpsMax=float(peak[self.time_col]),
                snrMean=float(g["snrMean"].mean()) if "snrMean" in g.columns else float(g[self.snr_col].mean()),
                snrMax=float(peak[self.snr_col]),
                freqMean=float(g["freqMean"].mean()) if "freqMean" in g.columns else float(peak[self.freq_col]),
                freqMax=float(peak["freqMax"]) if "freqMax" in g.columns else float(peak[self.freq_col]),
                freqMin=float(g["freqMin"].min()) if "freqMin" in g.columns else float(peak[self.freq_col]),
                duration=gps_span_s if n > 1 else float(peak.get("duration", 0.0)),
                wave=peak.get("wave"),
                n_triggers=n,
                gps_span_s=gps_span_s,
            ))
        return pd.DataFrame(rows, columns=CLUSTERED_EVENT_COLUMNS)


PIXEL_COLUMNS = ["trigger_index", "ifo", "t_lo", "t_hi", "f_lo", "f_hi", "energy"]
PIXEL_CLUSTERED_EVENT_COLUMNS = [
    "cluster_id", "gpsStart", "gpsEnd", "freqMin", "freqMax",
    "total_energy", "n_pixels", "n_triggers", "ifos",
]


def collect_significant_pixels(triggers: pd.DataFrame, fs: float) -> pd.DataFrame:
    """Time-frequency tiles of the non-zero wavelet coefficients, in absolute GPS.

    Each trigger's `wt*` coefficients are mapped to their tiles via
    `wavelet_coeff_tiles` and shifted to absolute time by the trigger's `gps`.
    `wt*` is expected in the form `WaveletThreshold` emits, where coefficients
    that did not pass thresholding are exactly 0, so keeping the non-zero ones
    selects exactly the tiles that survived it.

    :type triggers: pandas.DataFrame
    :param triggers: triggers carrying `gps` and `wt0..wtN` columns
        (`fullPrint >= 1`); triggers without `wt*` columns are skipped.
    :type fs: float
    :param fs: sampling rate the coefficients were computed at, Hz.
    :return: pandas.DataFrame -- one row per tile with `PIXEL_COLUMNS`:
        `trigger_index`, `ifo`, `t_lo`, `t_hi`, `f_lo`, `f_hi`, `energy`.
    """
    from wdf.analysis.wavelets import wavelet_coeff_tiles

    rows = []
    for idx, trig in triggers.iterrows():
        wt_cols = sorted((c for c in trig.index if c.startswith("wt") and c[2:].isdigit()),
                          key=lambda c: int(c[2:]))
        if not wt_cols:
            continue
        wt = trig[wt_cols].to_numpy(dtype=float)
        tiles = wavelet_coeff_tiles(wt, fs)
        gps0 = trig["gps"]
        for t_lo, t_hi, f_lo, f_hi, mag in tiles:
            if mag == 0.0:
                continue
            rows.append(dict(
                trigger_index=idx, ifo=trig.get("ifo"),
                t_lo=gps0 + t_lo, t_hi=gps0 + t_hi,
                f_lo=f_lo, f_hi=f_hi,
                energy=mag ** 2,
            ))
    return pd.DataFrame(rows, columns=PIXEL_COLUMNS)


class WaveletPixelClusterer:
    """Connected-component clustering directly on WDF's own wavelet-coefficient
    time-frequency tiles, instead of on each trigger's single (gpsPeak,
    freqPeak) point summary -- the same idea coherent WaveBurst (cWB) uses
    (percolation over connected pixels in a time-frequency map). WDF already
    computes the full per-trigger tile map (`wt*` columns, `fullPrint>=1`);
    `TriggerClusterer` never sees it, only the scalar per-window stats
    (`freqPeak` etc.) that get written to the trigger CSV.

    Two pixels are adjacent (and so chained into the same cluster, possibly
    from different, overlapping WDF windows) if their frequency bands touch
    or overlap, and their time spans are within `time_tol_s` of each other.
    """

    def __init__(self, time_tol_s: float):
        self.time_tol_s = time_tol_s

    def fit(self, triggers: pd.DataFrame, fs: float) -> pd.DataFrame:
        """Returns `collect_significant_pixels`'s output with an added
        `cluster_id` column (connected-component label per pixel)."""
        pixels = collect_significant_pixels(triggers, fs)
        pixels = pixels.reset_index(drop=True)
        labels = self._connected_components(pixels) if len(pixels) else np.array([], dtype=int)
        pixels["cluster_id"] = labels
        return pixels

    def _connected_components(self, pixels: pd.DataFrame) -> np.ndarray:
        t_lo = pixels["t_lo"].to_numpy()
        t_hi = pixels["t_hi"].to_numpy()
        f_lo = pixels["f_lo"].to_numpy()
        f_hi = pixels["f_hi"].to_numpy()

        # vectorized pairwise adjacency (O(n^2) memory/time -- fine for the
        # few-thousand-pixel scale of a near-target investigative window;
        # revisit with a spatial index if this needs to scale to a full
        # segment's worth of pixels at once).
        time_adj = (t_lo[:, None] <= t_hi[None, :] + self.time_tol_s) & \
                   (t_lo[None, :] <= t_hi[:, None] + self.time_tol_s)
        freq_adj = (f_lo[:, None] <= f_hi[None, :]) & (f_lo[None, :] <= f_hi[:, None])
        adjacency = time_adj & freq_adj
        np.fill_diagonal(adjacency, False)

        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        _, labels = connected_components(csr_matrix(adjacency), directed=False)
        return labels

    def clustered_events(self, pixels: pd.DataFrame) -> pd.DataFrame:
        """One row per connected component: total energy (sum of |coefficient|^2
        over every member pixel -- an actual integrated energy statistic,
        unlike a single window's `EnWDF`/`snrPeak`), and the cluster's real
        time/frequency span (not one window's worth)."""
        if pixels.empty:
            return pd.DataFrame(columns=PIXEL_CLUSTERED_EVENT_COLUMNS)
        rows = []
        for cid, g in pixels.groupby("cluster_id"):
            rows.append(dict(
                cluster_id=int(cid),
                gpsStart=float(g["t_lo"].min()),
                gpsEnd=float(g["t_hi"].max()),
                freqMin=float(g["f_lo"].min()),
                freqMax=float(g["f_hi"].max()),
                total_energy=float(g["energy"].sum()),
                n_pixels=len(g),
                n_triggers=int(g["trigger_index"].nunique()),
                ifos=sorted(g["ifo"].dropna().unique().tolist()) if "ifo" in g else [],
            ))
        return pd.DataFrame(rows, columns=PIXEL_CLUSTERED_EVENT_COLUMNS)
