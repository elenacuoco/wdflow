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
    "cluster_id", "is_noise", "trigger_index", "ifo", "gpsStart", "gpsPeak", "EnWDF",
    "freqMean", "freqMax", "freqMin", "freqPeak", "duration", "wave",
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
        stat_weight: float = 0.0,
        time_col: str = "gpsPeak",
        freq_col: str = "freqMean",
        stat_col: str = "EnWDF",
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
        :type stat_weight: float
        :param stat_weight: if > 0, adds `stat_weight * log10(stat_col)` as a third
            clustering feature alongside time/frequency (0 disables it).
        :type time_col: str
        :param time_col: trigger column used as the time-axis clustering feature.
        :type freq_col: str
        :param freq_col: trigger column used as the frequency-axis clustering feature.
            `freqMean` is a spectral moment and tracks a narrowband carrier more
            closely than `freqPeak`, which answers where the local signal-to-noise
            ratio peaks and is the sharper of the two only for a sweeping transient.
        :type stat_col: str
        :param stat_col: trigger column used to rank cluster members (peak selection
            in `clustered_events`) and as the optional third clustering feature.
        :raises ValueError: if `method` is not "dbscan" or "greedy".
        """
        if method not in ("dbscan", "greedy"):
            raise ValueError(f"method must be 'dbscan' or 'greedy', got {method!r}")
        self.method = method
        self.time_eps_s = time_eps_s
        self.freq_eps_hz = freq_eps_hz
        self.min_samples = min_samples
        self.stat_weight = stat_weight
        self.time_col = time_col
        self.freq_col = freq_col
        self.stat_col = stat_col

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
        if self.stat_weight > 0:
            statistic = np.log10(np.clip(df[self.stat_col].to_numpy(dtype=float), 1e-12, None))
            features.append(self.stat_weight * statistic)
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
            peak = g.loc[g[self.stat_col].idxmax()]
            n = len(g)
            gps_span_s = float(g[self.time_col].max() - g[self.time_col].min())
            is_noise = bool(g[CLUSTER_COL].iloc[0] == -1)
            rows.append(dict(
                cluster_id=int(g[CLUSTER_COL].iloc[0]),
                is_noise=is_noise,
                trigger_index=int(g["_row_index"].iloc[0]) if is_noise else -1,
                ifo=g["ifo"].iloc[0] if "ifo" in g.columns else None,
                gpsStart=float(g["gps"].min()) if "gps" in g.columns else float(g[self.time_col].min()),
                gpsPeak=float(peak[self.time_col]),
                EnWDF=float(peak[self.stat_col]),
                freqMean=float(g["freqMean"].mean()) if "freqMean" in g.columns else float(peak[self.freq_col]),
                freqMax=float(peak["freqMax"]) if "freqMax" in g.columns else float(peak[self.freq_col]),
                freqMin=float(g["freqMin"].min()) if "freqMin" in g.columns else float(peak[self.freq_col]),
                freqPeak=float(peak[self.freq_col]),
                duration=gps_span_s if n > 1 else float(peak.get("duration", 0.0)),
                wave=peak.get("wave"),
                n_triggers=n,
                gps_span_s=gps_span_s,
            ))
        return pd.DataFrame(rows, columns=CLUSTERED_EVENT_COLUMNS)


PIXEL_COLUMNS = ["trigger_index", "ifo", "t_lo", "t_hi", "f_lo", "f_hi", "energy", "sigma"]
PIXEL_CLUSTERED_EVENT_COLUMNS = [
    "cluster_id", "ifo", "gpsStart", "gpsEnd", "gpsPeak", "duration",
    "freqMin", "freqMax", "freqPeak", "EnWDF", "sigma",
    "total_energy", "n_pixels", "n_triggers", "ifos",
]


def collect_significant_pixels(triggers: pd.DataFrame, fs: float) -> pd.DataFrame:
    """Time-frequency tiles of the non-zero wavelet coefficients, in absolute GPS.

    Each trigger's surviving coefficients are placed at their own tiles and
    shifted to absolute time by the trigger's `gps`. A trigger carries exactly
    the coefficients that passed thresholding, so every one of them is a tile.

    :type triggers: pandas.DataFrame
    :param triggers: triggers carrying `gps` and the coefficient columns;
        triggers without coefficients are skipped.
    :type fs: float
    :param fs: sampling rate the coefficients were computed at, Hz.
    :return: pandas.DataFrame -- one row per tile with `PIXEL_COLUMNS`:
        `trigger_index`, `ifo`, `t_lo`, `t_hi`, `f_lo`, `f_hi`, `energy`.
    """
    from wdf.analysis.coefficients import SparseCoefficients

    if "wt_index" not in triggers:
        return pd.DataFrame(columns=PIXEL_COLUMNS)

    rows = []
    for idx, trig in triggers.iterrows():
        record = SparseCoefficients(
            index=np.asarray(trig["wt_index"]), value=np.asarray(trig["wt_value"]),
            n_coeff=int(trig["n_coeff"]), wave=str(trig.get("wave", "")),
            sigma=float(trig.get("sigma", np.nan)), fs=fs)
        tiles = record.tiles()
        gps0 = trig["gps"]
        for t_lo, t_hi, f_lo, f_hi, magnitude in zip(
                tiles["t_lo"], tiles["t_hi"], tiles["f_lo"], tiles["f_hi"],
                tiles["magnitude"]):
            rows.append(dict(
                trigger_index=idx, ifo=trig.get("ifo"),
                t_lo=gps0 + t_lo, t_hi=gps0 + t_hi,
                f_lo=f_lo, f_hi=f_hi,
                energy=magnitude ** 2,
                sigma=record.sigma,
            ))
    return pd.DataFrame(rows, columns=PIXEL_COLUMNS)


class WaveletPixelClusterer:
    """Connected-component clustering directly on WDF's own wavelet-coefficient
    time-frequency tiles, instead of on each trigger's single (gpsPeak,
    freqPeak) point summary -- the same idea coherent WaveBurst (cWB) uses
    (percolation over connected pixels in a time-frequency map). WDF already
    computes the full per-trigger tile map;
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

    def trigger_labels(self, pixels: pd.DataFrame) -> pd.Series:
        """Cluster label per trigger, for grouping triggers before stitching.

        Overlapping windows put pixels of the same transient in several
        triggers, and one trigger's window can touch more than one cluster; the
        trigger is assigned to the cluster it deposits most energy in.

        :type pixels: pandas.DataFrame
        :param pixels: `fit`'s output.
        :return: pandas.Series -- cluster label indexed by `trigger_index`.
        """
        if pixels.empty:
            return pd.Series(dtype=int)
        totals = pixels.groupby(["trigger_index", "cluster_id"])["energy"].sum()
        return totals.reset_index().sort_values("energy").groupby(
            "trigger_index")["cluster_id"].last()

    def clustered_events(self, pixels: pd.DataFrame) -> pd.DataFrame:
        """One row per connected component: the cluster's real time/frequency
        span (not one window's worth), the tile where the local
        signal-to-noise ratio peaks, and `EnWDF` on the noise scale.

        `total_energy` sums each member pixel's squared coefficient. Because
        consecutive WDF windows overlap, pixels covering the same samples are
        counted once per window they appear in, so `EnWDF` derived from it is
        an upper bound. `wavegram_events` reconstructs the cluster in the time
        domain instead, where each sample is counted exactly once, and is the
        value to release.

        :type pixels: pandas.DataFrame
        :param pixels: `fit`'s output.
        :return: pandas.DataFrame -- one row per cluster,
            `PIXEL_CLUSTERED_EVENT_COLUMNS`.
        """
        from wdf.analysis.wavelets import tile_frequency

        if pixels.empty:
            return pd.DataFrame(columns=PIXEL_CLUSTERED_EVENT_COLUMNS)
        rows = []
        for cid, g in pixels.groupby("cluster_id"):
            loudest = g.loc[g["energy"].idxmax()]
            scales = g["sigma"].to_numpy(dtype=float) if "sigma" in g else np.array([])
            scales = scales[np.isfinite(scales)]
            sigma = float(np.median(scales)) if scales.size else float("nan")
            total_energy = float(g["energy"].sum())
            gps_start = float(g["t_lo"].min())
            gps_end = float(g["t_hi"].max())
            rows.append(dict(
                cluster_id=int(cid),
                ifo=g["ifo"].iloc[0] if "ifo" in g else None,
                gpsStart=gps_start,
                gpsEnd=gps_end,
                gpsPeak=0.5 * float(loudest["t_lo"] + loudest["t_hi"]),
                duration=gps_end - gps_start,
                freqMin=float(g["f_lo"].min()),
                freqMax=float(g["f_hi"].max()),
                freqPeak=tile_frequency(float(loudest["f_lo"]), float(loudest["f_hi"])),
                EnWDF=float(np.sqrt(total_energy) / sigma) if sigma > 0 else float("nan"),
                sigma=sigma,
                total_energy=total_energy,
                n_pixels=len(g),
                n_triggers=int(g["trigger_index"].nunique()),
                ifos=sorted(g["ifo"].dropna().unique().tolist()) if "ifo" in g else [],
            ))
        return pd.DataFrame(rows, columns=PIXEL_CLUSTERED_EVENT_COLUMNS)


def wavegram_events(triggers: pd.DataFrame, fs: float, window: int, overlap: int,
                    time_tol_s: float) -> pd.DataFrame:
    """Cluster triggers on the wavegram and score each cluster on its
    time-domain reconstruction.

    Percolation over the wavelet-coefficient tiles decides which windows belong
    to the same transient, so a signal longer than the analysis window is one
    event rather than a chain of partial ones. The event's `EnWDF` is then the
    norm of the reconstruction stitched across those windows, divided by the
    noise scale: each sample is counted exactly once, and the statistic covers
    the signal's whole extent instead of the best single window.

    :type triggers: pandas.DataFrame
    :param triggers: one detector's triggers, carrying `gps`, `sigma`, `wave`
        and the coefficient columns.
    :type fs: float
    :param fs: analysis sampling frequency, Hz.
    :type window: int
    :param window: analysis window length, samples.
    :type overlap: int
    :param overlap: overlap between consecutive windows, samples.
    :type time_tol_s: float
    :param time_tol_s: time gap two tiles may leave and still be chained into
        the same cluster, seconds.
    :return: pandas.DataFrame -- one row per event, with the geometry from the
        wavegram and `EnWDF` from the reconstruction.
    """
    from wdf.analysis.reconstruction import combined_snr

    clusterer = WaveletPixelClusterer(time_tol_s=time_tol_s)
    pixels = clusterer.fit(triggers, fs)
    events = clusterer.clustered_events(pixels)
    if events.empty:
        return events

    labels = clusterer.trigger_labels(pixels)
    reconstructed = []
    for _, event in events.iterrows():
        members = triggers.loc[labels.index[labels == event.cluster_id]]
        summary = combined_snr(members, fs, window, overlap)
        reconstructed.append(dict(
            cluster_id=event.cluster_id,
            EnWDF=summary["EnWDF"],
            loudest_window=summary["loudest_window"],
            span_s=summary["span_s"],
            windows=summary["windows"],
        ))

    return events.drop(columns=["EnWDF"]).merge(
        pd.DataFrame(reconstructed), on="cluster_id")
