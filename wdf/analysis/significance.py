"""Time-slide background estimation and false-alarm-probability for
CoincidenceFinder (or GNNCoincidenceScorer) candidates.

Mirrors the shift-based null-distribution pattern already used in
tandem-interact's tandemLib.characterization.timeslide_z / witness_coincidence
(rng-driven time shift -> recompute statistic -> compare candidate against the
resulting null distribution), but applied to cross-detector trigger
coincidence instead of phase coherence -- the standard GW-search technique for
estimating how often detectors would coincide by pure accident.

CAVEAT (see also roc.py and the project notebook): with a single continuous
GWOSC segment rather than months of real background, shifts within that one
segment are not fully independent in the way day-scale slides are in
production pipelines -- a single loud glitch can dominate many shifts. FAP/FAR
here should be read as "within this analyzed segment", not a calibrated
events/year rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .coincidence import CoincidenceFinder


class BackgroundEstimator:
    """Accidental-coincidence background via non-physical GPS time shifts."""

    def __init__(
        self,
        coincidence_finder: CoincidenceFinder,
        n_slides: int = 200,
        min_shift_s: float = 1.0,
        max_shift_s: float | None = None,
        seed: int = 0,
    ):
        self.coincidence_finder = coincidence_finder
        self.n_slides = n_slides
        self.min_shift_s = min_shift_s
        self.max_shift_s = max_shift_s
        self.seed = seed

    def background_distribution(
        self,
        clustered: dict[str, pd.DataFrame],
        segment_bounds: dict[str, tuple[float, float]],
    ) -> pd.DataFrame:
        """clustered: {ifo: clustered_events_df}. segment_bounds: {ifo:
        (gps_start, gps_end)} of the analyzed segment for that detector,
        needed to wrap shifted times back into the segment.

        The first IFO (dict insertion order) is kept fixed as the time
        reference; every other IFO's `gpsMax`/`gpsStart` are shifted by a
        random offset (modular wraparound within that IFO's segment,
        min_shift_s <= abs(shift) <= max_shift_s, sign random) drawn fresh per
        slide, then CoincidenceFinder.find is rerun on the shifted copies.

        Output: one row per (accidental) candidate found across all slides,
        with an added `slide_index` column. Slides producing no accidental
        coincidence contribute zero rows -- not padded with NaN.
        """
        ifos = list(clustered.keys())
        if len(ifos) < 2:
            raise ValueError("need >= 2 detectors to build a coincidence background")
        reference_ifo = ifos[0]
        shiftable_ifos = ifos[1:]

        rng = np.random.default_rng(self.seed)
        rows = []
        for slide_index in range(self.n_slides):
            shifted = {reference_ifo: clustered[reference_ifo]}
            for ifo in shiftable_ifos:
                gps_start, gps_end = segment_bounds[ifo]
                span = gps_end - gps_start
                max_shift = self.max_shift_s if self.max_shift_s is not None else span - self.min_shift_s
                if max_shift <= self.min_shift_s:
                    raise ValueError(
                        f"segment for {ifo} too short ({span}s) for "
                        f"min_shift_s={self.min_shift_s}/max_shift_s={max_shift}"
                    )
                magnitude = rng.uniform(self.min_shift_s, max_shift)
                shift = magnitude if rng.integers(0, 2) else -magnitude

                df = clustered[ifo].copy()
                for col in ("gpsMax", "gpsStart"):
                    if col in df.columns:
                        df[col] = gps_start + ((df[col] - gps_start + shift) % span)
                shifted[ifo] = df

            candidates = self.coincidence_finder.find(shifted)
            if candidates.empty:
                continue
            candidates = candidates.copy()
            candidates["slide_index"] = slide_index
            rows.append(candidates)

        if not rows:
            cols = list(self.coincidence_finder.find({}).columns) + ["slide_index"]
            return pd.DataFrame(columns=cols)
        return pd.concat(rows, ignore_index=True)

    def false_alarm_probability(
        self,
        candidate_row: pd.Series,
        background: pd.DataFrame,
        score_col: str = "network_snr",
        segment_duration_s: float | None = None,
    ) -> dict:
        """Empirical-tail "+1" FAP estimator: avoids FAP=0 from a finite
        number of background trials, the standard convention for small
        time-slide background counts.

        far_per_day is reported ONLY if segment_duration_s is given, and is
        only as reliable as the single-segment background it's derived
        from -- see module-level caveat.
        """
        score = candidate_row[score_col]
        n_ge = int((background[score_col] >= score).sum())
        fap = (1 + n_ge) / (1 + self.n_slides)
        result = dict(fap=fap, n_background_ge=n_ge, n_slides=self.n_slides)
        if segment_duration_s is not None:
            result["far_per_day"] = fap / (segment_duration_s / 86400.0)
        return result


def pool_backgrounds(backgrounds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pool background_distribution outputs from multiple segments into one
    null sample, tagging each row's originating segment via `segment_id` so
    per-segment vs. pooled results can both be inspected (per-segment
    background alone is often too sparse to be useful on its own).
    """
    frames = []
    for segment_id, df in backgrounds.items():
        if df.empty:
            continue
        tagged = df.copy()
        tagged["segment_id"] = segment_id
        frames.append(tagged)
    if not frames:
        return pd.DataFrame(columns=["segment_id"])
    return pd.concat(frames, ignore_index=True)
