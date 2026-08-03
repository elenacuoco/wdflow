"""Time-slide background estimation and false-alarm-probability for
CoincidenceFinder (or GNNCoincidenceScorer) candidates.

Applies the standard GW-search shift-based null-distribution technique (a
non-physical time shift decorrelates real coincident signals while leaving
each detector's own noise statistics intact, so recomputing the coincidence
statistic under many shifts estimates how often detectors would coincide by
pure accident) to cross-detector trigger timing.

CAVEAT (see also roc.py): with a single continuous real-data segment rather
than months of real background, shifts within that one segment are not fully
independent in the way day-scale slides are in production pipelines -- a
single loud glitch can dominate many shifts. FAP/FAR here should be read as
"within this analyzed segment", not a calibrated events/year rate.
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
        """
        :type coincidence_finder: CoincidenceFinder
        :param coincidence_finder: the same finder (with the same settings) used
            for the real, unshifted foreground candidates, so background and
            foreground are directly comparable.
        :type n_slides: int
        :param n_slides: number of independent time shifts to draw.
        :type min_shift_s: float
        :param min_shift_s: minimum shift magnitude, seconds (keeps a shift from
            landing close enough to zero that it barely decorrelates real signals).
        :type max_shift_s: float | None
        :param max_shift_s: maximum shift magnitude, seconds; default (None) uses
            the shifted detector's own segment span minus `min_shift_s`.
        :type seed: int
        :param seed: RNG seed for reproducible shifts.
        """
        self.coincidence_finder = coincidence_finder
        self.n_slides = n_slides
        self.min_shift_s = min_shift_s
        self.max_shift_s = max_shift_s
        self.seed = seed

    def background_distribution(
        self,
        clustered: dict[str, pd.DataFrame],
        segment_bounds: dict[str, tuple[float, float]],
        finder_method: str = "find",
        **finder_kwargs,
    ) -> pd.DataFrame:
        """clustered: {ifo: clustered_events_df}. segment_bounds: {ifo:
        (gps_start, gps_end)} of the analyzed segment for that detector,
        needed to wrap shifted times back into the segment.

        The first IFO (dict insertion order) is kept fixed as the time
        reference; every other IFO's `gpsMax`/`gpsStart` are shifted by a
        random offset (modular wraparound within that IFO's segment,
        min_shift_s <= abs(shift) <= max_shift_s, sign random) drawn fresh per
        slide, then `coincidence_finder.<finder_method>(shifted, **finder_kwargs)`
        is rerun on the shifted copies -- `finder_method="find_network"` for a
        3+ detector background (see `CoincidenceFinder.find_network`).

        Output: one row per (accidental) candidate found across all slides,
        with an added `slide_index` column. Slides producing no accidental
        coincidence contribute zero rows -- not padded with NaN.

        :type clustered: dict[str, pandas.DataFrame]
        :param clustered: {ifo: TriggerClusterer.clustered_events output for that
            detector}, the real (unshifted) per-IFO clustered events.
        :type segment_bounds: dict[str, tuple[float, float]]
        :param segment_bounds: {ifo: (gps_start, gps_end)} of the analyzed segment
            for that detector.
        :type finder_method: str
        :param finder_method: name of the `CoincidenceFinder` method to call on each
            slide's shifted candidates -- "find" (pairwise) or "find_network" (N-way).
        :param finder_kwargs: forwarded to `finder_method` on every slide (e.g.
            `min_ifos=3` for `find_network`).
        :return: pandas.DataFrame -- one row per accidental candidate across all
            slides, same columns as `finder_method`'s own output plus `slide_index`;
            empty if no slide produced any accidental coincidence.
        :raises ValueError: if `clustered` has fewer than 2 detectors, or if any
            shiftable detector's segment is too short for `min_shift_s`/`max_shift_s`.
        """
        finder = getattr(self.coincidence_finder, finder_method)
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

            candidates = finder(shifted, **finder_kwargs)
            if candidates.empty:
                continue
            candidates = candidates.copy()
            candidates["slide_index"] = slide_index
            rows.append(candidates)

        if not rows:
            cols = list(finder({}, **finder_kwargs).columns) + ["slide_index"]
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

    def rank_candidates(
        self,
        candidates: pd.DataFrame,
        background: pd.DataFrame,
        score_col: str = "network_snr",
        segment_duration_s: float | None = None,
    ) -> pd.DataFrame:
        """Vectorized `false_alarm_probability` over every row of `candidates`
        at once (a sorted-background-array + `searchsorted` lookup instead of
        a per-row `(background[score_col] >= score).sum()` scan), for the
        common case of ranking a whole candidate list rather than checking
        one candidate picked by some other means (e.g. GPS proximity to a
        known event).

        :type candidates: pandas.DataFrame
        :param candidates: candidate table (e.g. `CoincidenceFinder.find`/
            `find_network`'s output), one row per candidate.
        :type background: pandas.DataFrame
        :param background: `background_distribution`'s output -- the accidental
            background this same list should be ranked against.
        :type score_col: str
        :param score_col: column in both `candidates` and `background` used as the
            detection statistic.
        :type segment_duration_s: float | None
        :param segment_duration_s: if given, also reports `far_per_day` (with the
            same single-segment caveat as `false_alarm_probability`).
        :return: pandas.DataFrame -- `candidates` with `fap`/`n_background_ge`/
            `n_slides` columns added (plus `far_per_day` if `segment_duration_s` is
            given), sorted by `fap` ascending (most significant first). Empty input
            returns an empty DataFrame with those columns added.
        """
        out = candidates.copy()
        if out.empty:
            for col in ("fap", "n_background_ge", "n_slides"):
                out[col] = pd.Series(dtype=float if col == "fap" else int)
            if segment_duration_s is not None:
                out["far_per_day"] = pd.Series(dtype=float)
            return out

        bg_sorted = np.sort(background[score_col].to_numpy(dtype=float))
        scores = out[score_col].to_numpy(dtype=float)
        # n_ge = count of background scores >= score, via searchsorted on the
        # ascending-sorted background array: 'left' finds the first index
        # with bg_sorted[idx] >= score, so len(bg_sorted) - idx is exactly
        # that count.
        idx = np.searchsorted(bg_sorted, scores, side="left")
        n_ge = len(bg_sorted) - idx

        out["n_background_ge"] = n_ge
        out["n_slides"] = self.n_slides
        out["fap"] = (1 + n_ge) / (1 + self.n_slides)
        if segment_duration_s is not None:
            out["far_per_day"] = out["fap"] / (segment_duration_s / 86400.0)
        return out.sort_values("fap", kind="stable").reset_index(drop=True)


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
