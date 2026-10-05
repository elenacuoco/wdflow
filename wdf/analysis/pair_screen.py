"""Optional screens of the paper path's candidate pairs: the lag of the two
reconstructions against the light travel time, and the network's glitch
decision.

`ScreenedFinder` wraps the paper path's finder
(`WavegramCoincidenceFinder`). For every pair it returns, zero lag or time
slide alike, it measures:

- **the lag** (`NetworkLagCut`): ``network_lag_s``, the arrival-time
  difference of the two events' stitched reconstructions
  (`event_waveform`, `wdf.analysis.timing.arrival_time_difference`, searched
  over ``|tau| <= NetworkLagSearch``), first detector of `ifos_involved`
  minus the second, with its width ``network_lag_width_s``. Unlike the
  difference of the loudest tiles' instants (`dt_s`), it is read on the
  whole waveform.

- **the glitch decision** (`NetworkGlitchVeto`), when the detectors'
  census of transients is given (the online worker's ``GlitchCensusOnline``,
  `extent_first`/`extent_last`, or the offline census, `first`/`last`). A
  single detector cannot tell a loud merger from a glitch, so its census
  labels both. The network can: an event that overlaps a census transient of
  its detector is a glitch (``glitch_i``, ``glitch_j``) only when the other
  detector of the pair has NO census transient within the light travel time
  plus `NetworkGlitchTolerance` (0.01 s) of it. Transients coincident in the
  two detectors are not glitches. ``glitch_single_ifo`` is true when either
  event is one. "veto" drops those pairs; "penalty" multiplies the ranking
  column by `NetworkGlitchFactor` (0.1) for them; "none" (the default) only
  reports the columns. Online, the decision reads the other detector's
  census at the same latency as its triggers: run it once both detectors'
  outputs reach the same data time.

``window`` is set by `NetworkLagWindow`. "coincidence" (the default) is the
pair's admission tolerance of the classical coincidence,
`CoincidenceConfig.timing_tolerance` with the two events' timing spreads
(`tSpread`): the pair's light travel time plus `timing_sigma` times the
quadrature sum of the spreads, each at least `timing_jitter_s`, capped at
`maximum_tolerance` (`maximum_tolerance_travel_times` light travel times).
With the default configuration and no spread this is 10.0 + 14.8 = 24.9 ms
for H1-L1 (cap 25.0 ms), 42.1 ms for H1-V1 and 41.3 ms for L1-V1; long
spreads reach the cap, 25.0, 68.2 and 66.1 ms. Without a
`tSpread` column every event takes the jitter floor (logged).
"travel_plus_tolerance" is the pair's light travel time plus
`NetworkLagTolerance`. The window used is reported per pair
(``network_lag_window_s``, ``network_lag_window_rule``).
With "cut" a pair is kept only when ``|network_lag_s| <= window``; a pair
whose lag cannot be measured within the search is dropped. With "penalty"
every pair is kept and ``network_lag_factor``,

    ``1 / (1 + (max(0, |tau| - window) / scale)^2)``, with ``scale`` the pair's
own window when `NetworkLagScale` is "window" (the default) or that number of
seconds,

1 within the window, decreasing with the excess, the same for either sign,
0 when the lag cannot be measured, multiplies the ranking column into
``network_rank_statistic``, which the pairs are then ranked on.

A time slide moves an event's times; the screen reads each event's
displacement from its `gpsPeak` against the unslid events and moves its
reconstruction and its detector's census by it, so the slid pairs are
screened as the zero-lag ones.
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: The lag criteria (`NetworkLagCut`).
LAG_CUTS = ("none", "cut", "penalty")
#: The lag windows (`NetworkLagWindow`).
LAG_WINDOWS = ("coincidence", "travel_plus_tolerance")
#: `NetworkLagScale` value that makes the penalty's scale each pair's own window.
SCALE_IS_WINDOW = "window"
#: The column of an event's timing spread the coincidence reads (robust_events).
SPREAD_COLUMN = "tSpread"
#: The uses of the glitch decision (`NetworkGlitchVeto`).
GLITCH_VETOES = ("none", "veto", "penalty")
#: The column the pairs are ranked on when a penalty applies.
RANK_STATISTIC = "network_rank_statistic"


@dataclass
class PairScreenSettings:
    """The screens' choices, read from the run parameters.

    :param lag_cut: `NetworkLagCut`, one of `LAG_CUTS` (default "none").
    :param lag_window: `NetworkLagWindow`, one of `LAG_WINDOWS` (default
        "coincidence"): "coincidence" is each pair's admission tolerance of
        the classical coincidence, `CoincidenceConfig.timing_tolerance` with
        the two events' timing spreads (`tSpread`); "travel_plus_tolerance" is
        the pair's light travel time plus `NetworkLagTolerance`.
    :param lag_tolerance: `NetworkLagTolerance`, s added to the light travel
        time with "travel_plus_tolerance" (default 0.01).
    :param lag_scale: `NetworkLagScale`, the scale of the lag penalty: "window"
        (default) for each pair's own window, or a number of seconds.
    :param lag_search: `NetworkLagSearch`, half-width of the lag search, s
        (default `wdf.analysis.timing.MAX_LAG_S`, 0.25).
    :param glitch_veto: `NetworkGlitchVeto`, one of `GLITCH_VETOES`
        (default "none").
    :param glitch_tolerance: `NetworkGlitchTolerance`, s added to the light
        travel time for two transients to be coincident (default 0.01).
    :param glitch_factor: `NetworkGlitchFactor`, the rank factor of a
        single-detector glitch with "penalty" (default 0.1).
    """

    lag_cut: str = "none"
    lag_window: str = "coincidence"
    lag_tolerance: float = 0.01
    lag_scale: object = SCALE_IS_WINDOW
    lag_search: float = 0.25
    glitch_veto: str = "none"
    glitch_tolerance: float = 0.01
    glitch_factor: float = 0.1

    KEYS = dict(lag_cut="NetworkLagCut", lag_window="NetworkLagWindow", lag_tolerance="NetworkLagTolerance",
                lag_scale="NetworkLagScale", lag_search="NetworkLagSearch",
                glitch_veto="NetworkGlitchVeto", glitch_tolerance="NetworkGlitchTolerance",
                glitch_factor="NetworkGlitchFactor")
    WORDS = ("lag_cut", "lag_window", "glitch_veto")

    @classmethod
    def from_parameters(cls, par):
        """The settings a run asks for, defaults filled in and written back.

        :raises ValueError: on an unknown criterion or a non-positive scale.
        """
        out = cls()
        for name, key in cls.KEYS.items():
            value = getattr(par, key, None)
            if value is not None:
                if name in cls.WORDS or (name == "lag_scale" and str(value).strip().lower() == SCALE_IS_WINDOW):
                    value = str(value).strip().lower()
                else:
                    value = float(value)
                setattr(out, name, value)
            setattr(par, key, getattr(out, name))
        if out.lag_cut not in LAG_CUTS:
            raise ValueError(f"NetworkLagCut {out.lag_cut!r}: use one of {LAG_CUTS}")
        if out.glitch_veto not in GLITCH_VETOES:
            raise ValueError(f"NetworkGlitchVeto {out.glitch_veto!r}: use one of {GLITCH_VETOES}")
        if not 0 <= out.glitch_factor <= 1:
            raise ValueError("NetworkGlitchFactor must lie in [0, 1]")
        if out.lag_window not in LAG_WINDOWS:
            raise ValueError(f"NetworkLagWindow {out.lag_window!r}: use one of {LAG_WINDOWS}")
        if (out.lag_scale != SCALE_IS_WINDOW and out.lag_scale <= 0) or out.lag_search <= 0:
            raise ValueError(f"NetworkLagScale must be {SCALE_IS_WINDOW!r} or positive, and NetworkLagSearch "
                             "positive")
        return out

    @property
    def active(self):
        """Whether any screen applies."""
        return self.lag_cut != "none" or self.glitch_veto != "none"

    @property
    def penalised(self):
        """Whether the ranking column is multiplied by a factor."""
        return self.lag_cut == "penalty" or self.glitch_veto == "penalty"


def census_extents(census):
    """A census table's transients as (start, end) arrays, GPS: the online
    census's `extent_first`/`extent_last`, else the offline `first`/`last`."""
    if census is None or len(census) == 0:
        return np.zeros(0), np.zeros(0)
    for a, b in (("extent_first", "extent_last"), ("first", "last")):
        if a in census and b in census:
            return (pd.to_numeric(census[a]).to_numpy(float),
                    pd.to_numeric(census[b]).to_numpy(float))
    raise ValueError("a census table needs extent_first/extent_last or first/last")


def single_ifo_glitch(span, own, other, window):
    """Whether an event is a glitch of its detector alone.

    :param span: the event's (start, end), GPS.
    :param own: its detector's census, (starts, ends), already displaced.
    :param other: the other detector's census, (starts, ends), displaced.
    :param window: light travel time plus tolerance, s.
    :return: bool -- true when the event overlaps a transient of its own
        detector that has no transient of the other within `window`.
    """
    s0, s1 = span
    for a, b in zip(*own):
        if a <= s1 and s0 <= b:
            if not np.any((other[0] - window <= b) & (a <= other[1] + window)):
                return True
    return False


def lag_windows(ifo_a, ifo_b, spread_a, spread_b, settings, config=None):
    """Each pair's lag window, s.

    "coincidence": the classical coincidence's tolerance for the pair,
    `CoincidenceConfig.timing_tolerance(spread_a, spread_b, (ifo_a, ifo_b))`
    = min(travel + timing_sigma * hypot(max(spread, timing_jitter_s)),
    maximum_tolerance), every constant taken from `config`. A missing spread
    (NaN) counts as zero, so the jitter floor applies.
    "travel_plus_tolerance": the pair's light travel time plus
    `settings.lag_tolerance`.

    :param ifo_a: first detectors (array of names).
    :param ifo_b: second detectors.
    :param spread_a: their events' timing spreads, s (array).
    :param spread_b: the second events' spreads, s.
    :type settings: PairScreenSettings
    :param config: the `CoincidenceConfig` (default one when None).
    :return: numpy.ndarray
    """
    from wdf.analysis.detectors import light_travel_time

    if settings.lag_window == "travel_plus_tolerance":
        return np.array([light_travel_time(a, b) for a, b in zip(ifo_a, ifo_b)]) + settings.lag_tolerance
    if config is None:
        from wdf.analysis.robust_events import CoincidenceConfig

        config = CoincidenceConfig()
    sa = np.nan_to_num(np.asarray(spread_a, dtype=float), nan=0.0)
    sb = np.nan_to_num(np.asarray(spread_b, dtype=float), nan=0.0)
    return np.array([float(config.timing_tolerance(x, y, (a, b)))
                     for a, b, x, y in zip(ifo_a, ifo_b, sa, sb)])


def lag_factor(lag, window, scale):
    """``1 / (1 + (max(0, |lag| - window) / scale)^2)``; 0 for a NaN lag.

    :param lag: lags, s (array).
    :param window: the allowed lag, s (array or scalar).
    :param scale: s (array or scalar).
    :return: numpy.ndarray
    """
    lag = np.asarray(lag, dtype=float)
    excess = np.maximum(0.0, np.abs(lag) - window)
    return np.where(np.isfinite(lag), 1.0 / (1.0 + (excess / scale) ** 2), 0.0)


class ScreenedFinder:
    """A paper-path finder whose pairs are screened by their reconstructions'
    lag.

    It satisfies the interface `TimeSlideFAR` expects (`find`, and the
    coincidence configuration as `config`).

    :param finder: the finder screened (`WavegramCoincidenceFinder`).
    :param events: ``{ifo: event table}``, unslid.
    :param reconstruction: ``{ifo: callable(cluster_id) -> (t0, samples)}``,
        each event's stitched reconstruction.
    :param fs: their sampling rate, Hz.
    :type settings: PairScreenSettings
    :param rank_column: the column `network_rank_statistic` multiplies.
    :param census: ``{ifo: census table}`` or None; with it the glitch
        columns are reported.
    """

    def __init__(self, finder, events, reconstruction, fs, settings, rank_column, census=None):
        if settings.glitch_veto != "none" and census is None:
            raise ValueError("NetworkGlitchVeto needs the detectors' census of transients")
        self.census = None if census is None else {i: census_extents(c) for i, c in census.items()}
        self.finder = finder
        inner = getattr(finder, "finder", None)
        self.config = getattr(finder, "config", None) or getattr(inner, "config", None)
        self.builder = getattr(finder, "builder", None)
        self.reconstruction, self.fs = reconstruction, float(fs)
        self.settings, self.rank_column = settings, rank_column
        self.reference = {ifo: dict(zip(ev["cluster_id"].astype(int),
                                        pd.to_numeric(ev["gpsPeak"]).to_numpy(float)))
                          for ifo, ev in events.items()}

    def _lag(self, ifo_a, ca, da, ifo_b, cb, db):
        from wdf.analysis.timing import arrival_time_difference

        try:
            ta, ya = self.reconstruction[ifo_a](ca)
            tb, yb = self.reconstruction[ifo_b](cb)
            return arrival_time_difference((ta + da, ya), (tb + db, yb), self.fs,
                                           max_lag_s=self.settings.lag_search)
        except (ValueError, KeyError):
            return float("nan"), float("nan")

    def find(self, events_by_ifo):
        """The pairs of the screened finder, with the screens applied.

        :param events_by_ifo: ``{ifo: event table}``, possibly slid.
        :return: pandas.DataFrame
        """
        from wdf.analysis.detectors import light_travel_time

        table = self.finder.find(events_by_ifo)
        if table.empty or not (self.settings.active or self.census is not None):
            return table
        ifos = list(events_by_ifo)
        nodes = pd.concat([events_by_ifo[i].reset_index(drop=True).assign(ifo=i) for i in ifos],
                          ignore_index=True)
        ifo = nodes["ifo"].to_numpy()
        cid = nodes["cluster_id"].to_numpy(int)
        peak = pd.to_numeric(nodes["gpsPeak"]).to_numpy(float)
        disp = np.array([peak[k] - self.reference[ifo[k]].get(int(cid[k]), peak[k])
                         for k in range(len(nodes))])
        disp = np.where(np.isfinite(disp), disp, 0.0)
        i = table["node_i"].to_numpy(int)
        j = table["node_j"].to_numpy(int)
        out = table.copy()
        out["ifo_i"], out["cluster_i"] = ifo[i], cid[i]
        out["ifo_j"], out["cluster_j"] = ifo[j], cid[j]
        if SPREAD_COLUMN in nodes:
            spread = pd.to_numeric(nodes[SPREAD_COLUMN], errors="coerce").to_numpy(float)
            rule = f"{self.settings.lag_window} ({SPREAD_COLUMN})"
        else:
            spread = np.zeros(len(nodes))
            rule = f"{self.settings.lag_window} (no {SPREAD_COLUMN}: jitter floor)"
            if self.settings.lag_window == "coincidence" and not getattr(self, "_warned", False):
                logging.warning("NetworkLagWindow 'coincidence': the events carry no %s; each event's spread "
                                "is the coincidence's jitter floor", SPREAD_COLUMN)
                self._warned = True
        config = self.config if hasattr(self.config, "timing_tolerance") else None
        window = lag_windows(ifo[i], ifo[j], spread[i], spread[j], self.settings, config)
        keep = np.ones(len(out), bool)
        factor = np.ones(len(out))
        if self.settings.lag_cut != "none":
            lags = [self._lag(ifo[p], cid[p], disp[p], ifo[q], cid[q], disp[q])
                    for p, q in zip(i, j)]
            out["network_lag_s"] = [lag for lag, _ in lags]
            out["network_lag_width_s"] = [width for _, width in lags]
            out["network_lag_window_s"] = window
            out["network_lag_window_rule"] = rule
            lag = out["network_lag_s"].to_numpy(float)
            if self.settings.lag_cut == "cut":
                keep &= np.isfinite(lag) & (np.abs(lag) <= window)
            else:
                scale = window if self.settings.lag_scale == SCALE_IS_WINDOW else self.settings.lag_scale
                out["network_lag_factor"] = lag_factor(lag, window, scale)
                factor = factor * out["network_lag_factor"].to_numpy(float)
        if self.census is not None:
            start = pd.to_numeric(nodes["gpsStart"]).to_numpy(float)
            end = start + pd.to_numeric(nodes["duration"]).to_numpy(float)
            empty = (np.zeros(0), np.zeros(0))

            def moved(name, d):
                a, b = self.census.get(name, empty)
                return a + d, b + d

            gwin = np.array([light_travel_time(a, b) for a, b in zip(ifo[i], ifo[j])]) \
                + self.settings.glitch_tolerance
            gi, gj = [], []
            for k, (p, q) in enumerate(zip(i, j)):
                own_p, own_q = moved(ifo[p], disp[p]), moved(ifo[q], disp[q])
                gi.append(single_ifo_glitch((start[p] + disp[p], end[p] + disp[p]), own_p, own_q,
                                            gwin[k]))
                gj.append(single_ifo_glitch((start[q] + disp[q], end[q] + disp[q]), own_q, own_p,
                                            gwin[k]))
            out["glitch_i"], out["glitch_j"] = gi, gj
            flagged = np.array(gi, bool) | np.array(gj, bool)
            out["glitch_single_ifo"] = flagged
            if self.settings.glitch_veto == "veto":
                keep &= ~flagged
            elif self.settings.glitch_veto == "penalty":
                factor = factor * np.where(flagged, self.settings.glitch_factor, 1.0)
        if self.settings.penalised:
            out[RANK_STATISTIC] = out[self.rank_column].to_numpy(float) * factor
        out = out[keep].reset_index(drop=True)
        if "candidate_id" in out:
            out["candidate_id"] = np.arange(len(out))
        return out
