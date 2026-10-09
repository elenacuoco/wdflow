"""One candidate list from the stage 2-4 paths of the same stage 1.

The wavegram path and the track path (`wdf.analysis.network_driver`, docs
``design/online-offline.md``) each turn one stage 1 into its own candidate
table with its own time-slide background and false-alarm rate. This module
puts them side by side: a candidate found by either path is a row of the
combined list; when both find the same pair of events the two rows become one
that carries both statistics and both rates, and its false-alarm rate is the
better of the two times a trials factor, since the lower of two independently
calibrated rates is read as many times as there are paths.

Matching (`CombineMatch`): two candidates of different paths are the same
event when they lie within `CombineWindow` of each other, read on the
per-detector instants (``instant_<ifo>``; "instant", the default: every
detector of the pair within the window), on the candidate times
(``gps_candidate``; "candidate") or on the per-detector event extents
(``gpsStart_<ifo>``, ``duration_<ifo>``; "extent": the extents overlap, each
padded by the window). The default window is the coincidence window of the
pair (`CoincidenceConfig.maximum_tolerance`, 2.5 light travel times, 25 ms
for H1-L1). The matching is one to one, closest first.

Time and per-detector columns of a merged row come from the first path of
`CombineTimePreference` that found it (default the wavegram path, whose
envelope instant is the better clock; the track path otherwise).

The combined rate is `far_per_day` = min over the paths found x
`NetworkTrialsFactor` (default: the number of paths combined); the rate and
statistic of each path stay in ``far_per_day_<path>`` and
``statistic_<path>``; ``combined_statistic`` = -log10 of the combined rate per
day ranks the list (higher = more signal-like) so that the usual evaluation
tools read it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wdf.analysis.robust_events import CoincidenceConfig

#: Default `CombineMatch`: the per-detector instants within the window.
DEFAULT_MATCH = "instant"
#: The match rules.
MATCH_RULES = ("instant", "candidate", "extent")
#: Default `CombineTimePreference`: the wavegram path's time when it found the
#: candidate, else the track path's.
DEFAULT_TIME_PREFERENCE = ("w", "t")
#: The per-detector columns carried from the preferred path, when present.
PER_IFO_COLUMNS = ("instant", "gpsStart", "duration", "gpsPeak", "gpsEnd", "track_t_start",
                   "track_t_end", "hrss", "hrss_err", "hrss_noise")
#: Row columns carried from the preferred path, when present.
ROW_COLUMNS = ("gps_candidate", "ifos_involved", "n_ifos", "event_instant", "stretch",
               "network_enwdf", "network_min_enwdf", "hrss_network_det")
#: Per-path columns kept for every path, suffixed by the path name.
PATH_COLUMNS = ("candidate_id", "gps_candidate", "far_per_day", "n_background_ge", "rank")


def _positive(key, value):
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{key} must be positive, got {value!r}")
    return v


@dataclass
class CombineSettings:
    """The choices of the combination, read from the run parameters.

    :param window_s: `CombineWindow`, seconds, or None (default): the
        coincidence window of the detector pair.
    :param match: `CombineMatch`, "instant" (default), "candidate" or
        "extent".
    :param trials_factor: `NetworkTrialsFactor`, multiplies the best path's
        rate; None (default) means the number of paths combined.
    :param time_preference: `CombineTimePreference`, the path names in the
        order their time is preferred (default ("w", "t")).
    :param ifos: the detectors of the pair, for the default window and the
        per-detector columns (default ("H1", "L1")).
    """

    window_s: float | None = None
    match: str = DEFAULT_MATCH
    trials_factor: float | None = None
    time_preference: tuple = DEFAULT_TIME_PREFERENCE
    ifos: tuple = ("H1", "L1")

    def __post_init__(self):
        self.match = str(self.match).strip().lower()
        if self.match not in MATCH_RULES:
            raise ValueError(f"CombineMatch is one of {MATCH_RULES}, got {self.match!r}")
        if self.window_s is not None:
            self.window_s = _positive("CombineWindow", self.window_s)
        if self.trials_factor is not None:
            self.trials_factor = float(self.trials_factor)
            if not np.isfinite(self.trials_factor) or self.trials_factor < 1.0:
                raise ValueError(f"NetworkTrialsFactor counts the paths a rate is read from and must be "
                                 f"at least 1, got {self.trials_factor!r}")
        self.time_preference = tuple(str(p) for p in self.time_preference)
        self.ifos = tuple(str(i) for i in self.ifos)

    @classmethod
    def from_parameters(cls, par, ifos=None):
        """The settings a run asks for (every key optional)."""

        def get(key, default):
            value = getattr(par, key, None)
            return default if value is None else value

        pref = get("CombineTimePreference", DEFAULT_TIME_PREFERENCE)
        if isinstance(pref, str):
            pref = tuple(p.strip() for p in pref.split(",") if p.strip())
        kwargs = dict(window_s=get("CombineWindow", None), match=get("CombineMatch", DEFAULT_MATCH),
                      trials_factor=get("NetworkTrialsFactor", None), time_preference=tuple(pref))
        detectors = ifos if ifos is not None else get("Detectors", None)
        if detectors is not None:
            kwargs["ifos"] = tuple(str(i) for i in detectors)
        return cls(**kwargs)

    def window(self) -> float:
        """The matching window in seconds: the stated one, else the pair's coincidence window."""
        if self.window_s is not None:
            return float(self.window_s)
        return float(CoincidenceConfig().maximum_tolerance(self.ifos))

    def trials(self, n_paths) -> float:
        """The trials factor: the stated one, else the number of paths."""
        return float(n_paths) if self.trials_factor is None else float(self.trials_factor)


def combined_far(rates, trials_factor):
    """The combined rate: the smallest finite rate of the paths times the trials factor.

    :param rates: array (n, n_paths) of the rates per path, NaN where a path did not find the row.
    :param trials_factor: the multiplier.
    :return: numpy.ndarray (n,) of rates; NaN where no path has a rate.
    """
    r = np.asarray(rates, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    if not r.shape[1] or not len(r):
        return np.full(len(r), np.nan)
    best = np.min(np.where(np.isfinite(r), r, np.inf), axis=1)
    best = np.where(np.isfinite(best), best, np.nan)
    return best * float(trials_factor)


def _distance(a, b, ia, ib, settings):
    """Distance of candidate rows ``a[ia]`` and ``b[ib]`` under the match rule; inf when the rule cannot be read."""
    rule = settings.match
    if rule == "candidate" or (rule == "instant" and not all(f"instant_{i}" in a and f"instant_{i}" in b
                                                                 for i in settings.ifos)):
        return np.abs(a["gps_candidate"].to_numpy(float)[ia] - b["gps_candidate"].to_numpy(float)[ib])
    if rule == "instant":
        d = np.zeros(len(ia))
        for i in settings.ifos:
            d = np.maximum(d, np.abs(a[f"instant_{i}"].to_numpy(float)[ia] - b[f"instant_{i}"].to_numpy(float)[ib]))
        return d
    # extent: the padded extents of every detector overlap; the distance is the gap between the instants or
    # candidate times (ordering only)
    w = settings.window()
    ok = np.ones(len(ia), bool)
    for i in settings.ifos:
        if not all(c in t for t in (a, b) for c in (f"gpsStart_{i}", f"duration_{i}")):
            return np.abs(a["gps_candidate"].to_numpy(float)[ia] - b["gps_candidate"].to_numpy(float)[ib])
        sa, sb = a[f"gpsStart_{i}"].to_numpy(float)[ia], b[f"gpsStart_{i}"].to_numpy(float)[ib]
        ea, eb = sa + a[f"duration_{i}"].to_numpy(float)[ia], sb + b[f"duration_{i}"].to_numpy(float)[ib]
        ok &= (ea + w >= sb) & (eb + w >= sa)
    gap = np.abs(a["gps_candidate"].to_numpy(float)[ia] - b["gps_candidate"].to_numpy(float)[ib])
    return np.where(ok, gap, np.inf)


def match_candidates(a, b, settings):
    """One-to-one matching of two candidate tables, closest first.

    :param a: candidates of one path (``gps_candidate``, and the columns the rule reads).
    :param b: candidates of the other path.
    :param settings: `CombineSettings`.
    :return: (ia, ib, distance) -- integer positions of the matched rows of `a` and `b` and their distance.
    """
    ia = np.zeros(0, int)
    ib = np.zeros(0, int)
    if not len(a) or not len(b):
        return ia, ib, np.zeros(0)
    w = settings.window()
    ga, gb = a["gps_candidate"].to_numpy(float), b["gps_candidate"].to_numpy(float)
    # the candidate times of the same event differ by at most the window plus the extents under "extent": take a
    # wide reach from the candidate times, then the rule decides
    reach = w
    if settings.match == "extent":
        for i in settings.ifos:
            for t in (a, b):
                if f"duration_{i}" in t and len(t):
                    reach = max(reach, w + float(np.nanmax(t[f"duration_{i}"].to_numpy(float))))
    order = np.argsort(gb)
    gs = gb[order]
    ca, cb = [], []
    for k, t in enumerate(ga):
        lo, hi = np.searchsorted(gs, t - reach), np.searchsorted(gs, t + reach, side="right")
        if hi > lo:
            ca.extend([k] * (hi - lo))
            cb.extend(order[lo:hi].tolist())
    if not ca:
        return ia, ib, np.zeros(0)
    ca, cb = np.asarray(ca, int), np.asarray(cb, int)
    d = _distance(a, b, ca, cb, settings)
    keep = np.isfinite(d) & ((d <= w) if settings.match != "extent" else np.ones(len(d), bool))
    ca, cb, d = ca[keep], cb[keep], d[keep]
    order = np.argsort(d, kind="stable")
    used_a, used_b = set(), set()
    out_a, out_b, out_d = [], [], []
    for k in order:
        if ca[k] in used_a or cb[k] in used_b:
            continue
        used_a.add(ca[k])
        used_b.add(cb[k])
        out_a.append(ca[k])
        out_b.append(cb[k])
        out_d.append(d[k])
    return np.asarray(out_a, int), np.asarray(out_b, int), np.asarray(out_d, float)


def combine_candidates(candidates, settings=None, rank_columns=None):
    """One list from the candidate tables of several paths.

    :param candidates: dict path name -> candidate table (``gps_candidate``, ``far_per_day``; the per-detector and
        rank columns when it has them; an empty table, with or without columns, is a path with no candidate). Tables are taken in the dict's order; the first is matched to the
        others in turn.
    :param settings: `CombineSettings` (default: the defaults).
    :param rank_columns: dict path name -> the column of its detection statistic, copied to
        ``statistic_<path>``; a path missing from it has NaN there.
    :return: pandas.DataFrame -- one row per distinct candidate: ``found_by`` (the path names joined by "+"),
        ``n_paths``, the time and per-detector columns of the preferred path, ``time_path`` (which one),
        per path ``candidate_id_<p>``, ``gps_candidate_<p>``, ``far_per_day_<p>``, ``n_background_ge_<p>``,
        ``rank_<p>``, ``statistic_<p>``, ``instant_<ifo>_<p>``; ``combine_dt_s`` (the matching distance, NaN
        when one path); ``far_per_day`` (combined), ``far_path`` (the path of the best rate),
        ``trials_factor``, ``combined_statistic``; sorted by ``gps_candidate``.
    """
    settings = CombineSettings() if settings is None else settings
    rank_columns = dict(rank_columns or {})
    names = list(candidates)
    if not names:
        raise ValueError("no candidate table to combine")
    candidates = dict(candidates)
    for name, table in candidates.items():
        if table is None or not len(table):
            # a path with no candidate at all (e.g. no admitted pair on the stretch) may carry no columns
            candidates[name] = pd.DataFrame({"gps_candidate": np.zeros(0), "far_per_day": np.zeros(0)})
            continue
        for col in ("gps_candidate", "far_per_day"):
            if col not in table:
                raise ValueError(f"the candidates of path {name!r} carry no {col!r}")
    # groups: one per distinct candidate, {path: row position}
    base = names[0]
    groups = [{base: k} for k in range(len(candidates[base]))]
    owner = {base: {k: k for k in range(len(candidates[base]))}}
    for name in names[1:]:
        owner[name] = {}
        table = candidates[name]
        taken = set()
        # match against every earlier path, closest first across them
        matches = []
        for prev in names[:names.index(name)]:
            ia, ib, d = match_candidates(candidates[prev], table, settings)
            for x, y, dist in zip(ia, ib, d):
                matches.append((dist, prev, int(x), int(y)))
        for dist, prev, x, y in sorted(matches, key=lambda m: m[0]):
            g = owner[prev][x]
            if y in taken or name in groups[g]:
                continue
            groups[g][name] = y
            owner[name][y] = g
            taken.add(y)
            groups[g]["_dt"] = min(float(dist), groups[g].get("_dt", np.inf))
        for y in range(len(table)):
            if y not in taken:
                groups.append({name: y})
                owner[name][y] = len(groups) - 1
    n = len(groups)
    trials = settings.trials(len(names))
    rows = {}
    pref = [p for p in settings.time_preference if p in names] + [p for p in names if p not in settings.time_preference]
    time_path = np.array([next(p for p in pref if p in g) for g in groups], dtype=object)
    rows["found_by"] = np.array(["+".join(p for p in names if p in g) for g in groups], dtype=object)
    rows["n_paths"] = np.array([sum(p in g for p in names) for g in groups], int)
    rows["time_path"] = time_path
    rows["combine_dt_s"] = np.array([g.get("_dt", np.nan) for g in groups], float)

    def take(name, col):
        """The column of path `name` at the groups' rows, NaN (or None) where the path is absent."""
        table = candidates[name]
        if col not in table:
            return None
        src = table[col].to_numpy()
        idx = np.array([g.get(name, -1) for g in groups], int)
        if np.issubdtype(src.dtype, np.number) or np.issubdtype(src.dtype, np.bool_):
            out = np.full(n, np.nan)
            out[idx >= 0] = src[idx[idx >= 0]].astype(float)
            return out
        out = np.full(n, None, dtype=object)
        out[idx >= 0] = src[idx[idx >= 0]]
        return out

    def preferred(col):
        out = None
        for name in pref:
            v = take(name, col)
            if v is None:
                continue
            sel = time_path == name
            if out is None:
                out = np.full(n, np.nan) if np.issubdtype(np.asarray(v).dtype, np.number) else np.full(n, None, dtype=object)
            out[sel] = v[sel]
        return out

    for col in ROW_COLUMNS:
        v = preferred(col)
        if v is not None:
            rows[col] = v
    for i in settings.ifos:
        for col in PER_IFO_COLUMNS:
            v = preferred(f"{col}_{i}")
            if v is not None:
                rows[f"{col}_{i}"] = v
    rates = []
    for name in names:
        for col in PATH_COLUMNS:
            v = take(name, col)
            if v is not None:
                rows[f"{col}_{name}"] = v
        stat = rank_columns.get(name)
        v = take(name, stat) if stat is not None else None
        rows[f"statistic_{name}"] = np.full(n, np.nan) if v is None else v
        for i in settings.ifos:
            v = take(name, f"instant_{i}")
            if v is not None:
                rows[f"instant_{i}_{name}"] = v
        rates.append(rows[f"far_per_day_{name}"])
    rates = np.column_stack(rates)
    rows["far_per_day"] = combined_far(rates, trials)
    with np.errstate(invalid="ignore"):
        best = np.where(np.isfinite(rates), rates, np.inf)
        k = np.argmin(best, axis=1)
    rows["far_path"] = np.array([names[j] if np.isfinite(rates[r, j]) else None for r, j in enumerate(k)], dtype=object)
    rows["trials_factor"] = np.full(n, trials)
    with np.errstate(divide="ignore", invalid="ignore"):
        rows["combined_statistic"] = -np.log10(rows["far_per_day"])
    out = pd.DataFrame(rows)
    out = out.sort_values("gps_candidate", kind="stable").reset_index(drop=True)
    out.attrs.update(paths=names, trials_factor=trials, window_s=settings.window(), match=settings.match)
    return out


def combine_paths(candidates_w, candidates_t, settings=None, rank_w=None, rank_t=None):
    """The combined list of the wavegram path ("w") and the track path ("t") of one stage 1.

    :param candidates_w: the wavegram path's zero-lag candidates with ``far_per_day``.
    :param candidates_t: the track path's zero-lag candidates with ``far_per_day``.
    :param settings: `CombineSettings` (default: the defaults -- the coincidence window on the instants, the
        wavegram time first, trials factor 2).
    :param rank_w: the wavegram path's detection statistic column (e.g. ``gnn_logit``, ``network_morphology``).
    :param rank_t: the track path's (e.g. ``track_statistic``).
    :return: pandas.DataFrame, see `combine_candidates`.
    """
    return combine_candidates({"w": candidates_w, "t": candidates_t}, settings,
                              rank_columns={"w": rank_w, "t": rank_t})
