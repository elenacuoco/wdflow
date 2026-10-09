"""The candidates a search would send to GraceDB, one row each, written to disk.

A network run ends with a ranked candidate table and a false-alarm rate per
candidate. What an alert system wants is the subset below a rate, described in
the fields a burst event carries there (the common ones, ``gpstime``, ``far``,
``instruments``, ``group``, ``pipeline``, ``search``, and the per-pipeline
extra attributes a ``MultiBurst`` or ``MLyBurst`` model holds: central
frequency, bandwidth, duration, per-detector times, a detection statistic, an
hrss), plus what lets the row be traced back to the run that produced it. This
module makes that table (`alert_table`) and writes it as a CSV
(`write_alerts`); `run_network` does both when `AlertCSV` names a file.

The selection is on the false-alarm rate the candidate table carries
(``far_hz``, or ``far_per_day``, or ``n_background_ge`` over a livetime): a
candidate is kept when its rate is at or below `AlertFARThreshold`, and it is
flagged ``significant`` when the rate times `AlertTrialsFactor` is below
`AlertSignificantFAR`. Nothing here recomputes a rate; a table without one
yields no alert.

The sky is what the timing gives: with two detectors the arrival-time
difference fixes a ring on the sky (`wdf.analysis.skymap.localise`), and the
row records the difference used, its uncertainty, the area of the 50 % and
90 % regions and the position of the largest weight. The map itself is
written only when `AlertSkyMapDir` names a directory.

Columns absent from the inputs (no events given, no hrss estimated, no
cross-correlation lag) are left empty, never invented.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wdf.analysis.detectors import light_travel_time

#: Seconds in a Julian year.
SECONDS_PER_YEAR = 365.25 * 86400.0
#: Default `AlertFARThreshold`: one accidental per day, Hz.
DEFAULT_FAR_THRESHOLD_HZ = 1.0 / 86400.0
#: Default `AlertTrialsFactor`: the rate as measured.
DEFAULT_TRIALS_FACTOR = 1.0
#: Default `AlertSignificantFAR`: one per year, Hz, on the trials-corrected rate.
DEFAULT_SIGNIFICANT_FAR_HZ = 1.0 / SECONDS_PER_YEAR
#: Default `AlertPipeline`, `AlertGroup`, `SearchName`.
DEFAULT_PIPELINE, DEFAULT_GROUP, DEFAULT_SEARCH = "WDF", "Burst", "AllSky"
#: Default `AlertSkyResolution`: the sky grid's cell, degrees.
DEFAULT_SKY_RESOLUTION_DEG = 1.0
#: Default `AlertSkySigmaFloor`: "sample", one sample of the stream.
DEFAULT_SKY_SIGMA_FLOOR = "sample"
#: Default `AlertSkyTiming`: the reconstructions' cross-correlation lag when
#: the table carries one (`network_lag_s`), else the event instants.
DEFAULT_SKY_TIMING = "xcorr"

#: Event columns joined to the row of each detector, when events are given.
EVENT_COLUMNS = ("EnWDF", "EnWDF_window", "snrPeak", "freqMin", "freqMean", "freqMax",
                 "freqQ05", "freqQ95", "duration90", "gpsCentroid", "significance", "n_pixels")

#: The columns of every alert row, in order, before the per-detector and
#: per-pair ones (`per_ifo_columns`, `per_pair_columns`) and the provenance.
ALERT_COLUMNS = (
    "pipeline", "group", "search", "run", "candidate_id", "stretch", "rank", "ifos", "n_ifos",
    "gpstime", "event_instant", "start_time", "end_time", "duration", "central_freq",
    "bandwidth",
    "far_hz", "far_per_day", "trials_factor", "far_trials_hz", "significant",
    "far_threshold_hz", "significant_far_hz", "n_background_ge", "background_livetime_s",
    "detection_statistic_name", "detection_statistic", "network_enwdf", "network_min_enwdf",
    "network_morphology", "network_correlation", "tile_coherence", "gnn_logit", "gnn_score",
    "track_statistic", "track_coherence", "glitch",
    "hrss_network_det", "hrss_err_network_det",
    "dt_xcorr_s", "dt_xcorr_width_s", "dt_xcorr_window_s",
    "sky_timing", "sky_dt_s", "sky_sigma_s", "sky_area50_deg2", "sky_area90_deg2",
    "sky_ra_max_deg", "sky_dec_max_deg", "sky_map_path",
)
#: Per-detector columns, ``<name>_<ifo>``.
PER_IFO_COLUMNS = ("gps", "peak_time", "start_time", "duration", "duration90", "central_freq",
                   "bandwidth", "freq_low", "freq_high", "EnWDF", "snrPeak", "hrss", "hrss_err",
                   "hrss_noise", "glitch")
#: Per-pair columns, ``<name>_<a><b>`` (``dt`` is the first detector's instant
#: minus the second's).
PER_PAIR_COLUMNS = ("dt_envelope_s", "light_travel_s", "dt_physical")
#: Provenance columns.
PROVENANCE_COLUMNS = ("config_path", "config_sha256", "model_path", "model_sha256",
                      "wdflow_commit", "wdflow_version", "data_start", "data_end", "created_utc")


def _flag(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _rate(key, value, default):
    if value is None:
        return float(default)
    rate = float(value)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"{key} is a rate in Hz and must be positive, got {value!r}")
    return rate


@dataclass
class AlertSettings:
    """The alert table's choices, read from the run parameters.

    :param far_threshold_hz: `AlertFARThreshold`, Hz; a candidate is an alert
        at or below it (default one per day).
    :param trials_factor: `AlertTrialsFactor`, multiplies the rate before the
        significance test (default 1).
    :param significant_far_hz: `AlertSignificantFAR`, Hz; ``significant`` when
        the trials-corrected rate is below it (default one per year).
    :param csv: `AlertCSV`, the file `run_network` writes, or None (default:
        nothing written). ``{start}``, ``{end}`` (the searched span's GPS,
        whole seconds) and ``{run}`` are filled in.
    :param run: `AlertRunName`, the ``run`` column (default "").
    :param pipeline: `AlertPipeline` (default "WDF").
    :param group: `AlertGroup` (default "Burst").
    :param search: `SearchName` (default "AllSky").
    :param sky: `AlertSkyMap` (default true): localise each alert from its
        timing.
    :param sky_timing: `AlertSkyTiming`, "xcorr" (default; the
        cross-correlation lag of the reconstructions, falling back to the
        instants when the table has none) or "envelope" (the event instants).
    :param sky_sigma_floor: `AlertSkySigmaFloor`, "sample" (default: one
        sample of the stream, 1 / fs) or seconds; the smallest timing
        uncertainty a pair is given.
    :param sky_resolution_deg: `AlertSkyResolution`, the grid cell in degrees
        (default 1).
    :param sky_map_dir: `AlertSkyMapDir`, where each alert's map is written as
        ``<run>-<candidate_id>.npz`` (``ra``, ``dec``, ``weight``), or None
        (default: no map file).
    """

    far_threshold_hz: float = DEFAULT_FAR_THRESHOLD_HZ
    trials_factor: float = DEFAULT_TRIALS_FACTOR
    significant_far_hz: float = DEFAULT_SIGNIFICANT_FAR_HZ
    csv: str = None
    run: str = ""
    pipeline: str = DEFAULT_PIPELINE
    group: str = DEFAULT_GROUP
    search: str = DEFAULT_SEARCH
    sky: bool = True
    sky_timing: str = DEFAULT_SKY_TIMING
    sky_sigma_floor: object = DEFAULT_SKY_SIGMA_FLOOR
    sky_resolution_deg: float = DEFAULT_SKY_RESOLUTION_DEG
    sky_map_dir: str = None

    @classmethod
    def from_parameters(cls, par):
        """The settings a run asks for."""

        def get(key, default):
            value = getattr(par, key, None)
            return default if value is None else value

        trials = float(get("AlertTrialsFactor", DEFAULT_TRIALS_FACTOR))
        if not np.isfinite(trials) or trials < 1.0:
            raise ValueError(f"AlertTrialsFactor counts the searches a rate is shared with and "
                             f"must be at least 1, got {trials!r}")
        timing = str(get("AlertSkyTiming", DEFAULT_SKY_TIMING)).strip().lower()
        if timing not in ("xcorr", "envelope"):
            raise ValueError(f"AlertSkyTiming is 'xcorr' or 'envelope', got {timing!r}")
        floor = get("AlertSkySigmaFloor", DEFAULT_SKY_SIGMA_FLOOR)
        if isinstance(floor, str):
            if floor.strip().lower() != "sample":
                raise ValueError(f"AlertSkySigmaFloor is 'sample' or seconds, got {floor!r}")
            floor = "sample"
        else:
            floor = float(floor)
            if not np.isfinite(floor) or floor <= 0.0:
                raise ValueError(f"AlertSkySigmaFloor must be positive seconds, got {floor!r}")
        resolution = float(get("AlertSkyResolution", DEFAULT_SKY_RESOLUTION_DEG))
        if not np.isfinite(resolution) or resolution <= 0.0 or resolution > 90.0:
            raise ValueError(f"AlertSkyResolution is a cell in degrees, got {resolution!r}")
        csv = get("AlertCSV", None)
        sky_dir = get("AlertSkyMapDir", None)
        return cls(far_threshold_hz=_rate("AlertFARThreshold", get("AlertFARThreshold", None),
                                          DEFAULT_FAR_THRESHOLD_HZ),
                   trials_factor=trials,
                   significant_far_hz=_rate("AlertSignificantFAR",
                                            get("AlertSignificantFAR", None),
                                            DEFAULT_SIGNIFICANT_FAR_HZ),
                   csv=None if csv is None else str(csv),
                   run=str(get("AlertRunName", "")),
                   pipeline=str(get("AlertPipeline", DEFAULT_PIPELINE)),
                   group=str(get("AlertGroup", DEFAULT_GROUP)),
                   search=str(get("SearchName", DEFAULT_SEARCH)),
                   sky=_flag(get("AlertSkyMap", None), True),
                   sky_timing=timing, sky_sigma_floor=floor, sky_resolution_deg=resolution,
                   sky_map_dir=None if sky_dir is None else str(sky_dir))

    def sigma_floor_s(self, fs=None) -> float:
        """The timing floor in seconds; "sample" needs the stream rate `fs`."""
        if self.sky_sigma_floor == "sample":
            if fs is None or not np.isfinite(float(fs)) or float(fs) <= 0.0:
                raise ValueError("AlertSkySigmaFloor 'sample' is one sample of the stream: give "
                                 "fs (the events carry it), or set the floor in seconds")
            return 1.0 / float(fs)
        return float(self.sky_sigma_floor)

    @property
    def sky_grid(self):
        """``(n_ra, n_dec)`` of the sky grid at `sky_resolution_deg`."""
        n_ra = max(int(round(360.0 / self.sky_resolution_deg)), 4)
        return n_ra, max(n_ra // 2, 2)


def _sha256(path):
    if not path or not os.path.isfile(str(path)):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _commit():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", here],
                               capture_output=True, text=True, check=True,
                               timeout=10).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return out + ("-dirty" if dirty else "")


def provenance(par=None, data_start=None, data_end=None, model_path=None, config_path=None):
    """Where an alert came from: configuration, model, library, data stretch.

    :param par: the run configuration; `ConfigFile` (set by `load_config`) is
        the configuration's path and `GNNModel` the scorer's, unless given.
    :param data_start: GPS start of the data the candidates were searched in.
    :param data_end: its GPS end.
    :param model_path: the scorer's file, when not read from `par`.
    :param config_path: the configuration's file, when not read from `par`.
    :return: dict -- `PROVENANCE_COLUMNS` to their values; a file that does
        not exist has no hash (None).
    """
    try:
        from importlib.metadata import version as _version
        version = _version("wdflow")
    except Exception:   # not installed as a distribution
        version = None

    config = config_path if config_path is not None else getattr(par, "ConfigFile", None)
    model = model_path if model_path is not None else getattr(par, "GNNModel", None)
    return {"config_path": None if config is None else str(config),
            "config_sha256": _sha256(config),
            "model_path": None if model is None else str(model),
            "model_sha256": _sha256(model),
            "wdflow_commit": _commit(), "wdflow_version": version,
            "data_start": None if data_start is None else float(data_start),
            "data_end": None if data_end is None else float(data_end),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def far_hz_of(candidates, livetime_s=None) -> np.ndarray:
    """Each candidate's false-alarm rate in Hz, as the table states it.

    Read from ``far_hz``; else ``far_per_day`` / 86400; else
    ``(n_background_ge + 1) / livetime`` with the livetime from
    ``background_livetime_s`` or `livetime_s` (the `TimeSlideFAR` rule).

    :type candidates: pandas.DataFrame
    :param candidates: the network candidates.
    :param livetime_s: the accidental livetime, seconds, when the table holds
        only the counts.
    :return: numpy.ndarray -- Hz.
    :raises ValueError: if the table states no rate.
    """
    if "far_hz" in candidates:
        return pd.to_numeric(candidates["far_hz"], errors="coerce").to_numpy(dtype=float)
    if "far_per_day" in candidates:
        return pd.to_numeric(candidates["far_per_day"],
                             errors="coerce").to_numpy(dtype=float) / 86400.0
    if "n_background_ge" in candidates:
        if "background_livetime_s" in candidates:
            live = pd.to_numeric(candidates["background_livetime_s"]).to_numpy(dtype=float)
        elif livetime_s is not None and float(livetime_s) > 0.0:
            live = float(livetime_s)
        else:
            raise ValueError("the candidates carry accidental counts (n_background_ge) but no "
                             "livetime: give livetime_s")
        n_ge = pd.to_numeric(candidates["n_background_ge"]).to_numpy(dtype=float)
        return (n_ge + 1.0) / live
    raise ValueError("the candidates carry no false-alarm rate (far_hz, far_per_day or "
                     "n_background_ge): without one nothing is an alert")


def _ifos_of(row):
    return [i.strip() for i in str(row["ifos_involved"]).split(",") if i.strip()]


def _events_table(events):
    """One table of every detector's events with an ``ifo`` column, or None."""
    if events is None:
        return None
    if isinstance(events, dict):
        parts = [ev.assign(ifo=ifo) for ifo, ev in events.items() if ev is not None and len(ev)]
        if not parts:
            return None
        table = pd.concat(parts, ignore_index=True)
    else:
        table = events
    if "ifo" not in table or "cluster_id" not in table:
        raise ValueError("events need 'ifo' and 'cluster_id' columns to be joined to the pairs")
    return table.set_index(["ifo", pd.to_numeric(table["cluster_id"]).astype(int)]).sort_index()


def _event_value(table, ifo, cluster, column):
    if table is None or column not in table:
        return np.nan
    try:
        value = table.loc[(ifo, int(cluster)), column]
    except (KeyError, ValueError, TypeError):
        return np.nan
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return value


def _bool(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    return bool(value)


def _num(row, column):
    value = row.get(column, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _sky(arrivals, spreads, settings, path=None):
    """Area and peak of the ring the timing fixes, and the map file if asked."""
    from wdf.analysis.skymap import credible_area, localise

    n_ra, n_dec = settings.sky_grid
    ra, dec, weight = localise(arrivals, spreads, n_ra=n_ra, n_dec=n_dec)
    k = int(np.argmax(weight))
    out = {"sky_area50_deg2": credible_area(weight, 0.5),
           "sky_area90_deg2": credible_area(weight, 0.9),
           "sky_ra_max_deg": float(np.degrees(ra.reshape(-1)[k])),
           "sky_dec_max_deg": float(np.degrees(dec.reshape(-1)[k])),
           "sky_map_path": None}
    if path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, ra=ra, dec=dec, weight=weight,
                            ifos=np.asarray(sorted(arrivals)),
                            arrivals=np.asarray([arrivals[i] for i in sorted(arrivals)]),
                            spreads=np.asarray([spreads[i] for i in sorted(arrivals)]))
        out["sky_map_path"] = os.path.abspath(path)
    return out


def _row(cand, far_hz, settings, events, statistic, ifos_all, fs, prov):
    ifos = _ifos_of(cand)
    row = {"pipeline": settings.pipeline, "group": settings.group, "search": settings.search,
           "run": settings.run, "candidate_id": cand.get("candidate_id"),
           "stretch": cand.get("stretch"), "rank": cand.get("rank"), "ifos": ",".join(ifos),
           "n_ifos": len(ifos), "gpstime": _num(cand, "gps_candidate"),
           "event_instant": cand.get("event_instant")}
    # The side-indexed columns (cluster_i / glitch_i) read per detector.
    side = {}
    for s in ("i", "j"):
        ifo = cand.get(f"ifo_{s}")
        if ifo is not None and not (isinstance(ifo, float) and np.isnan(ifo)):
            side[str(ifo)] = s
    starts, ends, durations, centres, weights, lows, highs = [], [], [], [], [], [], []
    for ifo in ifos:
        s = side.get(ifo)
        cluster = cand.get(f"cluster_{s}") if s else None
        get = (lambda col: _event_value(events, ifo, cluster, col)) if cluster is not None \
            else (lambda col: np.nan)
        start = _num(cand, f"gpsStart_{ifo}")
        duration = _num(cand, f"duration_{ifo}")
        duration90 = get("duration90")
        f_mean, q05, q95 = get("freqMean"), get("freqQ05"), get("freqQ95")
        f_min, f_max = get("freqMin"), get("freqMax")
        enwdf = get("EnWDF")
        row.update({f"gps_{ifo}": _num(cand, f"instant_{ifo}"),
                    f"peak_time_{ifo}": _num(cand, f"gpsPeak_{ifo}"),
                    f"start_time_{ifo}": start, f"duration_{ifo}": duration,
                    f"duration90_{ifo}": duration90, f"central_freq_{ifo}": f_mean,
                    f"bandwidth_{ifo}": q95 - q05, f"freq_low_{ifo}": f_min,
                    f"freq_high_{ifo}": f_max, f"EnWDF_{ifo}": enwdf,
                    f"snrPeak_{ifo}": get("snrPeak"),
                    f"hrss_{ifo}": _num(cand, f"hrss_{ifo}"),
                    f"hrss_err_{ifo}": _num(cand, f"hrss_err_{ifo}"),
                    f"hrss_noise_{ifo}": _num(cand, f"hrss_noise_{ifo}"),
                    f"glitch_{ifo}": _bool(cand.get(f"glitch_{s}")) if s else None})
        starts.append(start)
        ends.append(start + duration)
        durations.append(duration90 if np.isfinite(duration90) else duration)
        centres.append(f_mean)
        weights.append(enwdf if np.isfinite(enwdf) and enwdf > 0 else 1.0)
        lows.append(q05 if np.isfinite(q05) else f_min)
        highs.append(q95 if np.isfinite(q95) else f_max)

    def agg(values, how):
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        return how(v) if v.size else np.nan

    centre = np.asarray(centres, dtype=float)
    ok = np.isfinite(centre)
    row.update({"start_time": agg(starts, np.min), "end_time": agg(ends, np.max),
                "duration": agg(durations, np.max),
                "central_freq": float(np.average(centre[ok], weights=np.asarray(weights)[ok]))
                if ok.any() else np.nan,
                "bandwidth": agg(highs, np.max) - agg(lows, np.min)})

    trials = settings.trials_factor * far_hz
    row.update({"far_hz": far_hz, "far_per_day": far_hz * 86400.0,
                "trials_factor": settings.trials_factor, "far_trials_hz": trials,
                "significant": bool(trials < settings.significant_far_hz),
                "far_threshold_hz": settings.far_threshold_hz,
                "significant_far_hz": settings.significant_far_hz,
                "n_background_ge": cand.get("n_background_ge"),
                "background_livetime_s": cand.get("background_livetime_s"),
                "detection_statistic_name": statistic,
                "detection_statistic": _num(cand, statistic) if statistic else np.nan})
    for col in ("network_enwdf", "network_min_enwdf", "network_morphology",
                "network_correlation", "tile_coherence", "gnn_logit", "gnn_score",
                "track_statistic", "track_coherence", "hrss_network_det"):
        row[col] = _num(cand, col)
    row["glitch"] = _bool(cand.get("glitch_single_ifo"))
    # Error of the root-sum-square, propagated from the detectors' errors.
    net = row["hrss_network_det"]
    if np.isfinite(net) and net > 0:
        terms = [(row[f"hrss_{i}"] * row[f"hrss_err_{i}"]) ** 2 for i in ifos
                 if np.isfinite(row[f"hrss_{i}"]) and np.isfinite(row[f"hrss_err_{i}"])]
        row["hrss_err_network_det"] = float(np.sqrt(sum(terms)) / net) if terms else np.nan
    else:
        row["hrss_err_network_det"] = np.nan

    # Timing: instants per pair, and the reconstructions' lag of the pair.
    row.update({"dt_xcorr_s": _num(cand, "network_lag_s"),
                "dt_xcorr_width_s": _num(cand, "network_lag_width_s"),
                "dt_xcorr_window_s": _num(cand, "network_lag_window_s")})
    for a_idx, a in enumerate(ifos):
        for b in ifos[a_idx + 1:]:
            tag = f"{a}{b}"
            dt = row[f"gps_{a}"] - row[f"gps_{b}"]
            try:
                travel = light_travel_time(a, b)
            except KeyError:
                travel = np.nan
            row[f"dt_envelope_s_{tag}"] = dt
            row[f"light_travel_s_{tag}"] = travel
            row[f"dt_physical_{tag}"] = bool(abs(dt) <= travel) if np.isfinite(dt) \
                and np.isfinite(travel) else None

    row.update({"sky_timing": None, "sky_dt_s": np.nan, "sky_sigma_s": np.nan,
                "sky_area50_deg2": np.nan, "sky_area90_deg2": np.nan,
                "sky_ra_max_deg": np.nan, "sky_dec_max_deg": np.nan, "sky_map_path": None})
    if settings.sky and len(ifos) >= 2:
        floor = settings.sigma_floor_s(fs)
        xcorr = settings.sky_timing == "xcorr" and len(ifos) == 2 \
            and np.isfinite(row["dt_xcorr_s"])
        arrivals, spreads = {}, {}
        if xcorr:
            width = row["dt_xcorr_width_s"]
            sigma = max(width if np.isfinite(width) else 0.0, floor)
            t0 = row[f"gps_{ifos[0]}"]
            arrivals = {ifos[0]: t0, ifos[1]: t0 - row["dt_xcorr_s"]}
            spreads = {i: sigma / np.sqrt(2.0) for i in ifos}
            row.update(sky_timing="xcorr", sky_dt_s=row["dt_xcorr_s"], sky_sigma_s=sigma)
        else:
            arrivals = {i: row[f"gps_{i}"] for i in ifos if np.isfinite(row[f"gps_{i}"])}
            spreads = {i: floor / np.sqrt(2.0) for i in arrivals}
            if len(arrivals) >= 2:
                row.update(sky_timing="envelope", sky_sigma_s=floor,
                           sky_dt_s=row.get(f"dt_envelope_s_{ifos[0]}{ifos[1]}", np.nan))
        if len(arrivals) >= 2:
            path = None
            if settings.sky_map_dir:
                name = f"{settings.run or 'alert'}-{row['candidate_id']}.npz"
                path = os.path.join(settings.sky_map_dir, name)
            try:
                row.update(_sky(arrivals, spreads, settings, path))
            except KeyError:
                pass    # a detector whose position the library does not know
    row.update(prov)
    return row


def per_ifo_columns(ifos):
    """The per-detector columns for `ifos`, in order."""
    return [f"{name}_{ifo}" for ifo in ifos for name in PER_IFO_COLUMNS]


def per_pair_columns(ifos):
    """The per-pair columns for `ifos`, in order."""
    ifos = list(ifos)
    return [f"{name}_{a}{b}" for k, a in enumerate(ifos) for b in ifos[k + 1:]
            for name in PER_PAIR_COLUMNS]


def alert_table(candidates, events=None, par=None, settings=None, livetime_s=None,
                statistic=None, fs=None, provenance_fields=None) -> pd.DataFrame:
    """The alerts a candidate table yields: one row per candidate at or below
    the rate threshold, in the GraceDB-like fields.

    :type candidates: pandas.DataFrame
    :param candidates: the network candidates (`run_network`), with a
        false-alarm rate (`far_hz_of`).
    :param events: ``{ifo: event table}`` of stage 2 (`NetworkResult.events`),
        or one table with an ``ifo`` column; joined on ``cluster_i`` /
        ``cluster_j`` for the morphology (`EVENT_COLUMNS`). None leaves those
        columns empty.
    :param par: the run configuration, read for `AlertSettings` when
        `settings` is None, and for the provenance.
    :param settings: `AlertSettings`, else from `par`.
    :param livetime_s: accidental livetime, seconds, when the table carries
        counts and no rate.
    :param statistic: the ranking column (``detection_statistic``); default
        ``network_rank_statistic`` when present (the penalised column the
        rate was measured on), else ``gnn_logit``, ``track_statistic`` or
        ``network_morphology``, whichever the table holds first.
    :param fs: the stream's sampling rate, Hz, for the "sample" timing floor;
        read from the events' ``fs`` when they carry it.
    :param provenance_fields: `provenance` output, else `provenance(par)`.
    :return: pandas.DataFrame -- `ALERT_COLUMNS`, the per-detector and
        per-pair columns of the detectors seen, the `PROVENANCE_COLUMNS`;
        sorted by ``far_hz`` then by the statistic, descending.
    """
    settings = settings or AlertSettings.from_parameters(par)
    table = _events_table(events)
    if fs is None and table is not None and "fs" in table:
        values = pd.to_numeric(table["fs"], errors="coerce").dropna()
        fs = float(values.iloc[0]) if len(values) else None
    if statistic is None:
        for name in ("network_rank_statistic", "gnn_logit", "track_statistic",
                     "network_morphology", "network_enwdf"):
            if name in candidates:
                statistic = name
                break
    prov = provenance_fields if provenance_fields is not None else provenance(par)
    prov = {k: prov.get(k) for k in PROVENANCE_COLUMNS}

    ifos_all = []
    for value in candidates["ifos_involved"] if "ifos_involved" in candidates else []:
        for ifo in str(value).split(","):
            if ifo.strip() and ifo.strip() not in ifos_all:
                ifos_all.append(ifo.strip())
    columns = list(ALERT_COLUMNS) + per_ifo_columns(ifos_all) + per_pair_columns(ifos_all) \
        + list(PROVENANCE_COLUMNS)
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    if "ifos_involved" not in candidates:
        raise ValueError("the candidates carry no 'ifos_involved': not a network table")

    far = far_hz_of(candidates, livetime_s)
    keep = np.isfinite(far) & (far <= settings.far_threshold_hz)
    rows = [_row(cand, float(far[k]), settings, table, statistic, ifos_all, fs, prov)
            for k, (_, cand) in enumerate(candidates.iterrows()) if keep[k]]
    out = pd.DataFrame(rows, columns=columns)
    if len(out):
        by = ["far_hz"] + (["detection_statistic"] if statistic else [])
        out = out.sort_values(by, ascending=[True, False][:len(by)]).reset_index(drop=True)
    return out


def alert_path(pattern, start=None, end=None, run="") -> str:
    """`AlertCSV` with ``{start}``, ``{end}`` (whole GPS seconds) and
    ``{run}`` filled in."""
    start = "" if start is None or not np.isfinite(float(start)) else int(np.floor(float(start)))
    end = "" if end is None or not np.isfinite(float(end)) else int(np.ceil(float(end)))
    return str(pattern).format(start=start, end=end, run=run)


def write_alerts(alerts, path) -> str:
    """Write an alert table as a CSV (header, one row per alert, empty cells
    for what is unknown).

    :type alerts: pandas.DataFrame
    :param alerts: as `alert_table` returns.
    :param path: the file to write; its directory is created.
    :return: str -- the absolute path written.
    """
    path = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    alerts.to_csv(path, index=False)
    return path
