"""Stages 2 to 4 of the WDF workflow, from the detectors' outputs to the
network candidates, on the path the configuration selects.

The workflow:

1. single-window triggers, per detector (`wdfUnitOnline`, `wdfUnitOffline`);
2. single-detector reconstruction (`Reconstruction`);
3. the graph between detectors (`Coincidence`);
4. the ranking of its candidate pairs (`Scorer`).

Two paths, which do not mix (`wdf.config.Parameters.network_path`):

- **"wavegram"**, the paper path and the default (optionally screened by
  the reconstructions' lag, `NetworkLagCut`, `wdf.analysis.pair_screen`):
  `build_detector_graph`,
  `detector_events` and `event_coefficients` per detector (the assembly map,
  and the comparison map at `NetworkComparisonBin` times the network's
  longest light travel time when that is positive), then
  `WavegramCoincidenceFinder` over `IndexedCoincidenceFinder`'s admitted
  pairs and a `TriggerGraphBuilder(ifos)`.
- **"track"**: the detectors' track events (`TrackEvents` written by the
  workers with `Reconstruction` "track", or `reconstruct` on a map), then
  `TrackCoincidenceFinder`, which admits a pair when its tracks are
  compatible and its coherence lag physical. The coherence ranks the pair
  (`track_statistic`) and is not cut on by default (`TrackCoherenceMin`
  "none"); with "calibrate", a threshold is set on the run's own time-slid
  pairs (`calibrate_coherence_min`, step `NetworkSlideStep`). The value used
  is recorded as `TrackCoherenceMinUsed`.

Stage 4 is "classic", the column `NetworkRankColumn` (default
"network_morphology" on the wavegram path, "track_statistic" on the track
path; "network_enwdf" and any other column of the table may be chosen),
or "gnn", a
`GNNCoincidenceScorer` loaded from `GNNModel` (column ``gnn_logit``). With
`NetworkSlides` above zero the time-slide background is drawn
(`TimeSlideFAR`, step `NetworkSlideStep`) and every candidate gets its false
alarm rate; ``rank`` is the candidate's rank within its detector pair by the
ranking column, 1 the loudest.

`run_network` runs one batch (offline, or a stretch of a stream);
`NetworkStream` holds the detectors' outputs as they arrive online and runs it
on what it holds.
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from wdf.analysis.alerts import AlertSettings, alert_path, alert_table, provenance, write_alerts
from wdf.analysis.hrss import HrssSettings, ModelHistory, add_hrss, network_hrss
from wdf.analysis.pair_screen import RANK_STATISTIC, PairScreenSettings, ScreenedFinder
from wdf.config.Parameters import network_path

#: Comparison map bin, in units of the network's longest light travel time.
DEFAULT_COMPARISON_BIN = 2.0
#: Light travel time used for the comparison map of a single detector, s.
SINGLE_DETECTOR_TRAVEL_S = 0.01
#: Default ranking column of the classic scorer on the wavegram path: the
#: coherent energy of the two events' tiles (`network_morphology`, the
#: absolute `tile_coherence`), the paper's deterministic ranking. It asks the
#: two detectors to be loud in the same places of the plane, where
#: `network_enwdf` (the quadrature sum of the two `EnWDF`) is large whenever
#: one detector is loud, a single loud glitch included.
DEFAULT_RANK_COLUMN = "network_morphology"
#: Default ranking column of the track path.
TRACK_RANK_COLUMN = "track_statistic"
#: Default slide step, s, and number of slides (0: no background).
DEFAULT_SLIDE_STEP_S, DEFAULT_SLIDES = 4.0, 0
#: Default memory of `NetworkStream`, s.
DEFAULT_NETWORK_MEMORY_S = 64.0


@dataclass
class NetworkSettings:
    """The network stage's choices, read from the run parameters.

    :param reconstruction: `Reconstruction`, "wavegram" or "track".
    :param coincidence: `Coincidence`, equal to it.
    :param scorer: `Scorer`, "classic" or "gnn".
    :param gnn_model: `GNNModel`, the saved scorer for "gnn".
    :param rank_column: `NetworkRankColumn`, the classic ranking column;
        "network_morphology" on the wavegram path, "track_statistic" on the
        track path, when not given.
    :param comparison_bin: `NetworkComparisonBin`, the comparison map's bin
        over the longest light travel time; 0 for no comparison map.
    :param slides: `NetworkSlides`, time slides of the background (0: none).
    :param slide_step_s: `NetworkSlideStep`, s.
    :param memory_s: `NetworkMemory`, what `NetworkStream` holds, s.
    :param event_instant: `NetworkEventInstant` (wavegram path), the instant
        of an event used for the pair's dt and for `gps_candidate`: "peak"
        (default: `INSTANT_COLUMNS`, i.e. the loudest tile's `gpsPeak` unless
        the events carry `gpsEnvelope`), "centroid" (`gpsCentroid`, the
        EnWDF-weighted centre), "envelope" (`gpsEnvelope`, the peak of the
        analytic envelope of the event's stitched reconstruction,
        `timing.envelope_instants`, searched over `NetworkEnvelopeWidth`
        about `gpsPeak`) or "coarse" (`gpsPeak` as the candidate time; the
        one-to-one assignment does not weigh the instants' difference,
        `CoincidenceConfig.time_weight` = 0: pairs are admitted on the
        events' extents and chosen on their frequency overlap, and the
        lag criterion, if any, is read on the reconstructions).
    :param envelope_width: `NetworkEnvelopeWidth`: "window" (default) for
        each event's analysis window (`n_coeff` / `fs`), or seconds.
    """

    reconstruction: str = "wavegram"
    coincidence: str = "wavegram"
    scorer: str = "classic"
    gnn_model: str = None
    rank_column: str = DEFAULT_RANK_COLUMN
    comparison_bin: float = DEFAULT_COMPARISON_BIN
    slides: int = DEFAULT_SLIDES
    slide_step_s: float = DEFAULT_SLIDE_STEP_S
    memory_s: float = DEFAULT_NETWORK_MEMORY_S
    event_instant: str = "peak"
    envelope_width: object = "window"

    @classmethod
    def from_parameters(cls, par):
        """The settings a run asks for; validated by `network_path`."""
        reconstruction, coincidence, scorer = network_path(par)

        def get(key, default):
            value = getattr(par, key, None)
            return default if value is None else value

        return cls(reconstruction=reconstruction, coincidence=coincidence, scorer=scorer,
                   gnn_model=getattr(par, "GNNModel", None),
                   rank_column=str(get("NetworkRankColumn",
                                       TRACK_RANK_COLUMN if reconstruction == "track"
                                       else DEFAULT_RANK_COLUMN)),
                   comparison_bin=float(get("NetworkComparisonBin", DEFAULT_COMPARISON_BIN)),
                   slides=int(get("NetworkSlides", DEFAULT_SLIDES)),
                   slide_step_s=float(get("NetworkSlideStep", DEFAULT_SLIDE_STEP_S)),
                   memory_s=float(get("NetworkMemory", DEFAULT_NETWORK_MEMORY_S)),
                   event_instant=_event_instant(get("NetworkEventInstant", "peak")),
                   envelope_width=_envelope_width(get("NetworkEnvelopeWidth", "window")))


@dataclass
class NetworkResult:
    """What `run_network` produced.

    :param candidates: the network candidates, ranked (``rank``).
    :param events: ``{ifo: event table}`` of stage 2.
    :param maps: ``{ifo: {cluster_id: EventWavegram}}`` (wavegram path, or
        the track path's maps for the GNN).
    :param comparison: the comparison maps, or None.
    :param waveforms: ``{ifo: {cluster_id: (t0, waveform)}}`` (track path).
    :param graphs: ``{ifo: DetectorGraph}`` (wavegram path).
    :param background: the time-slide background, or None.
    :param finder: the stage-3 finder, for further slides.
    :param settings: the settings used.
    :param alerts_path: the alert CSV written (`AlertCSV`), or None.
    """

    candidates: pd.DataFrame
    events: dict
    maps: dict = field(default_factory=dict)
    comparison: dict = None
    waveforms: dict = field(default_factory=dict)
    graphs: dict = field(default_factory=dict)
    background: pd.DataFrame = None
    finder: object = None
    settings: NetworkSettings = None
    alerts_path: str = None


def longest_travel(ifos):
    """The network's longest light travel time, s (`SINGLE_DETECTOR_TRAVEL_S`
    for one detector)."""
    from wdf.analysis.detectors import light_travel_time

    if len(ifos) < 2:
        return SINGLE_DETECTOR_TRAVEL_S
    return max(light_travel_time(a, b) for a, b in combinations(ifos, 2))


def _reconstructions(graphs, labels):
    """Per detector, a cached ``cluster_id -> (t0, samples)`` of the events'
    stitched reconstructions (`event_waveform`)."""
    from wdf.analysis.detector_graph import event_waveform

    def one(graph, label):
        cache = {}

        def get(cid):
            if cid not in cache:
                cache[cid] = event_waveform(graph, labels=label, cluster_id=int(cid))[int(cid)][:2]
            return cache[cid]
        return get

    return {ifo: one(graphs[ifo], labels[ifo]) for ifo in graphs}


def wavegram_stage(triggers):
    """Stage 2 of the paper path, per detector.

    :param triggers: ``{ifo: trigger table with coefficients}``.
    :return: tuple -- (graphs, labels, events, maps), each ``{ifo: ...}``.
    """
    from wdf.analysis.detector_graph import (build_detector_graph, detector_events,
                                             event_coefficients)

    graphs, labels, events, maps = {}, {}, {}, {}
    for ifo, table in triggers.items():
        g = build_detector_graph(table)
        graphs[ifo], labels[ifo] = g, g.components()
        ev = detector_events(g, labels=labels[ifo])
        ev["ifo"] = ifo
        events[ifo] = ev
        maps[ifo] = event_coefficients(g, labels=labels[ifo])
    return graphs, labels, events, maps


def _bounds(events, triggers, fs_window):
    out = {}
    for ifo, ev in events.items():
        t = triggers[ifo] if triggers is not None and ifo in triggers else ev
        gps = pd.to_numeric(t["gps"]).to_numpy(float)
        out[ifo] = (float(gps.min()), float(gps.max()) + fs_window(t))
    return out


def _window_s(table):
    if "n_coeff" in table and "fs" in table and len(table):
        return float(np.nanmax(pd.to_numeric(table["n_coeff"]) / pd.to_numeric(table["fs"])))
    return 0.0


def _rank(candidates, column):
    if candidates.empty:
        return candidates
    out = candidates.copy()
    if column not in out:
        raise KeyError(f"NetworkRankColumn {column!r} is not a column of the candidate table")
    out["rank"] = out.groupby("ifos_involved")[column].rank(ascending=False,
                                                           method="min").astype(int)
    return out


#: `NetworkEventInstant` choices and the event columns each reads.
EVENT_INSTANTS = {"peak": None, "centroid": ("gpsCentroid",), "envelope": ("gpsEnvelope",),
                  "coarse": ("gpsPeak",)}


def _event_instant(value):
    value = str(value).strip().lower()
    if value not in EVENT_INSTANTS:
        raise ValueError(f"NetworkEventInstant {value!r}: use one of {tuple(EVENT_INSTANTS)}")
    return value


def _envelope_width(value):
    if str(value).strip().lower() == "window":
        return "window"
    value = float(value)
    if value <= 0:
        raise ValueError("NetworkEnvelopeWidth must be 'window' or positive")
    return value


def instant_column(settings):
    """The event column `gps_candidate` and dt are read on."""
    columns = EVENT_INSTANTS[settings.event_instant]
    return "gpsPeak" if columns is None else columns[0]


def coincidence_config(settings):
    """The `CoincidenceConfig` of the wavegram path for `settings.event_instant`."""
    from wdf.analysis.robust_events import CoincidenceConfig

    columns = EVENT_INSTANTS[settings.event_instant]
    kwargs = {} if columns is None else dict(instant_columns=columns)
    if settings.event_instant == "coarse":
        kwargs["time_weight"] = 0.0
    return CoincidenceConfig(**kwargs)


def add_envelope_instants(events, graphs, labels, width):
    """`gpsEnvelope` per event: the analytic-envelope peak of its stitched
    reconstruction within `width` about `gpsPeak` ("window": the event's own
    analysis window, `n_coeff` / `fs`); `gpsPeak` where it cannot be read.

    :return: ``{ifo: events with gpsEnvelope}``.
    """
    from wdf.analysis.timing import envelope_instant

    recon = _reconstructions(graphs, labels)
    out = {}
    for ifo, ev in events.items():
        ev = ev.copy()
        peak = pd.to_numeric(ev["gpsPeak"]).to_numpy(float)
        if width == "window":
            widths = (pd.to_numeric(ev["n_coeff"]).to_numpy(float) / pd.to_numeric(ev["fs"]).to_numpy(float)
                      if {"n_coeff", "fs"} <= set(ev.columns) else np.full(len(ev), np.nan))
        else:
            widths = np.full(len(ev), float(width))
        fs = pd.to_numeric(ev["fs"]).to_numpy(float) if "fs" in ev else np.full(len(ev), np.nan)
        value = np.full(len(ev), np.nan)
        for k, cid in enumerate(ev["cluster_id"].astype(int).to_numpy()):
            if not (np.isfinite(widths[k]) and np.isfinite(fs[k])):
                continue
            try:
                value[k] = envelope_instant(recon[ifo](cid), peak[k], fs[k], widths[k])
            except (ValueError, KeyError, IndexError):
                pass
        ev["gpsEnvelope"] = np.where(np.isfinite(value), value, peak)
        out[ifo] = ev
    return out


def _with_identities(candidates, nodes, instant="gpsPeak"):
    """Each pair's detectors and clusters, and per detector of the pair its
    event's extent (`gpsStart_<ifo>`, `duration_<ifo>`), instant
    (`instant_<ifo>`) and, when the events carry them (`HrssEstimate`), its
    `hrss_<ifo>`, `hrss_err_<ifo>`, `hrss_noise_<ifo>`; `gps_candidate` is the
    mean of the two instants."""
    if candidates.empty or "node_i" not in candidates:
        return candidates
    ifo = nodes["ifo"].to_numpy()
    when = {}
    for side in ("i", "j"):
        index = candidates[f"node_{side}"].astype(int).to_numpy()
        candidates[f"ifo_{side}"] = ifo[index]
        candidates[f"cluster_{side}"] = nodes["cluster_id"].to_numpy()[index]
        when[side] = pd.to_numeric(nodes[instant]).to_numpy(float)[index]
        for name in sorted(set(ifo)):
            sel = ifo[index] == name
            for column, source in (("gpsStart", "gpsStart"), ("duration", "duration"), ("instant", instant),
                                   ("hrss", "hrss"), ("hrss_err", "hrss_err"),
                                   ("hrss_noise", "hrss_noise")):
                if source not in nodes:
                    continue
                key = f"{column}_{name}"
                if key not in candidates:
                    candidates[key] = np.nan
                candidates.loc[sel, key] = pd.to_numeric(nodes[source]).to_numpy(float)[index][sel]
    candidates["gps_candidate"] = 0.5 * (when["i"] + when["j"])
    candidates["event_instant"] = instant
    return candidates


def _with_hrss(events, graphs, labels, ar_models, settings):
    """Every detector's events with `wdf.analysis.hrss.HRSS_COLUMNS`."""
    if ar_models is None:
        raise ValueError("HrssEstimate needs the workers' noise models (run_network ar_models=)")
    out = {}
    for ifo, ev in events.items():
        if ifo not in ar_models:
            raise ValueError(f"HrssEstimate: no noise model for {ifo}")
        history = ar_models[ifo]
        if not isinstance(history, ModelHistory):
            history = ModelHistory.from_files(history)
        out[ifo] = add_hrss(ev, graphs[ifo], labels[ifo], history, settings)
    return out


def run_network(par, triggers=None, tracks=None, waveforms=None, segment_bounds=None,
                census=None, ar_models=None):
    """Stages 2, 3 and 4 on the path `par` selects.

    :param par: run configuration (`Reconstruction`, `Coincidence`, `Scorer`,
        `GNNModel`, `NetworkRankColumn`, `NetworkComparisonBin`,
        `NetworkSlides`, `NetworkSlideStep`; the track path also reads
        `TrackTimingJitter`, `TrackFrequencyTolerance`,
        `TrackLightTravelTime`, `TrackGraphBasis`).
    :param triggers: ``{ifo: trigger table with its coefficients}`` (the
        wavegram path; read with `wdf.analysis.io.triggers_from_files`).
    :param tracks: ``{ifo: track event table}`` (the track path;
        `load_tracks` of the workers' ``TrackEvents`` files); rows named in
        a `track_supersedes` are skipped.
    :param waveforms: ``{ifo: {cluster_id: (t0, waveform)}}`` (track path).
    :param segment_bounds: ``{ifo: (start, end)}`` for the time slides; from
        the triggers or events when None.
    :param census: ``{ifo: census table}`` (wavegram path): the online
        worker's ``GlitchCensusOnline`` or the offline census. With it, the
        pairs carry the network's glitch decision (`glitch_single_ifo`),
        used as `NetworkGlitchVeto` says.
    :param ar_models: ``{ifo: ModelHistory}`` or ``{ifo: ARModels file(s)}``
        (wavegram path), the noise models the workers applied; read when
        `HrssEstimate` is true, which adds `wdf.analysis.hrss.HRSS_COLUMNS` to
        the events, ``hrss_<ifo>``, ``hrss_err_<ifo>``, ``hrss_noise_<ifo>``
        and `hrss_network_det` to the candidates.

    With `AlertCSV` (`wdf.analysis.alerts`, default off) the candidates at or
    below `AlertFARThreshold` are also written to that file, in the
    GraceDB-like fields of `alert_table`; `NetworkSlides` must be positive.
    :return: NetworkResult
    """
    from wdf.analysis.network_graph import TriggerGraphBuilder, WavegramCoincidenceFinder
    from wdf.analysis.robust_events import FARConfig, IndexedCoincidenceFinder, TimeSlideFAR

    settings = NetworkSettings.from_parameters(par)
    alerts = AlertSettings.from_parameters(par)
    if alerts.csv and settings.slides <= 0:
        raise ValueError("AlertCSV selects candidates on their false-alarm rate, which needs a "
                         "background: set NetworkSlides > 0")
    scorer = None
    if settings.scorer == "gnn":
        from wdf.analysis.gnn import GNNCoincidenceScorer

        scorer = GNNCoincidenceScorer.load(settings.gnn_model)
    rank_column = "gnn_logit" if scorer is not None else settings.rank_column

    if settings.reconstruction == "wavegram":
        if triggers is None:
            raise ValueError("the wavegram path reads the detectors' triggers")
        ifos = list(triggers)
        graphs, labels, events, maps = wavegram_stage(triggers)
        hrss = HrssSettings.from_parameters(par)
        if hrss.active:
            events = _with_hrss(events, graphs, labels, ar_models, hrss)
        comparison = None
        if settings.comparison_bin > 0:
            from wdf.analysis.detector_graph import event_coefficients

            width = settings.comparison_bin * longest_travel(ifos)
            comparison = {i: event_coefficients(graphs[i], labels=labels[i], bin_seconds=width)
                          for i in ifos}
        result = NetworkResult(candidates=pd.DataFrame(), events=events, maps=maps,
                               comparison=comparison, graphs=graphs, settings=settings)
        if len(ifos) < 2:
            return result
        if settings.event_instant == "envelope":
            events = add_envelope_instants(events, graphs, labels, settings.envelope_width)
            result.events = events
        config = coincidence_config(settings)
        builder = TriggerGraphBuilder(ifos=ifos, coincidence=config)
        finder = WavegramCoincidenceFinder(IndexedCoincidenceFinder(config), builder, maps,
                                           comparison=comparison, events=events, scorer=scorer)
        screen = PairScreenSettings.from_parameters(par)
        if screen.active or census is not None:
            fs = float(pd.to_numeric(next(iter(triggers.values()))["fs"]).iloc[0])
            finder = ScreenedFinder(finder, events, _reconstructions(graphs, labels), fs, screen,
                                    rank_column, census=census, signed=scorer is not None)
            if screen.penalised:
                rank_column = RANK_STATISTIC
        candidates = finder.find(events)
        nodes = pd.concat([events[i].reset_index(drop=True).assign(ifo=i) for i in ifos],
                          ignore_index=True)
        candidates = _with_identities(candidates, nodes, instant_column(settings))
        candidates = network_hrss(candidates, ifos)
    else:
        from wdf.analysis.track_coincidence import (TrackCoincidenceFinder,
                                                    TrackCoincidenceSettings,
                                                    track_event_coefficients)

        if tracks is None:
            raise ValueError("the track path reads the detectors' track events")
        if HrssSettings.from_parameters(par).active:
            raise ValueError("HrssEstimate is read on the wavegram path's reconstructions")
        ifos = list(tracks)
        from wdf.analysis.track_reconstruction import drop_superseded

        # The tracks a later track superseded (TrackOverlapPolicy "supersede")
        # are not events: a signal yields one track.
        events = {i: drop_superseded(tracks[i].reset_index(drop=True)).assign(ifo=i) for i in ifos}
        waveforms = waveforms or {i: {} for i in ifos}
        fs = float(next((float(ev["fs"].iloc[0]) for ev in events.values() if len(ev)), np.nan))
        result = NetworkResult(candidates=pd.DataFrame(), events=events, waveforms=waveforms,
                               settings=settings)
        if len(ifos) < 2:
            return result
        builder = maps = comparison = None
        track_settings = TrackCoincidenceSettings.from_parameters(par)
        if track_settings.coherence_min is None:
            from wdf.analysis.track_coincidence import calibrate_coherence_min

            bounds = segment_bounds if segment_bounds is not None else _bounds(events, None,
                                                                                 _window_s)
            value, record = calibrate_coherence_min(events, waveforms, fs, track_settings, bounds,
                                                    settings.slide_step_s)
            track_settings.coherence_min = value
            par.TrackCoherenceMinUsed = value
            par.TrackCoherenceCalibration = record
        else:
            par.TrackCoherenceMinUsed = track_settings.coherence_min
        if scorer is not None:
            basis = str(getattr(par, "TrackGraphBasis", None) or "LocalCos64")
            par.TrackGraphBasis = basis
            window = int(max(pd.to_numeric(ev["n_coeff"]).max() for ev in events.values()))
            maps = {i: track_event_coefficients(events[i], waveforms.get(i, {}), fs, window, basis)
                    for i in ifos}
            if settings.comparison_bin > 0:
                width = settings.comparison_bin * longest_travel(ifos)
                comparison = {i: track_event_coefficients(events[i], waveforms.get(i, {}), fs,
                                                          window, basis, bin_seconds=width)
                              for i in ifos}
            builder = TriggerGraphBuilder(ifos=ifos)
        finder = TrackCoincidenceFinder(IndexedCoincidenceFinder(), events, waveforms, fs,
                                        track_settings, scorer=scorer,
                                        builder=builder, coefficients=maps, comparison=comparison)
        candidates = finder.find(events)
        result.maps, result.comparison = maps or {}, comparison
    result.finder = finder
    candidates = _rank(candidates, rank_column)
    if settings.slides > 0 and not candidates.empty:
        far = TimeSlideFAR(finder, FARConfig(n_slides=settings.slides,
                                             min_shift_s=settings.slide_step_s))
        if segment_bounds is None:
            segment_bounds = _bounds(events, triggers, _window_s)
        background = far.background_distribution(events, segment_bounds)
        start, end = segment_bounds[ifos[0]]
        candidates = far.rank_candidates(candidates, background, end - start,
                                         score_column=rank_column)
        result.background = background
    result.candidates = candidates
    if alerts.csv:
        start, end = (segment_bounds or _bounds(events, triggers, _window_s))[ifos[0]]
        fs = next((float(pd.to_numeric(t["fs"]).iloc[0]) for t in (triggers or {}).values()
                   if "fs" in t and len(t)), None)
        table = alert_table(candidates, events=events, par=par, settings=alerts, fs=fs,
                            provenance_fields=provenance(par, data_start=start, data_end=end))
        result.alerts_path = write_alerts(table, alert_path(alerts.csv, start, end, alerts.run))
    return result


class NetworkStream:
    """The network stage online: the detectors' outputs held as they arrive,
    `run_network` on what is held.

    Each detector's rows (triggers with their coefficients for the wavegram
    path, track events and waveforms for the track path) are added with
    `add`; rows whose window ended more than `NetworkMemory` seconds before
    the latest one added are dropped. `candidates` runs the stages on what is
    held.

    :param par: run configuration (as `run_network`).
    :param ifos: the detectors.
    :param ar_models: ``{ifo: ModelHistory or ARModels file(s)}`` for
        `HrssEstimate` (as `run_network`); `models` replaces one detector's.
    """

    def __init__(self, par, ifos, ar_models=None):
        self.par = par
        self.ar_models = dict(ar_models or {})
        self.settings = NetworkSettings.from_parameters(par)
        self.ifos = list(ifos)
        self.tables = {i: pd.DataFrame() for i in self.ifos}
        self.waveforms = {i: {} for i in self.ifos}
        self.census = {i: pd.DataFrame() for i in self.ifos}
        self.latest = -np.inf

    def add(self, ifo, rows, waveforms=None, census=None):
        """Hold one detector's new rows.

        :param ifo: the detector.
        :param rows: a table or a list of records: triggers with `wt_index`
            and `wt_value` (wavegram path), or track events (track path,
            `OnlineResult.tracks`).
        :param waveforms: ``{cluster_id: (t0, waveform)}`` of track events.
        :param census: new census transients of the detector
            (`OnlineResult.census`), for the glitch decision.
        :return: None
        """
        if census is not None and len(census):
            frame_c = census if isinstance(census, pd.DataFrame) else pd.DataFrame(list(census))
            self.census[ifo] = pd.concat([self.census[ifo], frame_c], ignore_index=True)
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        if frame.empty:
            return
        self.tables[ifo] = pd.concat([self.tables[ifo], frame], ignore_index=True)
        if waveforms:
            self.waveforms[ifo].update(waveforms)
        end = pd.to_numeric(frame["gps"]).to_numpy(float) + _window_s(frame)
        self.latest = max(self.latest, float(np.nanmax(end)))
        self._forget()

    def models(self, ifo, history):
        """The noise models of `ifo` read by `HrssEstimate` from now on.

        :param history: `wdf.analysis.hrss.ModelHistory` or ARModels file(s).
        """
        self.ar_models[ifo] = history

    def _forget(self):
        horizon = self.latest - self.settings.memory_s
        for i, c in self.census.items():
            if len(c):
                from wdf.analysis.pair_screen import census_extents

                self.census[i] = c[census_extents(c)[1] >= horizon].reset_index(drop=True)
        for i in self.ifos:
            t = self.tables[i]
            if t.empty:
                continue
            end = pd.to_numeric(t["gps"]).to_numpy(float) + _window_s(t)
            if self.settings.reconstruction == "track":
                end = np.maximum(end, pd.to_numeric(t["track_t_end"]).to_numpy(float))
            keep = end >= horizon
            self.tables[i] = t[keep].reset_index(drop=True)
            if self.settings.reconstruction == "track":
                alive = set(self.tables[i]["cluster_id"].astype(int))
                self.waveforms[i] = {k: v for k, v in self.waveforms[i].items() if k in alive}

    def candidates(self):
        """`run_network` on what is held.

        :return: NetworkResult
        """
        held = {i: t for i, t in self.tables.items()}
        if any(t.empty for t in held.values()):
            return NetworkResult(candidates=pd.DataFrame(), events={}, settings=self.settings)
        if self.settings.reconstruction == "track":
            return run_network(self.par, tracks=held, waveforms=self.waveforms)
        census = ({i: c for i, c in self.census.items()}
                  if any(len(c) for c in self.census.values()) else None)
        return run_network(self.par, triggers=held, census=census,
                           ar_models=self.ar_models or None)
