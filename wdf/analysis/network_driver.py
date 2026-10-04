"""Stages 2 to 4 of the WDF workflow, from the detectors' outputs to the
network candidates, on the path the configuration selects.

The workflow:

1. single-window triggers, per detector (`wdfUnitOnline`, `wdfUnitOffline`);
2. single-detector reconstruction (`Reconstruction`);
3. the graph between detectors (`Coincidence`);
4. the ranking of its candidate pairs (`Scorer`).

Two paths, which do not mix (`wdf.config.Parameters.network_path`):

- **"wavegram"**, the paper path and the default: `build_detector_graph`,
  `detector_events` and `event_coefficients` per detector (the assembly map,
  and the comparison map at `NetworkComparisonBin` times the network's
  longest light travel time when that is positive), then
  `WavegramCoincidenceFinder` over `IndexedCoincidenceFinder`'s admitted
  pairs and a `TriggerGraphBuilder(ifos)`.
- **"track"**: the detectors' track events (`TrackEvents` written by the
  workers with `Reconstruction` "track", or `reconstruct` on a map), then
  `TrackCoincidenceFinder`, which admits a pair only when its tracks are
  compatible, its coherence lag physical and its coherence at least
  `TrackCoherenceMin`; left to "calibrate", that threshold is set on the
  run's own time-slid pairs (`calibrate_coherence_min`, step
  `NetworkSlideStep`) and recorded as `TrackCoherenceMinUsed`.

Stage 4 is "classic", the column `NetworkRankColumn` (default
"network_enwdf" on the wavegram path, "track_statistic" on the track path),
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

from wdf.config.Parameters import network_path

#: Comparison map bin, in units of the network's longest light travel time.
DEFAULT_COMPARISON_BIN = 2.0
#: Light travel time used for the comparison map of a single detector, s.
SINGLE_DETECTOR_TRAVEL_S = 0.01
#: Default ranking column of the classic scorer.
DEFAULT_RANK_COLUMN = "network_enwdf"
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
        "network_enwdf" on the wavegram path, "track_statistic" on the track
        path, when not given.
    :param comparison_bin: `NetworkComparisonBin`, the comparison map's bin
        over the longest light travel time; 0 for no comparison map.
    :param slides: `NetworkSlides`, time slides of the background (0: none).
    :param slide_step_s: `NetworkSlideStep`, s.
    :param memory_s: `NetworkMemory`, what `NetworkStream` holds, s.
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
                   memory_s=float(get("NetworkMemory", DEFAULT_NETWORK_MEMORY_S)))


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


def longest_travel(ifos):
    """The network's longest light travel time, s (`SINGLE_DETECTOR_TRAVEL_S`
    for one detector)."""
    from wdf.analysis.detectors import light_travel_time

    if len(ifos) < 2:
        return SINGLE_DETECTOR_TRAVEL_S
    return max(light_travel_time(a, b) for a, b in combinations(ifos, 2))


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


def _with_identities(candidates, nodes):
    if candidates.empty or "node_i" not in candidates:
        return candidates
    for side in ("i", "j"):
        index = candidates[f"node_{side}"].astype(int)
        candidates[f"ifo_{side}"] = nodes["ifo"].to_numpy()[index]
        candidates[f"cluster_{side}"] = nodes["cluster_id"].to_numpy()[index]
    return candidates


def run_network(par, triggers=None, tracks=None, waveforms=None, segment_bounds=None):
    """Stages 2, 3 and 4 on the path `par` selects.

    :param par: run configuration (`Reconstruction`, `Coincidence`, `Scorer`,
        `GNNModel`, `NetworkRankColumn`, `NetworkComparisonBin`,
        `NetworkSlides`, `NetworkSlideStep`; the track path also reads
        `TrackTimingJitter`, `TrackFrequencyTolerance`,
        `TrackLightTravelTime`, `TrackGraphBasis`).
    :param triggers: ``{ifo: trigger table with its coefficients}`` (the
        wavegram path; read with `wdf.analysis.io.triggers_from_files`).
    :param tracks: ``{ifo: track event table}`` (the track path;
        `load_tracks` of the workers' ``TrackEvents`` files).
    :param waveforms: ``{ifo: {cluster_id: (t0, waveform)}}`` (track path).
    :param segment_bounds: ``{ifo: (start, end)}`` for the time slides; from
        the triggers or events when None.
    :return: NetworkResult
    """
    from wdf.analysis.network_graph import TriggerGraphBuilder, WavegramCoincidenceFinder
    from wdf.analysis.robust_events import FARConfig, IndexedCoincidenceFinder, TimeSlideFAR

    settings = NetworkSettings.from_parameters(par)
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
        builder = TriggerGraphBuilder(ifos=ifos)
        finder = WavegramCoincidenceFinder(IndexedCoincidenceFinder(), builder, maps,
                                           comparison=comparison, events=events, scorer=scorer)
        candidates = finder.find(events)
        nodes = pd.concat([events[i].reset_index(drop=True).assign(ifo=i) for i in ifos],
                          ignore_index=True)
        candidates = _with_identities(candidates, nodes)
    else:
        from wdf.analysis.track_coincidence import (TrackCoincidenceFinder,
                                                    TrackCoincidenceSettings,
                                                    track_event_coefficients)

        if tracks is None:
            raise ValueError("the track path reads the detectors' track events")
        ifos = list(tracks)
        events = {i: tracks[i].reset_index(drop=True).assign(ifo=i) for i in ifos}
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
    """

    def __init__(self, par, ifos):
        self.par = par
        self.settings = NetworkSettings.from_parameters(par)
        self.ifos = list(ifos)
        self.tables = {i: pd.DataFrame() for i in self.ifos}
        self.waveforms = {i: {} for i in self.ifos}
        self.latest = -np.inf

    def add(self, ifo, rows, waveforms=None):
        """Hold one detector's new rows.

        :param ifo: the detector.
        :param rows: a table or a list of records: triggers with `wt_index`
            and `wt_value` (wavegram path), or track events (track path,
            `OnlineResult.tracks`).
        :param waveforms: ``{cluster_id: (t0, waveform)}`` of track events.
        :return: None
        """
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        if frame.empty:
            return
        self.tables[ifo] = pd.concat([self.tables[ifo], frame], ignore_index=True)
        if waveforms:
            self.waveforms[ifo].update(waveforms)
        end = pd.to_numeric(frame["gps"]).to_numpy(float) + _window_s(frame)
        self.latest = max(self.latest, float(np.nanmax(end)))
        self._forget()

    def _forget(self):
        horizon = self.latest - self.settings.memory_s
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
        return run_network(self.par, triggers=held)
