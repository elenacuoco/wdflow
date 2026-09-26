"""From each detector's triggers to the network's released candidates.

The search read end to end, with two thresholds per detector. Each detector
is searched on its own at a low first threshold, its triggers become events
at the detector stage, only the events loud enough for their extent reach the
network stage, and the pairs the network admits are given a false-alarm rate
by time slides of the same events:

    triggers --> tiles --> clusters --> ridges --> events
             --> second threshold --> pairs --> rate

The first threshold is the search's own, on each window: under the block rule
a window that keeps any coefficient has `EnWDF` of at least `sqrt(4.505)`, so
a first threshold at that floor writes every window holding a surviving block
and the grouping sees all of them.

The detector stage is `wdf.analysis.pixel_graph`. Tiles are joined on the
geometry of the analysis alone --- the stride the search advanced by and the
steps of the dyadic ladder it tiled frequency with, symmetric in time and in
direction --- and each connected cluster is then cut down to the path its own
ridge traces across the plane (`pixel_graph.follow_ridges`); what the path
leaves behind is regrouped into events of its own, so no tile is dropped.
Every event carries three statistics at the cluster level:

`EnWDF`
    the event's energy: the norm of the waveform the event's own tiles invert
    to, stitched across the windows it spans, on the noise scale. It is the
    quantity a matched filter reads off the same transient, and the event's
    signal-to-noise ratio in that sense.
`EnWDF_significance`
    that energy made independent of the event's extent,
    `-log P(EnWDF' >= EnWDF | H0, n_pixels)`, measured on the detector's own
    events, each scored by a calibration fitted away from its own time
    (`wdf.analysis.event_significance.significance_off_source`). A threshold
    admits every tile at a floor, so the raw energy of an accidental event
    grows with how many tiles it holds; the calibrated one is exponential with
    unit rate whatever the extent, which is what makes a threshold on it a
    statement about a rate.
`EnWDF_window`
    the loudest block the event holds: its share of the window carrying the
    most of it. It is what a search without the grouping would report, and
    it is kept beside the other two for that comparison.

The second threshold selects on `EnWDF_significance`, never on the raw energy,
and it is translated from the single threshold it stands for. With `N` events
built from the low-threshold triggers and `N_ref` built by the same detector
stage from the triggers a search at `reference_threshold` writes --- exactly
those whose own `EnWDF` reaches it, since the search's threshold decides only
whether a window is written --- the cut is

    EnWDF_significance >= S* = log(N / N_ref),

the threshold at which the calibrated cut passes, per unit time, as many
events of noise as a single threshold at the reference passes to coincidence.
On the raw energy it is a threshold that rises with the event's extent, as
`EventCalibration.statistic_at` reads it. It is applied at the reference and
measured at each of `reported_thresholds`.

The network stage is `wdf.analysis.network_graph`, on the pairs
`wdf.analysis.robust_events.IndexedCoincidenceFinder` admits --- the two
events' stretches of time meeting within the light travel time widened by
their own spreads --- and nothing else gates a pair. Each released pair also
carries the lag the two reconstructions measure and whether it is within the
light travel time, which is the only physical constraint a pair has; that is
reported and does not gate. The background is drawn by `TimeSlideFAR` inside
every stretch in which the same detectors were searched, since a slide must
wrap inside data that exists, and the stretches' backgrounds are pooled with
their livetimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wdf.analysis.cluster_coefficients import iter_tile_cluster_coefficients
from wdf.analysis.detectors import light_travel_time, network_light_travel_time
from wdf.analysis.event_significance import (EventCalibration,
                                             rate_matched_significance,
                                             significance_off_source)
from wdf.analysis.network_graph import (TriggerGraphBuilder,
                                        WavegramCoincidenceFinder)
from wdf.analysis.pixel_graph import (PixelGraphConfig, build_pixel_graph,
                                      cluster_events, cluster_wavegrams,
                                      follow_ridges, tile_labels)
from wdf.analysis.ridge import event_ridge, ridge_track
from wdf.analysis.robust_events import (CoincidenceConfig, FARConfig,
                                        IndexedCoincidenceFinder, TimeSlideFAR)
from wdf.analysis.scale import local_noise_scale, on_local_scale, pixel_cloud
from wdf.analysis.timing import MAX_LAG_S, arrival_time_difference

#: The per-event columns a released candidate carries from each of its two
#: events, suffixed `_i` and `_j`.
EVENT_COLUMNS = ["gpsStart", "gpsPeak", "duration", "freqMin", "freqQ05",
                 "freqMean", "freqQ95", "freqMax", "EnWDF", "EnWDF_window",
                 "EnWDF_significance", "passes_event_cut", "n_pixels",
                 "n_triggers"]


def passes_column(reference: float) -> str:
    """The event column saying whether the cut translated from a reference
    first threshold is passed, such as ``passes_5``."""
    return f"passes_{float(reference):g}"


@dataclass
class ReleaseConfig:
    """What the detector and network stages are told.

    :param window: analysis window, samples of the analysed stream.
    :param overlap: overlap between consecutive windows, samples.
    :param reference_threshold: the search statistic `EnWDF` a single-threshold
        search emits a window at; the second threshold is translated from it
        and applied. It must not be below the first threshold the triggers were
        written at, or the reference population is not a subset of what was
        searched.
    :param reported_thresholds: further references the second threshold is
        translated from and measured at, without being applied.
    :param event_cut: whether only the events passing the second threshold
        reach the network stage. Off, every event does.
    :param local_scale_neighbours: blocks the noise scale of each block is read
        over, as `wdf.analysis.scale.local_noise_scale`; None keeps each
        block's own. The search statistic and the tiles' scale are both read
        there before anything is grouped.
    :param pixel: when two tiles may belong to one transient.
    :param ridge_bins: time bins a cluster's extent is divided into for its
        ridge; None keeps every cluster whole.
    :param calibration_folds: stretches of equal duration the calibration is
        cut into, so that no event is scored against its own neighbourhood.
    :param calibration_min_count: fewest background events an extent bin of
        the calibration may hold.
    :param wavegram_time_bins: columns of an event's compact map.
    :param coincidence: when two events of two detectors may be one signal.
    :param slides: how the accidental background is drawn, per stretch. The
        step between displacements must exceed the extent almost every event
        has, or two lags pair the same clusters twice; a search run at a low
        first threshold assembles longer events than one at a single high
        threshold, and the step is stated accordingly.
    :param match_wavegrams: whether every admitted pair's two renderings are
        compared at the displacements its tolerance allows, which is what
        `coherent_statistic` and `network_correlation` are measured from. It
        costs one correlation profile per pair and per displacement, on the
        zero lag and on every slide, and it is most of the network stage's
        time; the statistics in the default ranking do not need it.
    :param ranking: the network statistics a false-alarm rate is attached to;
        the released list is ordered on the first. `network_min_significance`
        ranks a pair on the weaker of its two events' calibrated energies;
        `network_min_enwdf_timed` on the weaker loudest block, discounted by
        how much of its timing tolerance the pair used; `network_morphology`
        on the coherent energy of the tiles the two events share.
    :param minimum_interval_s: shortest stretch of common search time a
        background is drawn on; shorter ones hold too few distinct slides and
        are left out of the zero lag and the livetime alike.
    :param timed_candidates: how many of the released pairs, from the top of
        the list, are timed on their reconstructions; the lag costs a
        correlation per pair and is read where a candidate is read.
    """

    window: int = 512
    overlap: int = 32
    reference_threshold: float = 5.0
    reported_thresholds: tuple = (6.0,)
    event_cut: bool = True
    local_scale_neighbours: int | None = 41
    pixel: PixelGraphConfig = field(default_factory=lambda: PixelGraphConfig(
        stride_tolerance=2.0, band_tolerance=2))
    ridge_bins: int | None = 32
    calibration_folds: int = 10
    calibration_min_count: int = 200
    wavegram_time_bins: int = 64
    coincidence: CoincidenceConfig = field(default_factory=CoincidenceConfig)
    slides: FARConfig = field(default_factory=lambda: FARConfig(
        n_slides=100, min_shift_s=10.0))
    match_wavegrams: bool = False
    ranking: tuple = ("network_min_significance", "network_min_enwdf_timed",
                      "network_morphology")
    minimum_interval_s: float = 600.0
    timed_candidates: int = 500

    @property
    def references(self) -> tuple:
        """Every reference the second threshold is translated from, the
        applied one first."""
        return tuple(dict.fromkeys(
            [float(self.reference_threshold)]
            + [float(value) for value in self.reported_thresholds]))


@dataclass
class DetectorStage:
    """One detector's events, and everything they were built from.

    :param ifo: the detector.
    :param triggers: the triggers as the stage read them: `EnWDF` and `sigma`
        on the local noise scale, the search's own statistic kept as
        `EnWDF_search`.
    :param cloud: their tiles, as `wdf.analysis.scale.pixel_cloud` returns.
    :param graph: the pixel graph over the tiles.
    :param labels: the event of every node of the graph.
    :param tiles: the event of every tile of the cloud, repeated regions
        included; -1 for a tile no event holds.
    :param events: one row per event, with `EnWDF` measured on the
        reconstruction, `EnWDF_tiles` the norm over the tiles,
        `EnWDF_window`, `EnWDF_significance`, `passes_event_cut` for the
        applied threshold and one `passes_<reference>` column per reference.
    :param maps: `{cluster_id: EventWavegram}` for assembly.
    :param comparison: the same events rendered for the network comparison.
    :param n_reference: `{reference: events}` the same stage builds from the
        triggers a single-threshold search at that reference writes.
    :param event_threshold: `{reference: S*}`, the second threshold on
        `EnWDF_significance` translated from each reference.
    :param calibration: the calibration fitted on every event of the stretch;
        what `statistic_at` reads the size-dependent raw threshold from. The
        events themselves are scored out of their own time, not by this.
    """

    ifo: str
    triggers: pd.DataFrame
    cloud: pd.DataFrame
    graph: object
    labels: np.ndarray
    tiles: np.ndarray
    events: pd.DataFrame
    maps: dict
    comparison: dict
    n_reference: dict
    event_threshold: dict
    calibration: EventCalibration

    def coincident_events(self, event_cut: bool) -> pd.DataFrame:
        """The events this detector sends to the network stage.

        :type event_cut: bool
        :param event_cut: keep only those passing the applied threshold.
        :return: pandas.DataFrame -- rows of `events`, in their order.
        """
        if not event_cut:
            return self.events
        return self.events[self.events["passes_event_cut"].to_numpy(dtype=bool)]


def prepare_triggers(triggers: pd.DataFrame,
                     neighbours: int | None) -> pd.DataFrame:
    """The triggers on the scale every later stage reads them on.

    A block's own noise scale is measured on the data it holds, signal
    included, so a transient loud enough to matter divides itself by a scale
    it inflated. The statistic and the scale the tiles are normalised on are
    both read on the neighbouring blocks instead, as
    `wdf.analysis.scale.on_local_scale` does; the search's own statistic is
    kept as `EnWDF_search`, since that is the quantity the search's threshold
    was applied to.

    :type triggers: pandas.DataFrame
    :param triggers: one detector's triggers, carrying `gps`, `sigma`,
        `EnWDF` and the coefficient columns.
    :type neighbours: int | None
    :param neighbours: blocks the local scale is read over; None keeps each
        block's own scale.
    :return: pandas.DataFrame -- the triggers in time order with a fresh
        index, which the tiles' `trigger_index` refers to.
    """
    out = triggers.sort_values("gps", kind="stable").reset_index(drop=True)
    out = out.assign(EnWDF_search=out["EnWDF"].to_numpy(dtype=float))
    if neighbours is None or out.empty:
        return out
    return out.assign(
        EnWDF=on_local_scale(out, neighbours=int(neighbours)),
        sigma=local_noise_scale(out, neighbours=int(neighbours)))


def assemble(triggers: pd.DataFrame, config: ReleaseConfig):
    """The detector stage's grouping: tiles, graph, ridges and events.

    :type triggers: pandas.DataFrame
    :param triggers: one detector's triggers, as `prepare_triggers` returns.
    :type config: ReleaseConfig
    :param config: the run's configuration.
    :return: tuple -- `(cloud, graph, labels, events)`.
    :raises ValueError: if the stride allowance is asked for and the triggers
        declare no stride, which it rests on.
    """
    if config.pixel.stride_tolerance > 0.0 and "stride" not in triggers:
        raise ValueError(
            "stride_tolerance is measured in strides of the search, and these "
            "triggers declare none: read them with "
            "wdf.analysis.io.triggers_from_files")
    cloud = pixel_cloud(triggers)
    graph = build_pixel_graph(cloud, config=config.pixel)
    labels = graph.components()
    if config.ridge_bins is not None:
        labels = follow_ridges(graph, labels, n_bins=int(config.ridge_bins))
    return cloud, graph, labels, cluster_events(graph, labels=labels)


def measured_energy(cloud, tiles, triggers, window, overlap, n_events):
    """Each event's energy, on the waveform its own tiles invert to.

    :param cloud: the tile cloud.
    :param tiles: the event of every tile, as `tile_labels` returns.
    :param triggers: the triggers the cloud was built from.
    :type window: int
    :param window: analysis window, samples.
    :type overlap: int
    :param overlap: overlap, samples.
    :type n_events: int
    :param n_events: how many events there are.
    :return: numpy.ndarray -- one value per event, NaN where the event holds
        no measurable tile.
    """
    out = np.full(int(n_events), np.nan)
    for label, cluster in iter_tile_cluster_coefficients(
            cloud, tiles, triggers, window, overlap):
        out[label] = cluster.enwdf()
    return out


def reference_events(triggers: pd.DataFrame, reference: float,
                     config: ReleaseConfig) -> int:
    """How many events the same stage builds at a single threshold.

    :type triggers: pandas.DataFrame
    :param triggers: the detector's triggers as the search wrote them.
    :type reference: float
    :param reference: the single threshold on the search's own `EnWDF`.
    :type config: ReleaseConfig
    :param config: the run's configuration.
    :return: int -- the events built from the triggers at or above it.
    """
    kept = triggers.iloc[np.flatnonzero(
        triggers["EnWDF"].to_numpy(dtype=float) >= float(reference))]
    if kept.empty:
        return 0
    return len(assemble(prepare_triggers(kept, config.local_scale_neighbours),
                        config)[3])


def detector_stage(triggers: pd.DataFrame, config: ReleaseConfig,
                   comparison_bin_s: float) -> DetectorStage:
    """One detector's events, measured, calibrated and cut.

    :type triggers: pandas.DataFrame
    :param triggers: the detector's triggers, as the search wrote them, at a
        first threshold no higher than any of `config.references`.
    :type config: ReleaseConfig
    :param config: the run's configuration.
    :type comparison_bin_s: float
    :param comparison_bin_s: column of the map two detectors are compared on,
        seconds; of the order of the network's light travel time, or a real
        delay moves no cell.
    :return: DetectorStage
    """
    prepared = prepare_triggers(triggers, config.local_scale_neighbours)
    ifo = str(prepared["ifo"].iloc[0]) if "ifo" in prepared and len(prepared) else ""
    cloud, graph, labels, events = assemble(prepared, config)
    tiles = tile_labels(cloud, graph, labels)
    energy = measured_energy(cloud, tiles, prepared, config.window,
                             config.overlap, len(events))
    events = events.assign(
        EnWDF_tiles=events["EnWDF"].to_numpy(dtype=float),
        EnWDF=np.where(np.isfinite(energy), energy,
                       events["EnWDF_window"].to_numpy(dtype=float)))

    calibration_kwargs = dict(statistic="EnWDF", size_column="n_pixels",
                              min_count=config.calibration_min_count)
    significance = significance_off_source(
        events, events, folds=config.calibration_folds,
        time_column="gpsStart", **calibration_kwargs)
    events = events.assign(EnWDF_significance=significance)
    calibration = EventCalibration.fit(events, **calibration_kwargs)

    counted, thresholds, passing = {}, {}, {}
    for reference in config.references:
        counted[reference] = reference_events(triggers, reference, config)
        thresholds[reference] = rate_matched_significance(len(events),
                                                          counted[reference])
        passing[passes_column(reference)] = significance >= thresholds[reference]
    events = events.assign(
        passes_event_cut=passing[passes_column(config.reference_threshold)],
        **passing)

    maps = cluster_wavegrams(graph, labels, time_bins=config.wavegram_time_bins)
    comparison = cluster_wavegrams(graph, labels,
                                   time_bins=config.wavegram_time_bins,
                                   bin_seconds=comparison_bin_s)
    return DetectorStage(ifo=ifo, triggers=prepared, cloud=cloud, graph=graph,
                         labels=labels, tiles=tiles, events=events, maps=maps,
                         comparison=comparison, n_reference=counted,
                         event_threshold=thresholds, calibration=calibration)


def common_intervals(spans: dict, minimum_s: float = 0.0) -> list:
    """The stretches in which the same detectors were all being searched.

    A time slide wraps a detector's events inside a stretch, and it can only
    pair them with the other detectors' events where those were being searched
    too. The search time is cut where any detector starts or stops, and each
    piece is labelled with the detectors live throughout it; consecutive
    pieces with the same detectors are one stretch.

    :type spans: dict
    :param spans: `{ifo: [(start, end), ...]}`, the stretches each detector's
        search covered, GPS seconds.
    :type minimum_s: float
    :param minimum_s: shortest stretch kept.
    :return: list -- `(start, end, ifos)` per stretch holding at least two
        detectors, in time order, `ifos` in the order `spans` gives them.
    """
    ifos = list(spans)
    cuts = sorted({float(t) for ifo in ifos for span in spans[ifo] for t in span})
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        live = tuple(ifo for ifo in ifos
                     if any(a <= lo and hi <= b for a, b in spans[ifo]))
        if len(live) < 2:
            continue
        if out and out[-1][2] == live and out[-1][1] == lo:
            out[-1] = (out[-1][0], hi, live)
        else:
            out.append((lo, hi, live))
    return [(lo, hi, live) for lo, hi, live in out if hi - lo >= float(minimum_s)]


def _within(events: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    start = events["gpsStart"].to_numpy(dtype=float)
    return events[(start >= lo) & (start < hi)]


def with_node_statistic(table: pd.DataFrame, events: dict,
                        ifos) -> pd.DataFrame:
    """Pairs, with the weaker of their two events' calibrated energies.

    A pair's nodes are positions in the detectors' events laid end to end in
    the order the graph was built with, and a time slide keeps that order, so
    one lookup serves the zero lag and every slide.

    :type table: pandas.DataFrame
    :param table: pairs carrying `node_i` and `node_j`.
    :type events: dict
    :param events: `{ifo: events}` the graph was built from.
    :param ifos: the detectors, in the graph's order.
    :return: pandas.DataFrame -- `table` with `network_min_significance`.
    """
    if table.empty:
        return table.assign(network_min_significance=pd.Series(dtype=float))
    value = np.concatenate([events[ifo]["EnWDF_significance"].to_numpy(dtype=float)
                            for ifo in ifos])
    i = table["node_i"].to_numpy(dtype=int)
    j = table["node_j"].to_numpy(dtype=int)
    return table.assign(network_min_significance=np.minimum(value[i], value[j]))


def network_stage(stages: dict, intervals: list, config: ReleaseConfig):
    """The zero-lag candidates and their accidental background, per stretch.

    :type stages: dict
    :param stages: `{ifo: DetectorStage}`.
    :type intervals: list
    :param intervals: what `common_intervals` returns.
    :type config: ReleaseConfig
    :param config: the run's configuration.
    :return: tuple -- `(candidates, background, livetime_s, observed_s)`: the
        admitted pairs of every stretch, carrying the detector and the
        `cluster_id` of both events; the accidental pairs of every stretch's
        slides; the slid livetime; and the zero-lag time searched.
    :raises ValueError: if a stretch cannot hold the slides asked for.
    """
    candidates, background = [], []
    livetime, observed = 0.0, 0.0
    for number, (lo, hi, ifos) in enumerate(intervals):
        events = {ifo: _within(stages[ifo].coincident_events(config.event_cut),
                               lo, hi).reset_index(drop=True) for ifo in ifos}
        if any(frame.empty for frame in events.values()):
            continue
        maps = {ifo: stages[ifo].maps for ifo in ifos}
        comparison = {ifo: stages[ifo].comparison for ifo in ifos}
        builder = TriggerGraphBuilder(coincidence=config.coincidence,
                                      ifos=list(ifos),
                                      wavegram_time_bins=config.wavegram_time_bins,
                                      match_wavegrams=config.match_wavegrams)
        prepared = builder.prepare(events, maps, comparison=comparison)
        graph = builder.build_from_prepared(events, prepared)
        if len(graph.cross_edges):
            table = with_node_statistic(graph.candidate_table(), events, ifos)
            nodes = graph.nodes
            for side in ("i", "j"):
                node = table[f"node_{side}"].to_numpy(dtype=int)
                table[f"ifo_{side}"] = nodes["ifo"].to_numpy()[node]
                table[f"cluster_{side}"] = nodes["cluster_id"].to_numpy(dtype=int)[node]
            candidates.append(table.assign(interval=number))
        finder = WavegramCoincidenceFinder(
            IndexedCoincidenceFinder(config.coincidence), builder, maps,
            comparison=comparison, prepared=prepared)
        try:
            slid = TimeSlideFAR(finder, config.slides).background_distribution(
                events, {ifo: (lo, hi) for ifo in ifos})
        except ValueError as problem:
            raise ValueError(
                f"no background can be drawn on {lo:.0f}-{hi:.0f} "
                f"({','.join(ifos)}): {problem}") from problem
        livetime += float(slid.attrs["total_livetime_s"])
        observed += float(hi - lo)
        if len(slid):
            background.append(with_node_statistic(slid, events, ifos)
                              .assign(interval=number))
    candidates = (pd.concat(candidates, ignore_index=True) if candidates
                  else pd.DataFrame())
    background = (pd.concat(background, ignore_index=True) if background
                  else pd.DataFrame())
    background.attrs["total_livetime_s"] = livetime
    # The slides per second of zero lag, which is what a background that
    # produced no accidental at all is read with.
    background.attrs["n_slides"] = livetime / observed if observed > 0 else 0.0
    return candidates, background, livetime, observed


def rank(candidates: pd.DataFrame, background: pd.DataFrame, observed_s: float,
         statistics) -> pd.DataFrame:
    """Every candidate with the false-alarm rate of each ranking statistic.

    :type candidates: pandas.DataFrame
    :param candidates: the zero-lag pairs.
    :type background: pandas.DataFrame
    :param background: the pooled accidental pairs, carrying the slid livetime
        in `attrs["total_livetime_s"]`.
    :type observed_s: float
    :param observed_s: the zero-lag time searched, seconds; the false-alarm
        probability is read over it.
    :param statistics: the statistics, each a column of both tables.
    :return: pandas.DataFrame -- `candidates` with, for each statistic `s`,
        `far_per_day_<s>` and `fap_<s>`, ordered on the first statistic's rate
        and then on the statistic itself.
    """
    if candidates.empty:
        return candidates.copy()
    out = candidates.reset_index(drop=True).assign(
        _row=np.arange(len(candidates)))
    ranker = TimeSlideFAR(None)
    for statistic in statistics:
        ranked = ranker.rank_candidates(out[["_row", statistic]], background,
                                        observed_s, score_column=statistic)
        ranked = ranked.set_index("_row").reindex(out["_row"])
        out[f"far_per_day_{statistic}"] = ranked["far_per_day"].to_numpy()
        out[f"fap_{statistic}"] = ranked["fap"].to_numpy()
    first = statistics[0]
    return (out.sort_values([f"far_per_day_{first}", first],
                            ascending=[True, False])
            .drop(columns="_row").reset_index(drop=True))


def with_events(candidates: pd.DataFrame, stages: dict) -> pd.DataFrame:
    """The candidates, with what each of their two events measured.

    :type candidates: pandas.DataFrame
    :param candidates: pairs carrying `ifo_i`, `cluster_i`, `ifo_j` and
        `cluster_j`.
    :type stages: dict
    :param stages: `{ifo: DetectorStage}`.
    :return: pandas.DataFrame -- `candidates` with `EVENT_COLUMNS` suffixed
        `_i` and `_j`, and the pair's own extent as `gpsStart` and
        `duration`, which is what a candidate is matched in time on.
    """
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    table = pd.concat([stage.events[["cluster_id"] + EVENT_COLUMNS].assign(ifo=ifo)
                       for ifo, stage in stages.items()], ignore_index=True)
    for side in ("i", "j"):
        out = out.merge(
            table.rename(columns={name: f"{name}_{side}" for name in EVENT_COLUMNS}
                         | {"ifo": f"ifo_{side}", "cluster_id": f"cluster_{side}"}),
            on=[f"ifo_{side}", f"cluster_{side}"], how="left")
    first = np.minimum(out["gpsStart_i"], out["gpsStart_j"])
    last = np.maximum(out["gpsStart_i"] + out["duration_i"],
                      out["gpsStart_j"] + out["duration_j"])
    return out.assign(gpsStart=first, duration=last - first)


@dataclass
class Release:
    """What the search released, and everything it was read from.

    :param config: the configuration the release was made under.
    :param stages: `{ifo: DetectorStage}`.
    :param spans: `{ifo: [(start, end), ...]}`, what each detector searched.
    :param intervals: the stretches of common search time, as
        `common_intervals` returns.
    :param candidates: the released pairs, ordered on the first ranking
        statistic's false-alarm rate.
    :param background: the pooled accidental pairs.
    :param livetime_s: the slid livetime the rates are read over.
    :param observed_s: the zero-lag time searched in coincidence.
    """

    config: ReleaseConfig
    stages: dict
    spans: dict
    intervals: list
    candidates: pd.DataFrame
    background: pd.DataFrame
    livetime_s: float
    observed_s: float

    def counts(self) -> pd.DataFrame:
        """How many triggers and events each threshold leaves, per detector.

        :return: pandas.DataFrame -- one row per detector: the triggers at the
            first threshold, the events the stage built from them, and for
            each reference the triggers a single threshold there writes, the
            events built from those, the second threshold translated from it
            and the events passing it; then how many reach the network stage.
        """
        rows = []
        for ifo, stage in self.stages.items():
            written = stage.triggers["EnWDF_search"].to_numpy(dtype=float)
            row = dict(ifo=ifo, searched_s=sum(b - a for a, b in self.spans[ifo]),
                       triggers=len(written), events=len(stage.events))
            for reference in self.config.references:
                name = f"{reference:g}"
                row[f"triggers_at_{name}"] = int((written >= reference).sum())
                row[f"events_at_{name}"] = stage.n_reference[reference]
                row[f"threshold_S_{name}"] = stage.event_threshold[reference]
                row[f"events_passing_{name}"] = int(
                    stage.events[passes_column(reference)].sum())
            row["events_to_network"] = len(stage.coincident_events(
                self.config.event_cut))
            rows.append(row)
        return pd.DataFrame(rows)

    def event_coefficients(self, ifo: str, cluster_id: int):
        """One event's own coefficients, one row per window it touches.

        :type ifo: str
        :param ifo: the detector.
        :type cluster_id: int
        :param cluster_id: the event.
        :return: wdf.analysis.cluster_coefficients.ClusterCoefficients
        :raises KeyError: if the event holds no tile.
        """
        stage = self.stages[ifo]
        rows = np.flatnonzero(stage.tiles == int(cluster_id))
        if not len(rows):
            raise KeyError(f"{ifo} has no event {cluster_id} holding a tile")
        found = iter_tile_cluster_coefficients(
            stage.cloud.iloc[rows], stage.tiles[rows], stage.triggers,
            self.config.window, self.config.overlap)
        return next(found)[1]

    def event_tiles(self, ifo: str, cluster_id: int) -> pd.DataFrame:
        """The tiles an event owns, each region once, on the noise scale.

        :type ifo: str
        :param ifo: the detector.
        :type cluster_id: int
        :param cluster_id: the event.
        :return: pandas.DataFrame -- the graph's nodes of that event, with
            `snr`, each tile's amplitude on its window's noise scale.
        """
        stage = self.stages[ifo]
        nodes = stage.graph.nodes[stage.labels == int(cluster_id)]
        return nodes.assign(snr=np.sqrt(nodes["energy"].to_numpy(dtype=float))
                            / nodes["sigma"].to_numpy(dtype=float))

    def event_ridge(self, ifo: str, cluster_id: int):
        """The track an event's tiles trace, measured bins and filled ones.

        :type ifo: str
        :param ifo: the detector.
        :type cluster_id: int
        :param cluster_id: the event.
        :return: tuple -- `(time, log_frequency, centres, track, measured)`:
            the ridge as `wdf.analysis.ridge.event_ridge` measures it, one
            tile per bin; the centres of the bins; and the gap-filled track
            `ridge_track` draws through them, with the mask of the bins
            actually measured.
        """
        n_bins = int(self.config.ridge_bins or 32)
        tiles = self.event_tiles(ifo, cluster_id)
        time, log_f, _ = event_ridge(
            tiles["t_lo"], tiles["t_hi"], tiles["f_lo"], tiles["f_hi"],
            tiles["snr"] ** 2, n_bins=n_bins)
        edges = np.linspace(float(tiles["t_lo"].min()), float(tiles["t_hi"].max()),
                            n_bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        track, measured = ridge_track(time, log_f, bin_centres=centres)
        return time, log_f, centres, track, measured


def timed(release: Release, max_lag_s: float = MAX_LAG_S,
          limit: int | None = None) -> pd.DataFrame:
    """The released pairs, each with the lag its two reconstructions measure.

    The light travel time between the two sites is the only physical
    constraint a pair has, and the lag is what it constrains: the
    cross-correlation of the two events' stitched reconstructions, read below
    the tile (`wdf.analysis.timing.arrival_time_difference`). A pair is
    `physical` when that lag is within the light travel time widened by the
    width the correlation declares. The separation of the tile centres is not
    a second test --- a tile carries its own width, tens of milliseconds low in
    the band --- and neither this flag nor the lag gates a pair: a pair the
    geometry admits and that is not one event is for the ranking to put down.
    Where the two reconstructions sit further apart than the lag search
    reaches, the lag is the difference of the events' own instants.

    :type release: Release
    :param release: the release whose candidates are timed.
    :type max_lag_s: float
    :param max_lag_s: half-width of the lag search, seconds.
    :type limit: int | None
    :param limit: time only this many pairs, from the top of the list; the
        others carry no lag and `physical` is left unset. All of them when
        None.
    :return: pandas.DataFrame -- the candidates with `lag_s`, `lag_sigma_s`,
        `lag_from`, `light_travel_s` and `physical`.
    """
    out = release.candidates.copy()
    if out.empty:
        return out.assign(lag_s=[], lag_sigma_s=[], lag_from=[],
                          light_travel_s=[], physical=[])
    series = {}

    def reconstruction(ifo, cluster):
        if (ifo, cluster) not in series:
            coefficients = release.event_coefficients(ifo, cluster)
            series[(ifo, cluster)] = (coefficients.reconstruct(), coefficients.fs)
        return series[(ifo, cluster)]

    count = len(out) if limit is None else min(int(limit), len(out))
    lag = out["dt_s"].to_numpy(dtype=float).copy()
    lag[count:] = np.nan
    sigma = np.full(len(out), np.nan)
    source = np.full(len(out), "instants", dtype=object)
    source[count:] = None
    pairs = zip(out["ifo_i"].iloc[:count], out["cluster_i"].iloc[:count],
                out["ifo_j"].iloc[:count], out["cluster_j"].iloc[:count])
    for row, (ifo_i, cluster_i, ifo_j, cluster_j) in enumerate(pairs):
        (first, fs), (second, _) = (reconstruction(ifo_i, int(cluster_i)),
                                    reconstruction(ifo_j, int(cluster_j)))
        try:
            lag[row], sigma[row] = arrival_time_difference(first, second, fs,
                                                           max_lag_s)
        except ValueError:
            continue
        source[row] = "correlation"
    travel = np.array([light_travel_time(a, b)
                       for a, b in zip(out["ifo_i"], out["ifo_j"])])
    physical = pd.array(np.abs(lag) <= travel + np.nan_to_num(sigma),
                        dtype="boolean")
    physical[count:] = pd.NA
    return out.assign(lag_s=lag, lag_sigma_s=sigma, lag_from=source,
                      light_travel_s=travel, physical=physical)


def release(triggers: dict, spans: dict,
            config: ReleaseConfig | None = None) -> Release:
    """The network's released candidates, from each detector's triggers.

    :type triggers: dict
    :param triggers: `{ifo: triggers}`, each as
        `wdf.analysis.io.triggers_from_files` reads them, at one window length
        and one first threshold.
    :type spans: dict
    :param spans: `{ifo: [(start, end), ...]}`, the stretches each detector's
        search covered.
    :type config: ReleaseConfig | None
    :param config: the configuration; the default one when None.
    :return: Release -- the first `config.timed_candidates` of its
        candidates timed by `timed`.
    """
    config = ReleaseConfig() if config is None else config
    comparison_bin_s = 2.0 * network_light_travel_time(list(triggers))
    stages = {ifo: detector_stage(frame.assign(ifo=ifo), config,
                                  comparison_bin_s)
              for ifo, frame in triggers.items()}
    intervals = common_intervals(spans, minimum_s=config.minimum_interval_s)
    candidates, background, livetime, observed = network_stage(
        stages, intervals, config)
    ranked = rank(candidates, background, observed, config.ranking)
    made = Release(config=config, stages=stages, spans=spans,
                   intervals=intervals, candidates=with_events(ranked, stages),
                   background=background, livetime_s=livetime,
                   observed_s=observed)
    made.candidates = timed(made, limit=config.timed_candidates)
    return made
