"""From each detector's triggers to the network's released candidates, at one threshold.

The chain the method paper validates, read end to end on the triggers of a
stretch of data. Each detector's triggers are the windows the search wrote at
its single threshold; their statistic and the scale their tiles are read on
are re-expressed on the noise of the neighbouring blocks, and the detector
stage joins them at the trigger level (`wdf.analysis.detector_graph`): two
triggers are one event when their energy is close in time, in bands that touch
and continuous in coefficient energy. Every event goes to coincidence:

    triggers --> local noise scale --> detector graph --> events
             --> reconstruction: energy and instant --> wavegrams
             --> network graph, per stretch of common time --> rankings
             --> false-alarm rates from time slides --> lag of each pair

Each event carries two statistics at the cluster level. `EnWDF` is what the
event is worth: the norm of the waveform its own windows invert to, stitched so
that each sample counts once, on the noise scale. `EnWDF_window` is what
selects it: its loudest block. A pair ranks on its two loudest blocks, on the
coherent energy of the tiles the two events share (`network_morphology`) or on
a learned score read off the graph, never on the norm over the whole event,
which carries the threshold's floor once per tile.

The network stage is `wdf.analysis.network_graph` on the pairs
`wdf.analysis.robust_events.IndexedCoincidenceFinder` admits: the two events'
stretches of time meeting within the light travel time widened by their own
spreads. The background is drawn by `TimeSlideFAR` inside every stretch in
which both detectors were searched, since a slide must wrap inside data that
exists, and the stretches' backgrounds are pooled with their livetimes. A
learned ranking is applied to the zero lag and to every slide alike, where each
graph is formed, so its rate is measured on the population it ranks.

Every released pair is then timed on its two reconstructions: the lag that
maximises their cross-correlation (`wdf.analysis.timing`). The light travel
time between the two sites is the only physical constraint a pair has and the
lag is what it constrains, so a pair is `physical` when the lag lies within the
light travel time widened by the width the correlation declares. The
separation of the two events' tile centres is not a second test: a tile
carries its own width, tens of milliseconds low in the band. The lag is
measured on the zero-lag pairs and not on the slides, so neither it nor the
flag enters a rate: a rate read on the admitted pairs is an upper bound on the
rate of the physical ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wdf.analysis.cluster_coefficients import (ClusterCoefficients,
                                               iter_cluster_coefficients)
from wdf.analysis.detector_graph import (WAVEGRAM_TIME_BINS, DetectorGraphConfig,
                                         build_detector_graph, detector_events,
                                         event_coefficients, event_tiles)
from wdf.analysis.detectors import light_travel_time
from wdf.analysis.injections import candidate_spans
from wdf.analysis.network_graph import (TriggerGraphBuilder,
                                        WavegramCoincidenceFinder)
from wdf.analysis.ridge import event_ridge, ridge_features
from wdf.analysis.robust_events import (CoincidenceConfig, FARConfig,
                                        IndexedCoincidenceFinder, TimeSlideFAR)
from wdf.analysis.scale import local_noise_scale, on_local_scale
from wdf.analysis.timing import MAX_LAG_S, arrival_time_difference, envelope_instant

#: The per-event columns a released candidate carries from each of its two
#: events, suffixed `_i` and `_j`.
EVENT_COLUMNS = ["gpsStart", "gpsPeak", "gpsEnvelope", "duration", "freqMin",
                 "freqQ05", "freqMean", "freqQ95", "freqMax", "EnWDF",
                 "EnWDF_window", "snrPeak", "n_pixels", "n_triggers"]

#: What a slide keeps of every accidental pair beside the rankings: which two
#: events it paired, in which slide and stretch.
BACKGROUND_COLUMNS = ["node_i", "node_j", "slide_index", "interval"]


@dataclass
class TriggerReleaseConfig:
    """What the detector and network stages are told.

    :param window: analysis window of the search, samples of the analysed
        stream.
    :param overlap: overlap between consecutive windows, samples.
    :param local_scale_neighbours: blocks the noise scale of each block is read
        over, as `wdf.analysis.scale.local_noise_scale`; None keeps each
        block's own. The search statistic and the tiles' scale are both read
        there before anything is grouped.
    :param detector_graph: when two triggers are one event; `time_tolerance`
        is a fraction of the two windows' mean span, not a time.
    :param wavegram_time_bins: columns of an event's compact map, the node
        feature of the network graph.
    :param coincidence: when two events of two detectors may be one signal.
    :param intra_ifo_window_s: how far apart two events of one detector may be
        and still be neighbours in the network graph, seconds.
    :param match_wavegrams: whether every admitted pair's two renderings are
        compared at the displacements its tolerance allows. It costs one
        correlation profile per pair and per displacement, on the zero lag and
        on every slide, and it feeds `network_correlation` and
        `coherent_statistic`, which are identically zero when it is off.
    :param slides: how the accidental background is drawn, per stretch of
        common time.
    :param ranking: the network statistics a false-alarm rate is attached to;
        the released list is ordered on the first. Each must be a column the
        graph's candidate table carries, or one the scorer adds.
    :param minimum_interval_s: shortest stretch of common search time a
        background is drawn on; shorter ones hold too few distinct slides and
        are left out of the zero lag and the livetime alike.
    :param timed_candidates: how many released pairs, from the top of the
        list, are timed on their reconstructions; all when None.
    :param max_lag_s: half-width of the lag search, seconds.
    """

    window: int = 512
    overlap: int = 32
    local_scale_neighbours: int | None = 41
    detector_graph: DetectorGraphConfig = field(default_factory=DetectorGraphConfig)
    wavegram_time_bins: int = WAVEGRAM_TIME_BINS
    coincidence: CoincidenceConfig = field(default_factory=CoincidenceConfig)
    intra_ifo_window_s: float = 5.0
    match_wavegrams: bool = False
    slides: FARConfig = field(default_factory=lambda: FARConfig(
        n_slides=500, min_shift_s=10.0))
    ranking: tuple = ("network_morphology",)
    minimum_interval_s: float = 600.0
    timed_candidates: int | None = None
    max_lag_s: float = MAX_LAG_S


@dataclass
class TriggerStage:
    """One detector's events, and everything they were built from.

    :param ifo: the detector.
    :param triggers: the triggers in time order, with `EnWDF` and `sigma` on
        the local noise scale and the search's own kept as `EnWDF_search` and
        `sigma_search`.
    :param graph: the detector graph over them; its nodes are the triggers.
    :param labels: the event of every node.
    :param events: one row per event, as `detector_events` builds it, with
        `EnWDF` measured on the reconstruction, the graph's estimate of it
        kept as `EnWDF_sum`, and `gpsEnvelope`, the event's own instant.
    :param maps: `{cluster_id: EventWavegram}` for assembly.
    :param comparison: the same events rendered for the network comparison.
    :param series: `{cluster_id: (gps_start, samples)}`, each event's
        stitched reconstruction, in absolute time.
    :param window: analysis window, samples.
    :param overlap: overlap, samples.
    """

    ifo: str
    triggers: pd.DataFrame
    graph: object
    labels: np.ndarray
    events: pd.DataFrame
    maps: dict
    comparison: dict
    series: dict
    window: int
    overlap: int

    def members(self, cluster_id: int) -> np.ndarray:
        """The rows of `graph.nodes` an event is made of, in time order."""
        return np.flatnonzero(self.labels == int(cluster_id))

    def tiles(self, cluster_id: int) -> pd.DataFrame:
        """Every coefficient the event's windows kept, as tiles on the plane.

        :type cluster_id: int
        :param cluster_id: the event.
        :return: pandas.DataFrame -- `t_lo`, `t_hi`, `f_lo`, `f_hi`, `snr`
            (the amplitude on its window's noise scale, `|c| / sigma`) and
            `amplitude` (the same, signed); a region two overlapping windows
            both kept appears once per window.
        """
        t_lo, t_hi, f_lo, f_hi, energy, amplitude = event_tiles(
            self.graph.nodes, self.members(cluster_id))
        return pd.DataFrame(dict(t_lo=t_lo, t_hi=t_hi, f_lo=f_lo, f_hi=f_hi,
                                 snr=np.sqrt(energy), amplitude=amplitude))

    def coefficients(self, cluster_id: int) -> ClusterCoefficients:
        """The event's windows, every coefficient each of them kept.

        :type cluster_id: int
        :param cluster_id: the event.
        :return: ClusterCoefficients -- whose `reconstruct` is the event's
            waveform and `enwdf` its statistic.
        :raises KeyError: if the event holds no window.
        """
        members = self.graph.nodes.iloc[self.members(cluster_id)]
        if members.empty:
            raise KeyError(f"{self.ifo} has no event {cluster_id}")
        return ClusterCoefficients.from_triggers(
            members, float(members["fs"].iloc[0]), self.window, self.overlap,
            cluster_id=int(cluster_id))

    def ridge(self, cluster_id: int, n_bins: int = 32):
        """The track the event's tiles trace, and its descriptors.

        :type cluster_id: int
        :param cluster_id: the event.
        :type n_bins: int
        :param n_bins: time bins the event's extent is divided into.
        :return: tuple -- `(time, log_frequency, energy, features)`: the ridge
            as `wdf.analysis.ridge.event_ridge` measures it, one tile per bin,
            and its `ridge_features`, symmetric in the direction of the sweep.
        """
        tiles = self.tiles(cluster_id)
        time, log_f, energy = event_ridge(tiles.t_lo, tiles.t_hi, tiles.f_lo,
                                          tiles.f_hi, tiles.snr ** 2, n_bins=n_bins)
        return time, log_f, energy, ridge_features(time, log_f, energy)

    def around(self, gps: float, window_s: float) -> pd.DataFrame:
        """The triggers whose surviving tiles come within `window_s` of an instant.

        :type gps: float
        :param gps: the instant.
        :type window_s: float
        :param window_s: seconds either side of it.
        :return: pandas.DataFrame -- those rows of `graph.nodes`, with the
            event each belongs to as `cluster_id`.
        """
        nodes = self.graph.nodes
        start = nodes["gpsStart"].to_numpy(dtype=float)
        end = start + nodes["duration"].to_numpy(dtype=float)
        near = (end >= float(gps) - window_s) & (start <= float(gps) + window_s)
        return nodes[near].assign(cluster_id=self.labels[near])


def prepare_triggers(triggers: pd.DataFrame, neighbours: int | None) -> pd.DataFrame:
    """The triggers on the scale every later stage reads them on.

    A block's own noise scale is measured on the data it holds, signal
    included, so a transient loud enough to matter divides itself by a scale
    it inflated. The statistic and the scale the tiles are normalised on are
    both read on the neighbouring blocks instead
    (`wdf.analysis.scale.on_local_scale`); the search's own are kept as
    `EnWDF_search` and `sigma_search`, since those are what its threshold was
    applied to.

    :type triggers: pandas.DataFrame
    :param triggers: one detector's triggers, carrying `gps`, `sigma`, `EnWDF`
        and the coefficient columns.
    :type neighbours: int | None
    :param neighbours: blocks the local scale is read over; None keeps each
        block's own scale.
    :return: pandas.DataFrame -- the triggers in time order with a fresh index.
    """
    out = triggers.sort_values("gps", kind="stable").reset_index(drop=True)
    out = out.assign(EnWDF_search=out["EnWDF"].to_numpy(dtype=float),
                     sigma_search=out["sigma"].to_numpy(dtype=float))
    if neighbours is None or out.empty:
        return out
    return out.assign(
        EnWDF=on_local_scale(out, neighbours=int(neighbours)),
        sigma=local_noise_scale(out, neighbours=int(neighbours)))


def trigger_stage(triggers: pd.DataFrame, config: TriggerReleaseConfig,
                  comparison_bin_s: float) -> TriggerStage:
    """One detector's events, measured on their reconstructions.

    :type triggers: pandas.DataFrame
    :param triggers: the detector's triggers, as the search wrote them.
    :type config: TriggerReleaseConfig
    :param config: the run's configuration.
    :type comparison_bin_s: float
    :param comparison_bin_s: column of the map two detectors are compared on,
        seconds; of the order of the network's light travel time, or a real
        delay moves no cell.
    :return: TriggerStage
    """
    prepared = prepare_triggers(triggers, config.local_scale_neighbours)
    ifo = str(prepared["ifo"].iloc[0]) if "ifo" in prepared and len(prepared) else ""
    graph = build_detector_graph(prepared, config=config.detector_graph)
    labels = graph.components()
    events = detector_events(graph, labels=labels)
    fs = float(graph.nodes["fs"].iloc[0]) if len(graph.nodes) else 1.0
    block_s = config.window / fs

    # One inversion serves three readings: the statistic that measures the
    # event, its own instant and the waveform a pair is timed on. The instant
    # is sought within one block of the tile the event was ranked on, so a
    # long transient's own energy cannot pull it away from the feature the
    # search selected it for.
    value, instant, series = {}, {}, {}
    peak_of = dict(zip(events["cluster_id"].astype(int),
                       events["gpsPeak"].astype(float)))
    labelled = graph.nodes.assign(cluster_id=labels)
    for label, cluster in iter_cluster_coefficients(labelled, events, fs,
                                                    config.window, config.overlap):
        value[label] = cluster.enwdf()
        series[label] = cluster.reconstruct()
        instant[label] = envelope_instant(series[label], peak_of[label], fs, block_s)
    measured = events["cluster_id"].map(value).to_numpy(dtype=float)
    refined = events["cluster_id"].map(instant).to_numpy(dtype=float)
    events = events.assign(
        EnWDF_sum=events["EnWDF"].to_numpy(dtype=float),
        EnWDF=np.where(np.isfinite(measured), measured,
                       events["EnWDF_window"].to_numpy(dtype=float)),
        # Where the envelope could not be read the tile centre answers, so the
        # column always carries the best instant the event has.
        gpsEnvelope=np.where(np.isfinite(refined), refined,
                             events["gpsPeak"].to_numpy(dtype=float)))
    maps = event_coefficients(graph, labels, time_bins=config.wavegram_time_bins)
    comparison = event_coefficients(graph, labels, time_bins=config.wavegram_time_bins,
                                    bin_seconds=comparison_bin_s)
    return TriggerStage(ifo=ifo, triggers=prepared, graph=graph, labels=labels,
                        events=events, maps=maps, comparison=comparison,
                        series=series, window=int(config.window),
                        overlap=int(config.overlap))


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


def check_scorer(scorer, graph) -> None:
    """Refuse a learned ranking that was fitted on another representation.

    A model reads a fixed number of features per node and per edge; a graph
    rendered on another band ladder, window or detector set has other widths,
    and a model that loaded would score it without complaint on inputs it was
    never fitted on.

    :param scorer: a `wdf.analysis.gnn.GNNCoincidenceScorer`.
    :param graph: the `TriggerGraph` it is to score.
    :return: None
    :raises ValueError: if the node or the edge width differs from the model's.
    """
    node = int(scorer.encoder[0].in_features)
    hidden = int(scorer.encoder[0].out_features)
    edge = int(scorer.edge_head[0].in_features) - 2 * hidden - int(scorer.profile_dim)
    have = (int(graph.node_features.shape[1]), int(graph.cross_edge_features.shape[1]))
    if have != (node, edge):
        raise ValueError(
            f"the model reads {node} node and {edge} edge features and this graph "
            f"carries {have[0]} and {have[1]}: it was fitted on another "
            "representation (window, band ladder or detectors)")


def _within(events: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    start = events["gpsStart"].to_numpy(dtype=float)
    return events[(start >= lo) & (start < hi)].reset_index(drop=True)


def network_stage(stages: dict, intervals: list, config: TriggerReleaseConfig,
                  scorer=None):
    """The zero-lag candidates and their accidental background, per stretch.

    :type stages: dict
    :param stages: `{ifo: TriggerStage}`.
    :type intervals: list
    :param intervals: what `common_intervals` returns.
    :type config: TriggerReleaseConfig
    :param config: the run's configuration.
    :param scorer: a learned ranking applied to every graph formed, zero lag
        and slides alike, or None for the graph's own columns alone.
    :return: tuple -- `(candidates, background, livetime_s, observed_s)`: the
        admitted pairs of every stretch, carrying the detector and the
        `cluster_id` of both events; the accidental pairs of every stretch's
        slides, reduced to the rankings and `BACKGROUND_COLUMNS`; the slid
        livetime; and the zero-lag time searched.
    :raises ValueError: if a stretch cannot hold the slides asked for, if a
        ranking is not a column of the candidates, or if the scorer was fitted
        on another representation.
    """
    candidates, background = [], []
    livetime, observed = 0.0, 0.0
    for number, (lo, hi, ifos) in enumerate(intervals):
        events = {ifo: _within(stages[ifo].events, lo, hi) for ifo in ifos}
        if any(frame.empty for frame in events.values()):
            continue
        maps = {ifo: stages[ifo].maps for ifo in ifos}
        comparison = {ifo: stages[ifo].comparison for ifo in ifos}
        builder = TriggerGraphBuilder(intra_ifo_window_s=config.intra_ifo_window_s,
                                      coincidence=config.coincidence, ifos=list(ifos),
                                      wavegram_time_bins=config.wavegram_time_bins,
                                      match_wavegrams=config.match_wavegrams)
        prepared = builder.prepare(events, maps, comparison=comparison)
        graph = builder.build_from_prepared(events, prepared)
        if len(graph.cross_edges):
            if scorer is not None:
                check_scorer(scorer, graph)
            table = graph.candidate_table() if scorer is None else scorer.score(graph)
            missing = [s for s in config.ranking if s not in table]
            if missing:
                raise ValueError(f"no column {missing} to rank on; a learned "
                                 "ranking needs its scorer")
            nodes = graph.nodes
            for side in ("i", "j"):
                node = table[f"node_{side}"].to_numpy(dtype=int)
                table[f"ifo_{side}"] = nodes["ifo"].to_numpy()[node]
                table[f"cluster_{side}"] = nodes["cluster_id"].to_numpy(dtype=int)[node]
            candidates.append(table.assign(interval=number))
        finder = WavegramCoincidenceFinder(
            IndexedCoincidenceFinder(config.coincidence), builder, maps,
            comparison=comparison, prepared=prepared, scorer=scorer)
        kept = []
        columns = list(config.ranking) + BACKGROUND_COLUMNS

        def reduce(table, _livetime, kept=kept, number=number):
            # A rate is an order statistic of a float: the slide keeps the
            # rankings, in single precision, and which pair each value belongs
            # to, and releases everything else as it is formed.
            small = table.assign(interval=number)[[c for c in columns if c in table
                                                   or c == "interval"]].copy()
            for column in config.ranking:
                small[column] = small[column].astype(np.float32)
            kept.append(small)

        try:
            slid = TimeSlideFAR(finder, config.slides).background_distribution(
                events, {ifo: (lo, hi) for ifo in ifos}, reduce=reduce)
        except ValueError as problem:
            raise ValueError(
                f"no background can be drawn on {lo:.0f}-{hi:.0f} "
                f"({','.join(ifos)}): {problem}") from problem
        livetime += float(slid.attrs["total_livetime_s"])
        observed += float(hi - lo)
        if kept:
            background.append(pd.concat(kept, ignore_index=True))
    candidates = (pd.concat(candidates, ignore_index=True) if candidates
                  else pd.DataFrame())
    background = (pd.concat(background, ignore_index=True) if background
                  else pd.DataFrame(columns=list(config.ranking) + BACKGROUND_COLUMNS))
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
        `far_per_day_<s>`, `fap_<s>` and `n_background_ge_<s>`, ordered on the
        first statistic's rate and then on the statistic itself, with `rank`
        the position in that order, from one.
    """
    if candidates.empty:
        return candidates.copy()
    out = candidates.reset_index(drop=True).assign(_row=np.arange(len(candidates)))
    ranker = TimeSlideFAR(None)
    for statistic in statistics:
        ranked = ranker.rank_candidates(out[["_row", statistic]], background,
                                        observed_s, score_column=statistic)
        ranked = ranked.set_index("_row").reindex(out["_row"])
        out[f"far_per_day_{statistic}"] = ranked["far_per_day"].to_numpy()
        out[f"fap_{statistic}"] = ranked["fap"].to_numpy()
        out[f"n_background_ge_{statistic}"] = ranked["n_background_ge"].to_numpy()
    first = statistics[0]
    out = (out.sort_values([f"far_per_day_{first}", first], ascending=[True, False])
           .drop(columns="_row").reset_index(drop=True))
    return out.assign(rank=np.arange(1, len(out) + 1))


def with_events(candidates: pd.DataFrame, stages: dict) -> pd.DataFrame:
    """The candidates, with what each of their two events measured.

    :type candidates: pandas.DataFrame
    :param candidates: pairs carrying `ifo_i`, `cluster_i`, `ifo_j` and
        `cluster_j`.
    :type stages: dict
    :param stages: `{ifo: TriggerStage}`.
    :return: pandas.DataFrame -- `candidates` with `EVENT_COLUMNS` suffixed
        `_i` and `_j`, and the pair's own extent as `gpsStart` and `duration`,
        the union of the two events' extents, which is what a candidate is
        matched in time on.
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


def timed(candidates: pd.DataFrame, stages: dict, max_lag_s: float = MAX_LAG_S,
          limit: int | None = None) -> pd.DataFrame:
    """The released pairs, each with the lag its two reconstructions measure.

    The lag is the cross-correlation lag of the two events' stitched
    reconstructions on one absolute time grid, read below the tile
    (`wdf.analysis.timing.arrival_time_difference`), with the width the
    correlation peak declares. A pair is `physical` when that lag lies within
    the light travel time between its two sites widened by that width; the
    separation of the two events' tile centres is not a second test. Where the
    two reconstructions sit further apart than the lag search reaches, the lag
    is the difference of the events' own instants. Neither the lag nor the
    flag gates a pair.

    :type candidates: pandas.DataFrame
    :param candidates: pairs carrying `ifo_i`, `cluster_i`, `ifo_j`,
        `cluster_j` and `dt_s`, in the order they are to be timed.
    :type stages: dict
    :param stages: `{ifo: TriggerStage}`, holding the reconstructions.
    :type max_lag_s: float
    :param max_lag_s: half-width of the lag search, seconds.
    :type limit: int | None
    :param limit: time only this many pairs, from the top; the others carry no
        lag and `physical` is left unset. All when None.
    :return: pandas.DataFrame -- the candidates with `lag_s`, `lag_sigma_s`,
        `lag_from`, `light_travel_s` and `physical`.
    """
    out = candidates.copy()
    if out.empty:
        return out.assign(lag_s=[], lag_sigma_s=[], lag_from=[],
                          light_travel_s=[], physical=[])
    count = len(out) if limit is None else min(int(limit), len(out))
    lag = out["dt_s"].to_numpy(dtype=float).copy()
    lag[count:] = np.nan
    sigma = np.full(len(out), np.nan)
    source = np.full(len(out), "instants", dtype=object)
    source[count:] = None
    fs = {ifo: float(stage.graph.nodes["fs"].iloc[0]) for ifo, stage in stages.items()}
    pairs = zip(out["ifo_i"].iloc[:count], out["cluster_i"].iloc[:count],
                out["ifo_j"].iloc[:count], out["cluster_j"].iloc[:count])
    for row, (ifo_i, cluster_i, ifo_j, cluster_j) in enumerate(pairs):
        first = stages[ifo_i].series.get(int(cluster_i))
        second = stages[ifo_j].series.get(int(cluster_j))
        if first is None or second is None:
            continue
        try:
            lag[row], sigma[row] = arrival_time_difference(first, second, fs[ifo_i],
                                                           max_lag_s)
        except ValueError:
            continue
        source[row] = "correlation"
    travel = np.array([light_travel_time(a, b)
                       for a, b in zip(out["ifo_i"], out["ifo_j"])])
    physical = pd.array(np.abs(lag) <= travel + np.nan_to_num(sigma), dtype="boolean")
    physical[count:] = pd.NA
    return out.assign(lag_s=lag, lag_sigma_s=sigma, lag_from=source,
                      light_travel_s=travel, physical=physical)


@dataclass
class TriggerRelease:
    """What the search released, and everything it was read from.

    :param config: the configuration the release was made under.
    :param stages: `{ifo: TriggerStage}`.
    :param spans: `{ifo: [(start, end), ...]}`, what each detector searched.
    :param intervals: the stretches of common search time, as
        `common_intervals` returns.
    :param candidates: the released pairs, ordered on the first ranking
        statistic's false-alarm rate, `rank` being the position from one.
    :param background: the pooled accidental pairs.
    :param livetime_s: the slid livetime the rates are read over.
    :param observed_s: the zero-lag time searched in coincidence.
    """

    config: TriggerReleaseConfig
    stages: dict
    spans: dict
    intervals: list
    candidates: pd.DataFrame
    background: pd.DataFrame
    livetime_s: float
    observed_s: float

    def counts(self) -> pd.DataFrame:
        """How many triggers and events each detector holds, and how many pairs.

        :return: pandas.DataFrame -- one row per detector: the time it searched,
            its triggers, its events, and the zero-lag pairs it takes part in.
        """
        rows = []
        for ifo, stage in self.stages.items():
            involved = int(((self.candidates.get("ifo_i") == ifo)
                            | (self.candidates.get("ifo_j") == ifo)).sum()) \
                if len(self.candidates) else 0
            rows.append(dict(ifo=ifo, searched_s=sum(b - a for a, b in self.spans[ifo]),
                             triggers=len(stage.triggers), events=len(stage.events),
                             pairs=involved))
        return pd.DataFrame(rows)

    def at(self, gps: float, window_s: float) -> pd.DataFrame:
        """The released pairs whose extent comes within `window_s` of an instant.

        :type gps: float
        :param gps: the instant.
        :type window_s: float
        :param window_s: seconds either side of the pair's extent.
        :return: pandas.DataFrame -- those rows of `candidates`, in rank order.
        """
        if self.candidates.empty:
            return self.candidates.copy()
        start, end = candidate_spans(self.candidates, candidate_time="gps_candidate")
        near = (end >= float(gps) - window_s) & (start <= float(gps) + window_s)
        return self.candidates[near]

    def events_near(self, gps: float, window_s: float) -> pd.DataFrame:
        """The events holding a trigger within reach of an instant, in every detector.

        :type gps: float
        :param gps: the instant.
        :type window_s: float
        :param window_s: seconds either side of it a trigger's tiles may reach.
        :return: pandas.DataFrame -- one row per event: `ifo`, `cluster_id`,
            the start and end of its tiles as `start_s` and `end_s`, seconds
            from `gps`; `EnWDF_window`, `EnWDF`, `freqQ05`, `freqQ95`,
            `n_triggers`; `pairs`, the released pairs it takes part in, and
            the best `rank` among them.
        """
        rows = []
        for ifo, stage in self.stages.items():
            clusters = set(stage.around(gps, window_s)["cluster_id"].astype(int))
            held = stage.events[stage.events["cluster_id"].astype(int).isin(clusters)]
            for event in held.itertuples():
                mine = (self.candidates[
                    ((self.candidates["ifo_i"] == ifo)
                     & (self.candidates["cluster_i"].astype(int) == int(event.cluster_id)))
                    | ((self.candidates["ifo_j"] == ifo)
                       & (self.candidates["cluster_j"].astype(int) == int(event.cluster_id)))]
                    if len(self.candidates) else self.candidates)
                rows.append(dict(
                    ifo=ifo, cluster_id=int(event.cluster_id),
                    start_s=float(event.gpsStart) - float(gps),
                    end_s=float(event.gpsStart + event.duration) - float(gps),
                    EnWDF_window=float(event.EnWDF_window), EnWDF=float(event.EnWDF),
                    freqQ05=float(event.freqQ05), freqQ95=float(event.freqQ95),
                    n_triggers=int(event.n_triggers), pairs=len(mine),
                    best_rank=float(mine["rank"].min()) if len(mine) else np.nan))
        return pd.DataFrame(rows, columns=[
            "ifo", "cluster_id", "start_s", "end_s", "EnWDF_window", "EnWDF", "freqQ05",
            "freqQ95", "n_triggers", "pairs", "best_rank"])

    def trace(self, gps: float, window_s: float) -> pd.DataFrame:
        """What each stage holds around an instant, detector by detector.

        Where a transient is in the data and no candidate stands for it, the
        stage that lost it is the first at which it is missing: no trigger
        near it, the triggers joined into events the pairs do not reach, or
        pairs formed and ranked low. `events_near` lists the events one by
        one.

        :type gps: float
        :param gps: the instant.
        :type window_s: float
        :param window_s: seconds either side of it.
        :return: pandas.DataFrame -- one row per detector: `triggers`, the
            windows whose tiles come within reach, and the loudest of them on
            the search's own statistic and on the local scale; `events`, the
            events holding those windows, and the largest `EnWDF_window` and
            `EnWDF` among them; `pairs`, the released pairs one of those events
            takes part in, and the best `rank` among them.
        """
        listed = self.events_near(gps, window_s)
        rows = []
        for ifo, stage in self.stages.items():
            near = stage.around(gps, window_s)
            held = listed[listed["ifo"] == ifo]
            rows.append(dict(
                ifo=ifo, triggers=len(near),
                loudest_search=float(near["EnWDF_search"].max()) if len(near) else np.nan,
                loudest_local=float(near["EnWDF"].max()) if len(near) else np.nan,
                events=len(held),
                loudest_window=float(held["EnWDF_window"].max()) if len(held) else np.nan,
                loudest_event=float(held["EnWDF"].max()) if len(held) else np.nan,
                pairs=int(held["pairs"].sum()),
                best_rank=float(held["best_rank"].min()) if held["pairs"].sum() else np.nan))
        return pd.DataFrame(rows)


def release(triggers: dict, spans: dict, config: TriggerReleaseConfig | None = None,
            scorer=None) -> TriggerRelease:
    """The network's released candidates, from each detector's triggers.

    :type triggers: dict
    :param triggers: `{ifo: triggers}`, each as
        `wdf.analysis.io.triggers_from_files` reads them, at one window length
        and one threshold. Every detector given enters the network stage.
    :type spans: dict
    :param spans: `{ifo: [(start, end), ...]}`, the stretches each detector's
        search covered.
    :type config: TriggerReleaseConfig | None
    :param config: the configuration; the default one when None.
    :param scorer: a learned ranking applied to every graph formed, or None.
    :return: TriggerRelease -- its candidates timed by `timed`.
    """
    config = TriggerReleaseConfig() if config is None else config
    ifos = list(triggers)
    comparison_bin_s = 2.0 * max(light_travel_time(a, b) for a in ifos for b in ifos
                                 if a != b)
    stages = {ifo: trigger_stage(frame.assign(ifo=ifo), config, comparison_bin_s)
              for ifo, frame in triggers.items()}
    intervals = common_intervals({ifo: spans[ifo] for ifo in ifos},
                                 minimum_s=config.minimum_interval_s)
    candidates, background, livetime, observed = network_stage(
        stages, intervals, config, scorer=scorer)
    ranked = rank(candidates, background, observed, config.ranking)
    ranked = timed(with_events(ranked, stages), stages, config.max_lag_s,
                   config.timed_candidates)
    return TriggerRelease(config=config, stages=stages, spans=spans,
                          intervals=intervals, candidates=ranked,
                          background=background, livetime_s=livetime,
                          observed_s=observed)
