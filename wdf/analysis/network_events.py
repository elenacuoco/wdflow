"""Network events: the detector events above a threshold, joined across
detectors in the time domain by a matched filter of their reconstructions.

The stage of the offline flavour that turns the detectors' clustered events
into network nodes. It reads, per detector, the events of stage 2
(`wdf.analysis.detector_graph`, one row per connected component of the
detector graph), their energy measured on the stitched reconstruction
(`score_events_by_reconstruction`), the whitened stream divided by the sigma
of the noise model in force, and that model history (`hrss.ModelHistory`).

1. **References.** A detector event is a reference when its `EnWDF` (the
   norm of its stitched reconstruction in noise units) is at least
   `EventThreshold` and it is not marked ``guided``. Its template is its
   time-domain reconstruction (`event_waveform`) divided by the sigma of the
   model in force at each sample; ``rho_own`` is the template's norm.
2. **Matched filter.** Every reference is filtered in every other detector
   searched at its time, over the lags ``|tau|`` up to the pair's light
   travel time plus `NetworkEventsJitter`, by `NetworkEventsLagStep` samples:
   ``rho(tau) = |<h, s(t + tau)> - i <H h, s(t + tau)>| / ||h||`` with ``H``
   the Hilbert transform (`analytic_matched_filter`). In white noise of unit
   variance ``rho`` at one lag is Rayleigh of scale 1. The best lag gives
   ``rho_B``, ``tau`` (``t_B - t_A``) and the two amplitudes of the
   least-squares fit ``s ~ amp_re h + amp_im H h`` (`match_template`).
3. **Nodes.** The support of an event is the interval where the analytic
   envelope of its reconstruction is at least `NetworkEventsSupportFraction`
   of its peak (`reconstruction_support`). Two references of different
   detectors are one node when the louder one's support, shifted by its best
   ``tau`` in the other detector, meets the other's support
   (`template_meets`). Pairs are joined greedily by
   ``sqrt(rho_own_A^2 + rho_own_B^2)``, one event per detector, every cross
   pair of a joined node meeting the same way (`coincident_nodes`); every
   reference left alone is a node of its own. The node's reference is its
   loudest event (largest ``rho_own``).
4. **Network energy.** ``network_energy = sqrt(rho_ref^2 + sum_B rho_B^2)``,
   ``rho_B`` the matched filter of the reference's template in every other
   detector searched at its time, whether or not that detector has an event
   in the node. ``kind`` is "coincident" (events above the threshold in two
   or more detectors) or "single+MF" (one); it is a label only.
5. **Checks, never cuts.** Per other detector: its own event that meets the
   shifted support (the node's event there, else the loudest of any `EnWDF`
   within the light travel time plus the jitter), its `EnWDF`, band, the
   difference of the envelope instants against ``tau`` and ``on_track``, the
   fraction of its tile energy on tiles meeting the reference's tiles
   shifted by ``tau`` with bands widened by `NetworkEventsOnTrackOctaves`
   (`on_track_fraction`); per pair the lag of the matched filter against
   that of the envelope instants; the template's duration and band, marked
   ``template_short`` under `NetworkEventsTemplateShort`.

No significance is computed here: the network energy is the statistic a
later background (event-level time slides) is measured on. Every default is
in the table ``[network_events]`` of wdf/config/defaults.toml.

**Background by time slides.** `prepare_network_events` reads what does not
depend on time: the events, their supports, the references and their
templates. `PreparedNetwork.run(shifts)` then builds the matched filters and
the nodes with each detector's whitened stream, events and reconstructions
moved by its shift (``{ifo: s}``, rounded to a sample, circular within each
of the detector's streams), and tags the table with ``shift_<ifo>``;
`PreparedNetwork.slides` does it for several sets of shifts. Stage 1 is not
redone. With no shift the table is the one `network_events` gives.

`network_events` runs the stage on prepared inputs (`DetectorInput`);
`network_events_from_stage` prepares them from the outputs of stage 2;
`run_network` calls it when `NetworkEvents` is true. `whitened_stream`
rebuilds a run's whitened stream from its raw strain, as the worker
whitened it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wdf.config.defaults import defaults, parameter

_NE = defaults("network_events")
SECTION = "network_events"

#: The node kinds.
COINCIDENT, SINGLE = "coincident", "single+MF"
#: Per-detector columns of the node table, ``<ifo>_<name>``.
PER_DETECTOR = ("source", "E", "cluster", "start", "duration", "rec_start", "rec_end", "f_min",
                "f_max", "f_mean", "EnWDF", "rho_own", "envelope", "rho_B", "tau", "rho0_B",
                "amp_re", "amp_im", "tile_n", "tile_cluster", "tile_EnWDF", "tile_above",
                "tile_guided", "tile_start", "tile_duration", "tile_rec_start", "tile_rec_end",
                "tile_f_min", "tile_f_max", "tile_dt", "tile_dt_diff", "tile_on_track")
#: Node columns before the per-detector ones.
NODE_COLUMNS = ("kind", "label", "rank", "n_nodes", "network_energy", "network_energy_tiles",
                "ref", "ref_cluster", "n_event_detectors", "template_duration_s",
                "template_f_min", "template_f_max", "template_short")
#: Per-pair columns, ``<name>_<ifo_x><ifo_y>`` (``t_y - t_x``).
PER_PAIR = ("dt", "dt_env", "dt_diff")


@dataclass
class NetworkEventsSettings:
    """Every choice of the stage (table ``[network_events]``).

    :param active: `NetworkEvents`: `run_network` runs the stage.
    :param event_threshold: `EventThreshold`: least `EnWDF` of a reference.
    :param jitter_s: `NetworkEventsJitter`, s added to the light travel time.
    :param lag_step: `NetworkEventsLagStep`, samples between two lags.
    :param support_fraction: `NetworkEventsSupportFraction`, envelope level of
        the support over its peak, in (0, 1).
    :param on_track_octaves: `NetworkEventsOnTrackOctaves`, octaves by which
        a reference tile is widened on each side for ``on_track``.
    :param template_short_s: `NetworkEventsTemplateShort`, s.
    :param stream_blocks: `NetworkEventsStreamBlocks`, conditioning blocks
        read beyond the stream wanted on each side (`whitened_stream`).
    :param stream_pad_s: `NetworkEventsStreamPad`, s read beyond those.
    :param stream_margin_s: `NetworkEventsStreamMargin`, s of stream kept
        beyond the searched span and the largest lag.
    """

    active: bool = _NE["NetworkEvents"]
    event_threshold: float = _NE["EventThreshold"]
    jitter_s: float = _NE["NetworkEventsJitter"]
    lag_step: int = _NE["NetworkEventsLagStep"]
    support_fraction: float = _NE["NetworkEventsSupportFraction"]
    on_track_octaves: float = _NE["NetworkEventsOnTrackOctaves"]
    template_short_s: float = _NE["NetworkEventsTemplateShort"]
    stream_blocks: int = _NE["NetworkEventsStreamBlocks"]
    stream_pad_s: float = _NE["NetworkEventsStreamPad"]
    stream_margin_s: float = _NE["NetworkEventsStreamMargin"]

    #: The run parameter each field is read from.
    KEYS = dict(active="NetworkEvents", event_threshold="EventThreshold",
                jitter_s="NetworkEventsJitter", lag_step="NetworkEventsLagStep",
                support_fraction="NetworkEventsSupportFraction",
                on_track_octaves="NetworkEventsOnTrackOctaves",
                template_short_s="NetworkEventsTemplateShort",
                stream_blocks="NetworkEventsStreamBlocks", stream_pad_s="NetworkEventsStreamPad",
                stream_margin_s="NetworkEventsStreamMargin")

    def __post_init__(self):
        if not 0.0 < float(self.support_fraction) < 1.0:
            raise ValueError(f"NetworkEventsSupportFraction {self.support_fraction} must lie in (0, 1)")
        if int(self.lag_step) < 1:
            raise ValueError(f"NetworkEventsLagStep {self.lag_step} must be at least one sample")
        if float(self.jitter_s) < 0.0:
            raise ValueError(f"NetworkEventsJitter {self.jitter_s} must not be negative")

    @classmethod
    def from_parameters(cls, par):
        """The settings a run asks for, defaults filled in and written back.

        :param par: run configuration; it receives every value used.
        :return: NetworkEventsSettings
        """
        values = {f: parameter(par, key, SECTION) for f, key in cls.KEYS.items()}
        out = cls(active=bool(values["active"]), event_threshold=float(values["event_threshold"]),
                  jitter_s=float(values["jitter_s"]), lag_step=int(values["lag_step"]),
                  support_fraction=float(values["support_fraction"]),
                  on_track_octaves=float(values["on_track_octaves"]),
                  template_short_s=float(values["template_short_s"]),
                  stream_blocks=int(values["stream_blocks"]),
                  stream_pad_s=float(values["stream_pad_s"]),
                  stream_margin_s=float(values["stream_margin_s"]))
        for f, key in cls.KEYS.items():
            setattr(par, key, getattr(out, f))
        return out


@dataclass
class DetectorInput:
    """What the stage reads of one detector.

    :param events: every detector event of stage 2 (`detector_events`), with
        `EnWDF` measured on the stitched reconstruction, `cluster_id`,
        `gpsStart`, `duration`, `freqMin`, `freqMax`, `freqMean` and
        `gpsEnvelope`; an optional boolean ``guided`` excludes an event from
        the references.
    :param graph: the detector graph (`build_detector_graph`) the events are
        the components of.
    :param labels: its component labels (``graph.components()``).
    :param history: the noise models the worker applied (`hrss.ModelHistory`).
    :param streams: the whitened stream over sigma, one ``(t0, samples)`` per
        run of the detector.
    :param searched: per stream, the ``(start, end)`` GPS the run searched;
        the stream's own extent when None.
    """

    events: pd.DataFrame
    graph: object
    labels: np.ndarray
    history: object
    streams: list
    searched: list = None

    def __post_init__(self):
        self.streams = [(float(t0), np.asarray(y, dtype=float)) for t0, y in self.streams]
        if self.searched is None:
            self.searched = [None] * len(self.streams)
        if len(self.searched) != len(self.streams):
            raise ValueError("one searched span per stream")

    def run_holding(self, t, fs):
        """Index of the stream whose searched span holds `t`, else None."""
        for k, ((t0, y), span) in enumerate(zip(self.streams, self.searched)):
            a, b = span if span is not None else (t0, t0 + y.size / fs)
            if a <= t <= b:
                return k
        return None


@dataclass
class NetworkEventsResult:
    """What `network_events` produced.

    :param nodes: one row per node, by decreasing network energy (columns
        `node_columns`).
    :param info: the settings used, per pair the light travel time and the
        lag range, per detector the references and ``rho_own / EnWDF``.
    :param references: ``{ifo: the detector's references}`` with
        ``rho_own``, ``rec_start`` and ``rec_end``.
    :param matches: ``{(ifo, row, other): match_template result or None}``.
    """

    nodes: pd.DataFrame
    info: dict = field(default_factory=dict)
    references: dict = field(default_factory=dict)
    matches: dict = field(default_factory=dict)


# ------------------------------------------------------------------ the matched filter
def analytic_matched_filter(segment, template):
    """The analytic matched filter of `template` over the valid lags of
    `segment`.

    :param segment: samples, at least as long as the template.
    :param template: the template ``h``.
    :return: tuple -- (``rho`` per lag, the complex correlation ``c`` with
        ``H = h + i hilbert(h)``, ``H``); ``rho = |c| / ||h||``.
    """
    from scipy.signal import correlate, hilbert

    h = np.asarray(template, dtype=float)
    analytic = hilbert(h)
    c = correlate(np.asarray(segment, dtype=float), analytic, mode="valid", method="fft")
    return np.abs(c) / np.linalg.norm(h), c, analytic


def match_template(template, stream, half_s, step, fs):
    """The best lag of a template in a stream.

    :param template: ``(gps of the first sample, h)``.
    :param stream: ``(gps of the first sample, samples)``.
    :param half_s: largest ``|tau|``, s.
    :param step: samples between two lags tried.
    :param fs: sampling rate, Hz.
    :return: dict -- ``rho`` (best), ``tau`` (s, positive when the stream
        lags the template), ``rho0`` (at tau = 0), ``amp_re``, ``amp_im``
        (``s ~ amp_re h + amp_im hilbert(h)`` at tau); None when the stream
        does not hold the template and every lag, or the template is zero.
    """
    th, h = template
    t0, y = stream
    h = np.asarray(h, dtype=float)
    m = int(np.ceil(float(half_s) * fs))
    s0 = int(round((float(th) - float(t0)) * fs))
    if s0 - m < 0 or s0 + h.size + m > np.asarray(y).size or not np.any(h):
        return None
    rho, c, analytic = analytic_matched_filter(np.asarray(y)[s0 - m:s0 + h.size + m], h)
    lags = m + int(step) * np.arange(-(m // int(step)), m // int(step) + 1)
    k = int(lags[np.argmax(rho[lags])])
    quadrature = analytic.imag
    return dict(rho=float(rho[k]), tau=(k - m) / fs, rho0=float(rho[m]),
                amp_re=float(c[k].real / np.dot(h, h)),
                amp_im=float(-c[k].imag / np.dot(quadrature, quadrature)))


def reconstruction_support(t0, samples, fraction, fs):
    """``(start, end)`` GPS of the samples where the analytic envelope of
    `samples` is at least `fraction` of its peak; NaN for an empty or zero
    reconstruction."""
    from scipy.signal import hilbert

    x = np.asarray(samples, dtype=float)
    envelope = np.abs(hilbert(x)) if x.size else x
    if not envelope.size or envelope.max() <= 0:
        return np.nan, np.nan
    k = np.flatnonzero(envelope >= fraction * envelope.max())
    return float(t0) + k[0] / fs, float(t0) + (k[-1] + 1) / fs


def on_track_fraction(tiles_b, tiles_a, tau, octaves):
    """Fraction of the tile energy of event B on tiles that meet in time a
    tile of A shifted by `tau` whose band, widened by `octaves` on each side,
    meets theirs.

    :param tiles_b: `event_tiles` of B (t_lo, t_hi, f_lo, f_hi, energy, ...).
    :param tiles_a: `event_tiles` of A.
    :return: float -- in [0, 1]; NaN without tiles.
    """
    bl, bh, bfl, bfh, be = (np.asarray(x, dtype=float) for x in tiles_b[:5])
    al, ah, afl, afh = (np.asarray(x, dtype=float) for x in tiles_a[:4])
    if not bl.size or not al.size:
        return np.nan
    widen = 2.0 ** float(octaves)
    hit = ((bl[:, None] < ah[None, :] + tau) & (bh[:, None] > al[None, :] + tau)
           & (bfl[:, None] < afh[None, :] * widen) & (bfh[:, None] > afl[None, :] / widen)).any(axis=1)
    e = np.abs(be)
    return float(e[hit].sum() / e.sum()) if e.sum() > 0 else np.nan


def events_meeting(events, start, end, half):
    """The rows of `events` whose support (`rec_start`, `rec_end`) meets
    ``[start - half, end + half]``, loudest first."""
    if events is None or not len(events):
        return events
    a, b = events["rec_start"].to_numpy(float), events["rec_end"].to_numpy(float)
    return events[(b >= start - half) & (a <= end + half)].sort_values("EnWDF", ascending=False)


# ------------------------------------------------------------------ the nodes
def template_meets(refs, matches, a, b):
    """``[i, j]``: reference i of `a` and reference j of `b` are one node
    when the louder one's support, shifted by its best tau in the other
    detector, meets the other's support; False where the louder one has no
    matched filter there.

    :param refs: ``{ifo: references}`` with ``rho_own``, ``rec_start``,
        ``rec_end``.
    :param matches: ``{(ifo, row, other): match_template result or None}``.
    """
    def shifted(x, y):
        tau = np.array([matches[(x, k, y)]["tau"] if matches.get((x, k, y)) is not None else np.nan
                        for k in range(len(refs[x]))])
        s0 = refs[x]["rec_start"].to_numpy(float) + tau
        s1 = refs[x]["rec_end"].to_numpy(float) + tau
        t0, t1 = refs[y]["rec_start"].to_numpy(float), refs[y]["rec_end"].to_numpy(float)
        return (s1[:, None] >= t0[None, :]) & (s0[:, None] <= t1[None, :])

    louder = refs[a]["rho_own"].to_numpy(float)[:, None] >= refs[b]["rho_own"].to_numpy(float)[None, :]
    return np.where(louder, shifted(a, b), shifted(b, a).T)


def coincident_nodes(refs, ifos, matches):
    """Nodes of two or more detectors, ``[{ifo: row}]``.

    The pairs that meet (`template_meets`) are taken by decreasing
    ``sqrt(rho_own_A^2 + rho_own_B^2)``; two nodes are joined when they share
    no detector and every cross pair meets.
    """
    meet, edges = {}, []
    for a, b in itertools.combinations(ifos, 2):
        if not len(refs[a]) or not len(refs[b]):
            meet[(a, b)] = np.zeros((len(refs[a]), len(refs[b])), bool)
            continue
        m = template_meets(refs, matches, a, b)
        meet[(a, b)] = m
        ea, eb = refs[a]["rho_own"].to_numpy(float), refs[b]["rho_own"].to_numpy(float)
        for i, j in zip(*np.nonzero(m)):
            if np.isfinite(ea[i]) and np.isfinite(eb[j]):
                edges.append((float(np.hypot(ea[i], eb[j])), a, int(i), b, int(j)))
    edges.sort(key=lambda e: -e[0])

    def coincide(x, i, y, j):
        return bool(meet[(x, y)][i, j]) if (x, y) in meet else bool(meet[(y, x)][j, i])

    node_of, nodes = {}, []
    for _, a, i, b, j in edges:
        na, nb = node_of.get((a, i)), node_of.get((b, j))
        if na is not None and na == nb:
            continue
        left = nodes[na] if na is not None else {a: i}
        right = nodes[nb] if nb is not None else {b: j}
        if set(left) & set(right):
            continue
        if not all(coincide(x, p, y, q) for x, p in left.items() for y, q in right.items()):
            continue
        merged = dict(left, **right)
        for n in (na, nb):
            if n is not None:
                nodes[n] = None
        nodes.append(merged)
        for x, p in merged.items():
            node_of[(x, p)] = len(nodes) - 1
    return [n for n in nodes if n]


def node_columns(ifos):
    """The columns of the node table for the detectors `ifos`, in order."""
    cols = list(NODE_COLUMNS)
    cols += [f"{i}_{k}" for i in ifos for k in PER_DETECTOR]
    for x, y in itertools.combinations(ifos, 2):
        cols += [f"{name}_{x}{y}" for name in PER_PAIR]
    return cols


def _node_row(members, refs, every, matches, ifos, kind, half, tiles_of, settings):
    """One node of the table (`network_events`)."""
    ref = max(members, key=lambda i: refs[i].loc[members[i], "rho_own"])
    rr = refs[ref].loc[members[ref]]
    r_start, r_dur = float(rr["gpsStart"]), float(rr["duration"])
    s_start, s_end = float(rr["rec_start"]), float(rr["rec_end"])
    seen = [i for i in ifos if i in members]
    row = dict(kind=kind, label=(f"seen above threshold in {','.join(seen)}" if len(members) > 1
                                 else f"{ref} + matched filter"),
               ref=ref, ref_cluster=int(rr["cluster_id"]), n_event_detectors=len(members),
               template_duration_s=s_end - s_start, template_f_min=float(rr["freqMin"]),
               template_f_max=float(rr["freqMax"]),
               template_short=bool(s_end - s_start < float(settings.template_short_s)))
    energy, tile_energy, tau, envelope = ({ref: float(rr["rho_own"])}, {ref: float(rr["EnWDF"])},
                                          {ref: 0.0}, {})
    for ifo in ifos:
        source = None
        if ifo in members:
            r = refs[ifo].loc[members[ifo]]
            source = "event"
            row.update({f"{ifo}_cluster": int(r["cluster_id"]), f"{ifo}_start": float(r["gpsStart"]),
                        f"{ifo}_duration": float(r["duration"]), f"{ifo}_f_min": float(r["freqMin"]),
                        f"{ifo}_f_max": float(r["freqMax"]), f"{ifo}_f_mean": float(r["freqMean"]),
                        f"{ifo}_EnWDF": float(r["EnWDF"]), f"{ifo}_rho_own": float(r["rho_own"]),
                        f"{ifo}_envelope": float(r["gpsEnvelope"]),
                        f"{ifo}_rec_start": float(r["rec_start"]), f"{ifo}_rec_end": float(r["rec_end"])})
            envelope[ifo] = float(r["gpsEnvelope"])
        if ifo != ref:
            m = matches.get((ref, members[ref], ifo))
            if m is None:
                source = "not searched"
            else:
                source = "event" if ifo in members else "mf"
                energy[ifo], tau[ifo] = m["rho"], m["tau"]
                row.update({f"{ifo}_rho_B": m["rho"], f"{ifo}_tau": m["tau"], f"{ifo}_rho0_B": m["rho0"],
                            f"{ifo}_amp_re": m["amp_re"], f"{ifo}_amp_im": m["amp_im"]})
                if ifo not in members:
                    row.update({f"{ifo}_start": r_start + m["tau"], f"{ifo}_duration": r_dur,
                                f"{ifo}_rec_start": s_start + m["tau"], f"{ifo}_rec_end": s_end + m["tau"],
                                f"{ifo}_f_min": float(rr["freqMin"]), f"{ifo}_f_max": float(rr["freqMax"]),
                                f"{ifo}_f_mean": float(rr["freqMean"])})
                # B's own event: the node's event of B, else the loudest of B's
                # events whose support meets the reference's shifted by tau
                if ifo in members:
                    own = refs[ifo].loc[[members[ifo]]]
                else:
                    own = events_meeting(every[ifo], s_start + m["tau"], s_end + m["tau"], half[(ref, ifo)])
                row[f"{ifo}_tile_n"] = int(len(own))
                tile_energy[ifo] = 0.0
                if len(own):
                    b = own.iloc[0]
                    tile_energy[ifo] = float(b["EnWDF"])
                    dt_env = float(b["gpsEnvelope"]) - float(rr["gpsEnvelope"])
                    row.update({f"{ifo}_tile_cluster": int(b["cluster_id"]),
                                f"{ifo}_tile_EnWDF": float(b["EnWDF"]),
                                f"{ifo}_tile_above": bool(float(b["EnWDF"]) >= float(settings.event_threshold)),
                                f"{ifo}_tile_guided": bool(b.get("guided", False)),
                                f"{ifo}_tile_start": float(b["gpsStart"]),
                                f"{ifo}_tile_duration": float(b["duration"]),
                                f"{ifo}_tile_rec_start": float(b["rec_start"]),
                                f"{ifo}_tile_rec_end": float(b["rec_end"]),
                                f"{ifo}_tile_f_min": float(b["freqMin"]),
                                f"{ifo}_tile_f_max": float(b["freqMax"]),
                                f"{ifo}_tile_dt": dt_env, f"{ifo}_tile_dt_diff": m["tau"] - dt_env,
                                f"{ifo}_tile_on_track": on_track_fraction(
                                    tiles_of(ifo, int(b["cluster_id"])), tiles_of(ref, int(rr["cluster_id"])),
                                    m["tau"], settings.on_track_octaves)})
        row[f"{ifo}_source"] = source
        row[f"{ifo}_E"] = energy.get(ifo, np.nan)
    row["network_energy"] = float(np.sqrt(sum(v ** 2 for v in energy.values())))
    row["network_energy_tiles"] = float(np.sqrt(sum(v ** 2 for v in tile_energy.values())))
    for x, y in itertools.combinations(ifos, 2):
        p = x + y
        row[f"dt_{p}"] = tau.get(y, np.nan) - tau.get(x, np.nan)
        row[f"dt_env_{p}"] = envelope[y] - envelope[x] if x in envelope and y in envelope else np.nan
        row[f"dt_diff_{p}"] = row[f"dt_{p}"] - row[f"dt_env_{p}"]
    return row


def _templates(det, refs, fs):
    """``{row: (gps of the first sample, reconstruction / sigma)}`` of the
    references of one detector."""
    from wdf.analysis.detector_graph import event_waveform

    out = {}
    for j, r in refs.iterrows():
        cid = int(r["cluster_id"])
        if det.run_holding(float(r["gpsStart"]), fs) is None:
            continue
        rec = event_waveform(det.graph, det.labels, cid).get(cid)
        if rec is None:
            continue
        t = float(rec[0]) + np.arange(len(rec[1])) / fs
        out[j] = (float(rec[0]), np.asarray(rec[1], dtype=float) / det.history.sigma_at(t))
    return out


def _with_support(events, graph, labels, fraction, fs):
    """`events` with ``rec_start`` and ``rec_end`` (`reconstruction_support`)."""
    from wdf.analysis.detector_graph import event_waveform

    events = events.copy()
    if not len(events):
        events["rec_start"], events["rec_end"] = np.zeros(0), np.zeros(0)
        return events
    rec = event_waveform(graph, labels)
    sup = {c: reconstruction_support(float(r[0]), r[1], fraction, fs) for c, r in rec.items()}
    cid = events["cluster_id"].astype(int).to_numpy()
    events["rec_start"] = [sup.get(c, (np.nan, np.nan))[0] for c in cid]
    events["rec_end"] = [sup.get(c, (np.nan, np.nan))[1] for c in cid]
    return events


#: Event columns that are instants, moved with a detector's time shift.
SHIFTED_COLUMNS = ("gpsStart", "gpsEnvelope", "rec_start", "rec_end")


@dataclass
class PreparedNetwork:
    """The part of the stage that does not depend on the time shifts: per
    detector the events with their supports, the references, their
    templates and tiles (`prepare_network_events`). `run` builds the matched
    filters and the nodes for one set of shifts, `slides` for several.

    :param detectors: ``{ifo: DetectorInput}``.
    :param fs: rate, Hz.
    :param settings: NetworkEventsSettings.
    :param travel: ``{(a, b): light travel time}``; :param half: plus the
        jitter, the lag range of the matched filter.
    :param every: ``{ifo: every event with rec_start, rec_end}``.
    :param refs: ``{ifo: the references with rho_own}``.
    :param templates: ``{ifo: {row: (t0, h)}}`` of the references.
    """

    detectors: dict
    fs: float
    settings: NetworkEventsSettings
    travel: dict
    half: dict
    every: dict
    refs: dict
    templates: dict
    _tiles: dict = field(default_factory=dict)

    @property
    def ifos(self):
        return list(self.detectors)

    def tiles(self, ifo, cid):
        """An event's tiles (`event_tiles`), unshifted."""
        from wdf.analysis.detector_graph import event_tiles, tile_context

        det, cache = self.detectors[ifo], self._tiles
        if ("context", ifo) not in cache:
            labels = np.asarray(det.labels)
            order = np.argsort(labels, kind="stable")
            sizes = np.bincount(labels)
            cache[("context", ifo)] = tile_context(det.graph.nodes)
            cache[("members", ifo)] = (order, sizes, np.concatenate(([0], np.cumsum(sizes)[:-1])))
        if (ifo, cid) not in cache:
            order, sizes, starts = cache[("members", ifo)]
            cache[(ifo, cid)] = event_tiles(det.graph.nodes, order[starts[cid]:starts[cid] + sizes[cid]],
                                            cache[("context", ifo)])
        return cache[(ifo, cid)]

    def _moved(self, ifo, times, shift_samples):
        """Each of `times` moved by the shift, circularly within the stream
        that holds it (unchanged times outside every stream move by the
        shift)."""
        times = np.asarray(times, dtype=float)
        out = times + shift_samples / self.fs
        if not shift_samples:
            return times.copy()
        for t0, y in self.detectors[ifo].streams:
            span = y.size / self.fs
            inside = (times >= t0) & (times < t0 + span)
            out[inside] = t0 + np.mod(times[inside] - t0 + shift_samples / self.fs, span)
        return out

    def run(self, shifts=None):
        """The matched filters and the nodes with each detector's whitened
        stream, events and reconstructions moved by its shift.

        :param shifts: ``{ifo: s}``; a detector not named is not moved. The
            shift is rounded to a sample and applied circularly within each
            of the detector's streams (a sample moved past a stream's end
            re-enters at its start); every instant of an event (its extent,
            envelope instant, support, template and tiles) moves by the shift
            of the stream that holds its start.
        :return: NetworkEventsResult -- with ``shifts`` given, the node table
            carries ``shift_<ifo>`` (s) for every detector.
        """
        fs, settings, ifos = self.fs, self.settings, self.ifos
        samples = {i: int(round(float((shifts or {}).get(i, 0.0)) * fs)) for i in ifos}
        dets, refs, every, templates, delta = {}, {}, {}, {}, {}
        for ifo, det in self.detectors.items():
            k = samples[ifo]
            if not k:
                dets[ifo], refs[ifo], every[ifo], templates[ifo] = (det, self.refs[ifo], self.every[ifo],
                                                                    self.templates[ifo])
                delta[ifo] = {}
                continue
            dets[ifo] = DetectorInput(events=det.events, graph=det.graph, labels=det.labels, history=det.history,
                                      streams=[(t0, np.roll(y, k)) for t0, y in det.streams],
                                      searched=det.searched)
            starts = self.every[ifo]["gpsStart"].to_numpy(float)
            moved = self._moved(ifo, starts, k) - starts
            delta[ifo] = dict(zip(self.every[ifo]["cluster_id"].astype(int).to_numpy(), moved))
            for name, table in (("every", self.every[ifo]), ("refs", self.refs[ifo])):
                d = table["cluster_id"].astype(int).map(delta[ifo]).to_numpy(float) if len(table) else np.zeros(0)
                table = table.copy()
                for c in SHIFTED_COLUMNS:
                    if c in table:
                        table[c] = table[c].to_numpy(float) + d
                (every if name == "every" else refs)[ifo] = table
            templates[ifo] = {j: (th + delta[ifo][int(self.refs[ifo].loc[j, "cluster_id"])], h)
                              for j, (th, h) in self.templates[ifo].items()}

        def tiles_of(ifo, cid):
            tiles = self.tiles(ifo, cid)
            d = delta[ifo].get(int(cid), 0.0)
            return (tiles[0] + d, tiles[1] + d) + tuple(tiles[2:]) if d else tiles

        matches = {}
        for a in ifos:
            for j, (th, h) in templates[a].items():
                for b in ifos:
                    if b == a:
                        continue
                    kb = dets[b].run_holding(th + 0.5 * h.size / fs, fs)
                    matches[(a, j, b)] = None if kb is None else match_template(
                        (th, h), dets[b].streams[kb], self.half[(a, b)], int(settings.lag_step), fs)

        joined = coincident_nodes(refs, ifos, matches)
        used = {(i, r) for n in joined for i, r in n.items()}
        rows = [_node_row(n, refs, every, matches, ifos, COINCIDENT, self.half, tiles_of, settings)
                for n in joined]
        for i in ifos:
            for j in range(len(refs[i])):
                if (i, j) not in used and np.isfinite(refs[i].loc[j, "rho_own"]):
                    rows.append(_node_row({i: j}, refs, every, matches, ifos, SINGLE, self.half, tiles_of,
                                          settings))
        nodes = pd.DataFrame(rows)
        if len(nodes):
            nodes = nodes.sort_values("network_energy", ascending=False, ignore_index=True)
            e = nodes["network_energy"].to_numpy(float)
            nodes["rank"] = [int((e > x).sum()) + 1 for x in e]
            nodes["n_nodes"] = len(nodes)
        nodes = nodes.reindex(columns=[c for c in node_columns(ifos) if c in nodes or not len(nodes)])
        if shifts is not None:
            for i in ifos:
                nodes[f"shift_{i}"] = samples[i] / fs

        ratio = {}
        for i in ifos:
            q = (refs[i]["rho_own"] / refs[i]["EnWDF"]).to_numpy(float)
            q = q[np.isfinite(q)]
            ratio[i] = dict(references=int(len(refs[i])), templates=int(q.size),
                            median=float(np.median(q)) if q.size else None)
        pairs = list(itertools.combinations(ifos, 2))
        info = dict(settings={k: getattr(settings, k) for k in NetworkEventsSettings.KEYS},
                    detectors=ifos, shifts_s={i: samples[i] / fs for i in ifos},
                    light_travel_s={a + b: self.travel[(a, b)] for a, b in pairs},
                    tau_half_s={a + b: self.half[(a, b)] for a, b in pairs},
                    rho_own_over_EnWDF=ratio,
                    nodes={k: int((nodes["kind"] == k).sum()) if len(nodes) else 0 for k in (COINCIDENT, SINGLE)})
        return NetworkEventsResult(nodes=nodes, info=info, references=refs, matches=matches)

    def slides(self, shifts):
        """The node tables of several sets of shifts, one after the other.

        :param shifts: iterable of ``{ifo: s}``.
        :return: pandas.DataFrame -- the node tables with ``slide`` (the index
            of the set) and ``shift_<ifo>``.
        """
        tables = [self.run(s).nodes.assign(slide=k) for k, s in enumerate(shifts)]
        return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def prepare_network_events(detectors, fs, settings=None, light_travel=None):
    """What the stage reads once, whatever the time shifts: the events with
    their supports, the references, their templates.

    :param detectors: ``{ifo: DetectorInput}``, in the order of the table's
        columns; leave out a detector whose whitening failed.
    :param fs: sampling rate of the streams and reconstructions, Hz.
    :type settings: NetworkEventsSettings
    :param settings: the stage's settings; the defaults when None.
    :param light_travel: ``{(a, b): s}`` overriding
        `detectors.light_travel_time`.
    :return: PreparedNetwork
    """
    from wdf.analysis.detectors import light_travel_time

    settings = settings or NetworkEventsSettings()
    ifos = list(detectors)
    fs = float(fs)
    travel = {}
    for a, b in itertools.permutations(ifos, 2):
        travel[(a, b)] = float((light_travel or {}).get((a, b), (light_travel or {}).get((b, a), np.nan)))
        if not np.isfinite(travel[(a, b)]):
            travel[(a, b)] = float(light_travel_time(a, b))
    half = {k: v + float(settings.jitter_s) for k, v in travel.items()}

    refs, every, templates = {}, {}, {}
    for ifo, det in detectors.items():
        every[ifo] = _with_support(det.events.reset_index(drop=True), det.graph, det.labels,
                                   float(settings.support_fraction), fs)
        keep = every[ifo]["EnWDF"].to_numpy(float) >= float(settings.event_threshold)
        if "guided" in every[ifo]:
            keep &= ~every[ifo]["guided"].astype(bool).to_numpy()
        refs[ifo] = every[ifo][keep].sort_values("EnWDF", ascending=False).reset_index(drop=True)
        templates[ifo] = _templates(det, refs[ifo], fs)
        refs[ifo]["rho_own"] = [float(np.linalg.norm(templates[ifo][j][1])) if j in templates[ifo]
                                else np.nan for j in range(len(refs[ifo]))]
    return PreparedNetwork(detectors=detectors, fs=fs, settings=settings, travel=travel, half=half,
                           every=every, refs=refs, templates=templates)


def network_events(detectors, fs, settings=None, light_travel=None, shifts=None):
    """The network events of a stretch (`prepare_network_events`, then
    `PreparedNetwork.run`).

    :param shifts: ``{ifo: s}``, time shifts of the detectors
        (`PreparedNetwork.run`); None: no shift and no ``shift_<ifo>``
        column.
    :return: NetworkEventsResult
    """
    return prepare_network_events(detectors, fs, settings, light_travel).run(shifts)


# ------------------------------------------------------------------ from stage 2
def scored_events(graph, labels, events, fs, window, overlap):
    """A detector's events with `EnWDF` measured on the stitched
    reconstruction (`score_events_by_reconstruction`); the per-window
    quadrature value is kept as ``EnWDF_quadrature``.

    :param window: `window`, samples; :param overlap: `overlap`, samples.
    """
    from wdf.analysis.cluster_coefficients import (collect_cluster_coefficients,
                                                   score_events_by_reconstruction)

    nodes = graph.nodes.copy()
    nodes["cluster_id"] = np.asarray(labels)
    out = score_events_by_reconstruction(
        events, collect_cluster_coefficients(nodes, events, float(fs), int(window), int(overlap)))
    if len(out):
        out.insert(out.columns.get_loc("EnWDF") + 1, "EnWDF_quadrature", events["EnWDF"].to_numpy())
    return out


def network_events_from_stage(par, graphs, labels, events, ar_models, streams, fs=None,
                              envelope_width=None, settings=None, prepare_only=False):
    """The stage on the outputs of stage 2 (`network_driver.wavegram_stage`).

    :param par: run configuration (`window`, `overlap`, `NetworkEnvelopeWidth`
        and the keys of `NetworkEventsSettings`).
    :param graphs: ``{ifo: DetectorGraph}``; :param labels: ``{ifo: labels}``.
    :param events: ``{ifo: detector events}`` as `detector_events` returns
        them (scored here on their reconstruction).
    :param ar_models: ``{ifo: ModelHistory or ARModels file(s)}``.
    :param streams: ``{ifo: [(t0, samples) or (t0, samples, (start, end))]}``,
        the whitened stream over sigma of each run (`whitened_stream`).
    :param fs: rate, Hz; from the graphs' nodes when None.
    :param envelope_width: `NetworkEnvelopeWidth` for `gpsEnvelope` when the
        events lack it.
    :param prepare_only: return the `PreparedNetwork`, for time slides.
    :return: NetworkEventsResult, or PreparedNetwork with `prepare_only`
    """
    from wdf.analysis.hrss import ModelHistory
    from wdf.analysis.network_driver import _envelope_width, add_envelope_instants

    settings = settings or NetworkEventsSettings.from_parameters(par)
    if fs is None:
        fs = float(pd.to_numeric(next(iter(graphs.values())).nodes["fs"]).iloc[0])
    if any(i not in ar_models or i not in streams for i in events):
        raise ValueError("the network events need every detector's noise models and whitened streams")
    if envelope_width is None:
        envelope_width = _envelope_width(parameter(par, "NetworkEnvelopeWidth", "network.driver"))
    scored = {i: scored_events(graphs[i], labels[i], events[i], fs, par.window, par.overlap)
              for i in events}
    missing = {i: ev for i, ev in scored.items() if "gpsEnvelope" not in ev}
    if missing:
        scored.update(add_envelope_instants(missing, graphs, labels, envelope_width))
    inputs = {}
    for i, ev in scored.items():
        history = ar_models[i] if isinstance(ar_models[i], ModelHistory) else ModelHistory.from_files(ar_models[i])
        runs = [tuple(s) for s in streams[i]]
        inputs[i] = DetectorInput(events=ev, graph=graphs[i], labels=labels[i], history=history,
                                  streams=[r[:2] for r in runs],
                                  searched=[r[2] if len(r) > 2 else None for r in runs])
    prepared = prepare_network_events(inputs, fs, settings)
    return prepared if prepare_only else prepared.run()


# ------------------------------------------------------------------ the streams
def whitened_stream(read, par, history, segment, start, end, atoms=None, settings=None):
    """The whitened stream over sigma of one run over ``[start, end]``,
    rebuilt from its raw strain as the zero-phase worker whitened it: the
    conditioning (`waveform_completion.conditioned_stretch`), the glitch
    atoms the run subtracted (`GlitchAtomsFrom`), then the models it applied
    (`waveform_completion.whitened_stretch`, divided by sigma).

    The read reaches half the whitening filter, `NetworkEventsStreamBlocks`
    conditioning blocks and `NetworkEventsStreamPad` s beyond ``[start,
    end]``. On a side where that reaches non-finite strain outside the run's
    `segment` (a run that starts after, or ends before, the valid data
    around it), the read is kept inside the segment on that side, as the
    worker conditioned it; the stream then starts later (ends earlier) and
    the caller keeps to what it holds.

    :param read: ``read(a, b) -> (samples, rate, gps of the first)``, the
        raw strain of the detector over ``[a, b)`` (NaN where missing).
    :param par: the run's configuration (`ZeroPhaseDuration`, `len` and the
        band-pass keys).
    :type history: hrss.ModelHistory
    :param history: the models the run applied.
    :param segment: ``(start, end)`` GPS of the run's valid data.
    :param atoms: a `glitch_subtraction.GlitchModel`, a GlitchAtoms CSV path,
        or None.
    :return: tuple -- (GPS of the first sample, samples, rate).
    :raises ValueError: if the stream holds non-finite samples.
    """
    from wdf.analysis.waveform_completion import conditioned_stretch, whitened_stretch
    from wdf.processes.BandPassDownSampling import BandPassDownSampling
    from wdf.processes.glitch_subtraction import Atom, GlitchModel

    settings = settings or NetworkEventsSettings.from_parameters(par)
    edge = (float(par.ZeroPhaseDuration) / 2 + int(settings.stream_blocks) * float(par.len)
            + float(settings.stream_pad_s))
    r0, r1 = int(np.floor(start - edge)), int(np.ceil(end + edge))
    seg0, seg1 = int(np.ceil(segment[0])), int(np.floor(segment[1]))
    raw, rate_in, t0 = read(r0, r1)
    raw = np.asarray(raw, dtype=float)
    n0 = int(round((seg0 - float(t0)) * rate_in))
    n1 = int(round((seg1 - float(t0)) * rate_in))
    cut0 = r0 < seg0 and not np.all(np.isfinite(raw[:max(n0, 0)]))
    cut1 = r1 > seg1 and not np.all(np.isfinite(raw[max(n1, 0):]))
    if cut0 or cut1:
        r0, r1 = (max(r0, seg0) if cut0 else r0), (min(r1, seg1) if cut1 else r1)
        raw, rate_in, t0 = read(r0, r1)
        raw = np.asarray(raw, dtype=float)
    y, y0, rate = conditioned_stretch(raw, float(rate_in), float(t0), BandPassDownSampling(par),
                                      float(par.len))
    if isinstance(atoms, str) and atoms:
        atoms = GlitchModel([Atom(**r) for r in pd.read_csv(atoms).to_dict("records")])
    if atoms is not None and len(atoms):
        y = atoms.subtract(y, y0, rate)
    first, w, rate = whitened_stretch(y, rate, y0, history, normalise=True)
    w = np.asarray(w, dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{int((~np.isfinite(w)).sum())} non-finite samples in the whitened stream "
                         f"[{r0}, {r1}]")
    return float(first), w, float(rate)
