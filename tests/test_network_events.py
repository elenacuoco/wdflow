"""wdf.analysis.network_events: the matched filter, the supports, the node
joining and the network energy."""
import numpy as np
import pandas as pd
import pytest

from wdf.analysis.network_events import (COINCIDENT, SINGLE, DetectorInput, NetworkEventsSettings,
                                         analytic_matched_filter, coincident_nodes,
                                         match_template, network_events, node_columns,
                                         on_track_fraction, reconstruction_support)
from wdf.config.Parameters import Parameters

FS = 2048.0


def sine_gaussian(n, centre, f, q, fs=FS, phase=0.0):
    t = (np.arange(n) - centre) / fs
    tau = q / (2 * np.pi * f)
    return np.exp(-t ** 2 / tau ** 2) * np.cos(2 * np.pi * f * t + phase)


# ------------------------------------------------------------------ the matched filter
@pytest.mark.parametrize("delay,amplitude,phase", [(0, 3.0, 0.0), (17, 2.5, 0.0), (-23, 4.0, np.pi / 2),
                                                    (5, 3.0, -0.7)])
def test_the_matched_filter_recovers_a_planted_delay_and_amplitude(delay, amplitude, phase):
    rng = np.random.default_rng(1)
    h = sine_gaussian(512, 256, 150.0, 8.0)
    h *= 20.0 / np.linalg.norm(h)                       # rho of the template in unit white noise: 20
    n = 8192
    t0 = 1000.0
    th = t0 + 4000 / FS
    y = rng.standard_normal(n)
    y[4000 + delay:4000 + delay + h.size] += amplitude * sine_gaussian(512, 256, 150.0, 8.0, phase=phase) \
        * 20.0 / np.linalg.norm(sine_gaussian(512, 256, 150.0, 8.0))
    m = match_template((th, h), (t0, y), 0.02, 1, FS)
    assert m["tau"] == pytest.approx(delay / FS, abs=0.5 / FS)
    # s = A (cos(phase) h - sin(phase) H h): amp_re = A cos(phase), amp_im = -A sin(phase)
    assert np.hypot(m["amp_re"], m["amp_im"]) == pytest.approx(amplitude, rel=0.1)
    assert m["amp_re"] == pytest.approx(amplitude * np.cos(phase), abs=0.15)
    assert m["amp_im"] == pytest.approx(-amplitude * np.sin(phase), abs=0.15)
    assert m["rho"] == pytest.approx(amplitude * 20.0, rel=0.1)
    assert m["rho"] >= m["rho0"]


def test_the_matched_filter_is_rayleigh_of_scale_one_in_white_noise():
    rng = np.random.default_rng(2)
    h = sine_gaussian(256, 128, 200.0, 6.0)
    rho, _, _ = analytic_matched_filter(rng.standard_normal(200000), h)
    # Rayleigh of scale 1: mean sqrt(pi / 2), second moment 2
    assert np.mean(rho ** 2) == pytest.approx(2.0, rel=0.03)
    assert np.mean(rho) == pytest.approx(np.sqrt(np.pi / 2), rel=0.03)


def test_the_lag_step_and_the_lag_range_are_kept():
    rng = np.random.default_rng(3)
    h = sine_gaussian(256, 128, 100.0, 8.0)
    y = rng.standard_normal(4096) * 0.01
    y[2000 + 9:2000 + 9 + h.size] += h
    m1 = match_template((2000 / FS, h), (0.0, y), 0.01, 1, FS)
    m3 = match_template((2000 / FS, h), (0.0, y), 0.01, 3, FS)
    assert m1["tau"] * FS == pytest.approx(9)
    assert round(m3["tau"] * FS) % 3 == 0              # only lags on the step
    narrow = match_template((2000 / FS, h), (0.0, y), 3 / FS, 1, FS)
    assert abs(narrow["tau"]) <= 3 / FS + 1e-12
    # the stream must hold the template and every lag
    assert match_template((0.0, h), (0.0, y), 0.01, 1, FS) is None
    assert match_template((2000 / FS, np.zeros(16)), (0.0, y), 0.01, 1, FS) is None


def test_the_support_is_where_the_envelope_reaches_the_fraction():
    h = sine_gaussian(2048, 1024, 100.0, 20.0)
    a, b = reconstruction_support(10.0, h, 0.1, FS)
    tau = 20.0 / (2 * np.pi * 100.0)
    half = tau * np.sqrt(np.log(10.0))                  # exp(-t^2 / tau^2) = 0.1
    assert a == pytest.approx(10.0 + 1024 / FS - half, abs=2 / FS)
    assert b == pytest.approx(10.0 + 1024 / FS + half, abs=2 / FS)
    padded = np.concatenate([np.zeros(500), h, np.zeros(700)])
    a2, b2 = reconstruction_support(10.0 - 500 / FS, padded, 0.1, FS)
    assert (a2, b2) == pytest.approx((a, b), abs=1e-9)  # empty steps do not widen it
    assert np.isnan(reconstruction_support(0.0, np.zeros(10), 0.1, FS)[0])


def test_on_track_reads_time_and_band():
    a = (np.array([0.0]), np.array([0.1]), np.array([100.0]), np.array([200.0]), np.array([1.0]))
    on = (np.array([0.01]), np.array([0.09]), np.array([120.0]), np.array([180.0]), np.array([4.0]))
    off_band = (np.array([0.01]), np.array([0.09]), np.array([500.0]), np.array([600.0]), np.array([1.0]))
    b = tuple(np.concatenate([x, y]) for x, y in zip(on, off_band))
    assert on_track_fraction(b, a, 0.0, 0.0) == pytest.approx(0.8)
    assert on_track_fraction(b, a, 0.0, 2.0) == pytest.approx(1.0)       # 200 Hz x 4 reaches 500 Hz
    assert on_track_fraction(b, a, 0.5, 0.0) == pytest.approx(0.0)       # shifted away in time


# ------------------------------------------------------------------ the nodes
def refs_table(rho, start, end):
    return pd.DataFrame(dict(rho_own=np.asarray(rho, float), rec_start=np.asarray(start, float),
                             rec_end=np.asarray(end, float)))


def test_nodes_join_the_pairs_that_meet_loudest_first():
    refs = {"H1": refs_table([30.0, 8.0], [10.0, 50.0], [10.2, 50.1]),
            "L1": refs_table([20.0, 9.0], [10.005, 50.0], [10.2, 50.1]),
            "V1": refs_table([6.0], [10.0], [10.1])}
    matches = {}
    for a in refs:
        for k in range(len(refs[a])):
            for b in refs:
                if b != a:
                    matches[(a, k, b)] = dict(tau=0.004)
    nodes = coincident_nodes(refs, list(refs), matches)
    assert {"H1": 0, "L1": 0, "V1": 0} in nodes          # the three loud ones: one node
    assert {"H1": 1, "L1": 1} in nodes
    assert len(nodes) == 2


def test_the_louder_template_decides_and_a_detector_is_used_once():
    # H1 louder; its support shifted by its tau misses L1's: no node, whatever L1's own tau says
    refs = {"H1": refs_table([30.0], [10.0], [10.1]), "L1": refs_table([10.0, 9.0], [10.2, 10.0], [10.3, 10.1])}
    matches = {("H1", 0, "L1"): dict(tau=0.15), ("L1", 0, "H1"): dict(tau=-0.2), ("L1", 1, "H1"): dict(tau=0.0)}
    nodes = coincident_nodes(refs, ["H1", "L1"], matches)
    # H1 at +0.15 meets L1 event 0 ([10.2, 10.3]) and not event 1 ([10.0, 10.1]): one node, H1 used once
    assert nodes == [{"H1": 0, "L1": 0}]
    matches[("H1", 0, "L1")] = None                     # L1 not searched at H1's time: no node
    assert coincident_nodes(refs, ["H1", "L1"], matches) == []


# ------------------------------------------------------------------ the whole stage
class History:
    """A model history of constant sigma."""

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def sigma_at(self, times):
        return np.full(np.asarray(times).size, self.sigma)


@pytest.fixture(scope="module")
def three_detectors():
    """H1 and L1 see one sine-Gaussian, L1 5 ms later and weaker; V1 sees
    nothing. Triggers from the clean signal, streams with unit white
    noise."""
    pytest.importorskip("py4tsa")
    from _synth import triggers_from_signal

    from wdf.analysis.detector_graph import build_detector_graph, detector_events
    from wdf.analysis.network_driver import add_envelope_instants
    from wdf.analysis.network_events import scored_events

    rng = np.random.default_rng(7)
    n, t0 = int(64 * FS), 1000.0
    at = int(32 * FS)
    shape = sine_gaussian(4096, 2048, 180.0, 9.0)
    shape *= 25.0 / np.linalg.norm(shape)
    delay = int(round(0.005 * FS))
    out = {}
    for ifo, d, amp in (("H1", 0, 1.0), ("L1", delay, 0.6), ("V1", 0, 0.0)):
        y = rng.standard_normal(n)
        y[at - 2048 + d:at + 2048 + d] += amp * shape
        if amp:
            sig = np.zeros(8192)
            sig[2048 + d:6144 + d] = amp * shape
            trig = triggers_from_signal(sig, FS, 512, 128, gps0=t0 + (at - 4096) / FS, ifo=ifo, wave="DaubC4")
            trig["stride"] = (512 - 128) / FS
            graph = build_detector_graph(trig)
            labels = graph.components()
            ev = detector_events(graph, labels=labels)
            ev = scored_events(graph, labels, ev, FS, 512, 128)
            ev = add_envelope_instants({ifo: ev}, {ifo: graph}, {ifo: labels}, "window")[ifo]
        else:
            graph, labels, ev = None, np.zeros(0, int), pd.DataFrame(columns=["cluster_id", "EnWDF"])
        out[ifo] = DetectorInput(events=ev, graph=graph, labels=labels, history=History(),
                                 streams=[(t0, y)], searched=[(t0 + 4, t0 + 60)])
    return out, delay


def test_a_signal_in_two_detectors_is_one_coincident_node(three_detectors):
    inputs, delay = three_detectors
    res = network_events(inputs, FS, NetworkEventsSettings(event_threshold=5.0))
    top = res.nodes.iloc[0]
    assert top["kind"] == COINCIDENT and top["ref"] == "H1" and top["n_event_detectors"] == 2
    assert top["L1_source"] == "event" and top["V1_source"] == "mf"
    assert top["L1_tau"] == pytest.approx(delay / FS, abs=1.5 / FS)
    assert top["dt_H1L1"] == pytest.approx(top["L1_tau"])
    # one network energy: the reference's own rho and the matched filter in every other detector
    assert top["network_energy"] == pytest.approx(np.sqrt(top["H1_rho_own"] ** 2 + top["L1_rho_B"] ** 2
                                                          + top["V1_rho_B"] ** 2))
    assert top["H1_E"] == top["H1_rho_own"] and top["L1_E"] == top["L1_rho_B"]
    assert top["L1_rho_B"] == pytest.approx(0.6 * top["H1_rho_own"], rel=0.15)
    assert np.hypot(top["L1_amp_re"], top["L1_amp_im"]) == pytest.approx(0.6, rel=0.15)
    assert top["V1_rho_B"] < 5.0
    assert top["L1_tile_n"] == 1 and bool(top["L1_tile_above"])
    assert top["L1_tile_on_track"] > 0.5
    assert top["rank"] == 1 and list(res.nodes.columns) == [c for c in node_columns(["H1", "L1", "V1"])
                                                              if c in res.nodes.columns]


def test_below_the_threshold_in_one_detector_it_is_a_single_with_matched_filter(three_detectors):
    inputs, delay = three_detectors
    threshold = float(inputs["L1"].events["EnWDF"].max()) + 1.0
    res = network_events(inputs, FS, NetworkEventsSettings(event_threshold=threshold))
    top = res.nodes.iloc[0]
    assert top["kind"] == SINGLE and top["ref"] == "H1"
    assert top["L1_source"] == "mf" and not bool(top["L1_tile_above"])     # L1's own event, sub-threshold
    assert top["L1_tau"] == pytest.approx(delay / FS, abs=1.5 / FS)
    assert top["network_energy"] == pytest.approx(np.sqrt(top["H1_rho_own"] ** 2 + top["L1_rho_B"] ** 2
                                                          + top["V1_rho_B"] ** 2))
    # a detector not searched at the reference's time contributes nothing
    alone = dict(inputs)
    alone["V1"] = DetectorInput(events=inputs["V1"].events, graph=None, labels=np.zeros(0, int),
                                history=History(), streams=inputs["V1"].streams, searched=[(0.0, 1.0)])
    top = network_events(alone, FS, NetworkEventsSettings(event_threshold=threshold)).nodes.iloc[0]
    assert top["V1_source"] == "not searched" and np.isnan(top["V1_E"])
    assert top["network_energy"] == pytest.approx(np.hypot(top["H1_rho_own"], top["L1_rho_B"]))


def test_guided_events_are_never_references(three_detectors):
    inputs, _ = three_detectors
    guided = dict(inputs)
    ev = inputs["H1"].events.copy()
    ev["guided"] = True
    guided["H1"] = DetectorInput(events=ev, graph=inputs["H1"].graph, labels=inputs["H1"].labels,
                                 history=History(), streams=inputs["H1"].streams,
                                 searched=inputs["H1"].searched)
    nodes = network_events(guided, FS, NetworkEventsSettings(event_threshold=5.0)).nodes
    assert (nodes["ref"] != "H1").all() and nodes.iloc[0]["ref"] == "L1"


# ------------------------------------------------------------------ settings
def test_the_settings_are_read_from_the_run_and_written_back():
    par = Parameters()
    s = NetworkEventsSettings.from_parameters(par)
    assert s == NetworkEventsSettings()
    assert s.active is False and s.event_threshold == 0.0 and s.jitter_s == pytest.approx(0.01)
    assert par.NetworkEvents is False and par.NetworkEventsLagStep == 1
    par = Parameters(NetworkEvents=True, EventThreshold=5.0, NetworkEventsJitter=0.0035)
    s = NetworkEventsSettings.from_parameters(par)
    assert s.active and s.event_threshold == 5.0 and s.jitter_s == 0.0035
    with pytest.raises(ValueError):
        NetworkEventsSettings(support_fraction=1.0)
    with pytest.raises(ValueError):
        NetworkEventsSettings(lag_step=0)


# ------------------------------------------------------------------ run_network
def snapshot_triggers():
    import os

    pytest.importorskip("py4tsa")
    fixtures = os.path.join(os.path.dirname(__file__), "fixtures")
    return {ifo: pd.read_parquet(os.path.join(fixtures, f"paper_path_triggers_{ifo}.parquet"))
            for ifo in ("H1", "L1")}


def unit_history(start):
    from wdf.analysis.hrss import ModelHistory

    return ModelHistory([dict(switch_gps=start, fit_start=start, fit_end=start, sigma=1.0,
                              ar=np.array([1.0, 0.0]))],
                        dict(WhiteningMode="zerophase", resampling=FS, ZeroPhaseDuration=4.0))


def test_by_default_run_network_is_unchanged():
    from wdf.analysis.network_driver import run_network

    triggers = snapshot_triggers()
    base = run_network(Parameters(), triggers=triggers)
    assert base.network_events is None
    par = Parameters(NetworkEvents=False, EventThreshold=5.0)
    off = run_network(par, triggers=triggers, ar_models={}, streams={})
    pd.testing.assert_frame_equal(off.candidates, base.candidates)
    assert off.network_events is None and not hasattr(par, "NetworkEventsJitter")
    with pytest.raises(ValueError, match="NetworkEvents"):
        run_network(Parameters(NetworkEvents=True), triggers=triggers)


def test_run_network_builds_the_network_events_when_asked():
    from wdf.analysis.network_driver import run_network

    triggers = snapshot_triggers()
    t0 = min(float(t["gps"].min()) for t in triggers.values()) - 20.0
    rng = np.random.default_rng(11)
    streams = {i: [(t0, rng.standard_normal(int(120 * FS)))] for i in triggers}
    base = run_network(Parameters(), triggers=triggers)
    par = Parameters(NetworkEvents=True, EventThreshold=5.0, window=1024, overlap=256)
    res = run_network(par, triggers=triggers, ar_models={i: unit_history(t0) for i in triggers},
                      streams=streams)
    pd.testing.assert_frame_equal(res.candidates, base.candidates)        # the classical path is untouched
    ne = res.network_events
    assert ne is not None and len(ne.nodes)
    assert par.NetworkEventsJitter == pytest.approx(0.01) and par.EventThreshold == 5.0
    for ifo, refs in ne.references.items():
        assert (refs["EnWDF"] >= 5.0).all()
        assert "EnWDF_quadrature" in refs and "gpsEnvelope" in refs
    top = ne.nodes.iloc[0]
    other = "L1" if top["ref"] == "H1" else "H1"
    assert top["network_energy"] == pytest.approx(np.hypot(top[f"{top['ref']}_rho_own"], top[f"{other}_rho_B"]))
    assert abs(top[f"{other}_tau"]) <= ne.info["tau_half_s"]["H1L1"] + 1e-12


# ------------------------------------------------------------------ the stream
def test_the_stream_is_rebuilt_inside_the_segment_when_the_data_start_late():
    """A run whose valid data start after the read's reach: the read is kept
    inside the segment on that side, and far from that edge the stream is the
    one rebuilt from data valid all along."""
    pytest.importorskip("py4tsa")
    from test_online_worker import parameters

    from wdf.analysis.hrss import ModelHistory
    from wdf.analysis.network_events import whitened_stream
    from wdf.processes.glitch_subtraction import Atom, GlitchModel
    from wdf.processes.wdfUnitOnline import wdfUnitOnline

    fs_in, t0, t1 = 4096.0, 1000.0, 1200.0
    x = np.random.default_rng(9).standard_normal(int((t1 - t0) * fs_in))
    late = x.copy()
    late[:int(40 * fs_in)] = np.nan                     # no valid data before 1040

    def reader(samples):
        def read(a, b):
            out = np.full(int(round((b - a) * fs_in)), np.nan)
            i0, i1 = int(round((max(a, t0) - t0) * fs_in)), int(round((min(b, t1) - t0) * fs_in))
            out[int(round((max(a, t0) - a) * fs_in)):][:i1 - i0] = samples[i0:i1]
            return out, fs_in, float(a)
        return read

    par = wdfUnitOnline(parameters(WhiteningMode="zerophase", ZeroPhaseDuration=4.0)).par
    history = ModelHistory([dict(switch_gps=t0, fit_start=t0, fit_end=t0, sigma=1.0, ar=np.r_[1.0, np.zeros(8)])],
                           dict(WhiteningMode="zerophase", resampling=float(par.resampling), ZeroPhaseDuration=4.0))
    s = NetworkEventsSettings()
    first, y, rate = whitened_stream(reader(x), par, history, (t0, t1), 1045.0, 1150.0, settings=s)
    with pytest.raises(ValueError, match="non-finite"):
        whitened_stream(reader(late), par, history, (t0, t1), 1045.0, 1150.0, settings=s)
    first_l, y_l, _ = whitened_stream(reader(late), par, history, (1040.0, t1), 1045.0, 1150.0, settings=s)
    assert first_l > first and first_l >= 1040.0 and np.all(np.isfinite(y_l))
    k = int(round((first_l - first) * rate))
    far = int(20 * rate)
    assert y_l[far:far + int(60 * rate)] == pytest.approx(y[k + far:k + far + int(60 * rate)], rel=1e-9, abs=1e-12)
    # the run's glitch atoms are subtracted before the whitening
    atom = Atom(centre=1100.0, frequency=100.0, q=8.0, alpha=50.0, beta=0.0, snr=10.0, rate=float(rate),
                half_length=int(rate))
    _, y_a, _ = whitened_stream(reader(x), par, history, (t0, t1), 1045.0, 1150.0, atoms=GlitchModel([atom]),
                                settings=s)
    i = int(round((1100.0 - first) * rate))
    assert np.abs(y_a - y)[i - 50:i + 50].max() > 1.0 and np.abs(y_a - y)[:i - int(4 * rate)].max() < 1e-9


# ------------------------------------------------------------------ background by time slides
def test_time_slides_reuse_the_events_and_lose_the_coincidence(three_detectors):
    from wdf.analysis.network_events import prepare_network_events

    inputs, delay = three_detectors
    settings = NetworkEventsSettings(event_threshold=5.0)
    plain = network_events(inputs, FS, settings).nodes
    prepared = prepare_network_events(inputs, FS, settings)
    zero = prepared.run({"H1": 0.0, "L1": 0.0, "V1": 0.0}).nodes
    assert {"shift_H1", "shift_L1", "shift_V1"} <= set(zero.columns) and (zero["shift_L1"] == 0).all()
    pd.testing.assert_frame_equal(zero.drop(columns=["shift_H1", "shift_L1", "shift_V1"]), plain)
    assert zero.iloc[0]["kind"] == COINCIDENT

    slid = prepared.run({"L1": 10.0}).nodes                # 10 s >> the light travel time
    assert (slid["shift_L1"] == 10.0).all() and (slid["shift_H1"] == 0.0).all()
    assert (slid["kind"] == SINGLE).all() and len(slid) == 2
    h1 = slid[slid["ref"] == "H1"].iloc[0]
    l1 = slid[slid["ref"] == "L1"].iloc[0]
    assert h1["L1_rho_B"] < 6.0 and h1["H1_rho_own"] == pytest.approx(zero.iloc[0]["H1_rho_own"])
    # L1's event moved with its stream: its matched filter into its own shifted stream is unchanged
    assert l1["L1_start"] == pytest.approx(zero.iloc[0]["L1_start"] + 10.0)
    assert l1["L1_rho_own"] == pytest.approx(zero.iloc[0]["L1_rho_own"])
    assert l1["H1_rho_B"] < 6.0

    # circular: a shift of a whole stream brings everything back
    span = inputs["L1"].streams[0][1].size / FS
    back = prepared.run({"L1": span}).nodes.drop(columns=["shift_H1", "shift_L1", "shift_V1"])
    pd.testing.assert_frame_equal(back, plain)

    table = prepared.slides([{}, {"L1": 10.0}, {"L1": -10.0}])
    assert table["slide"].tolist().count(0) == len(plain)
    assert (table.loc[table["slide"] > 0, "kind"] == SINGLE).all()
