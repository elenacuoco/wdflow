"""Stage 2 of the track path: seeds, seeded tracks, waveforms, events; online
within the declared latency, and offline."""
import glob
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wdf.analysis.ridge_tracking import background_z, stream_cells
from wdf.analysis.track_reconstruction import (TRACK_EVENT_COLUMNS, TrackSettings, anchored_path,
                                               reconstruct, seed_window)

FS, T0, SECONDS = 2048.0, 1.0e9, 128
CHIRP_AT, CHIRP_S, SNR = 80.0, 3.0, 25.0
#: The smallest normalised overlap of a track's waveform with the planted chirp.
MIN_OVERLAP = 0.6


def chirp(n, f0=40.0, f1=200.0):
    t = np.arange(n) / FS - CHIRP_AT
    inside = (t >= 0) & (t < CHIRP_S)
    f = f0 + (f1 - f0) * np.clip(t, 0, CHIRP_S) / CHIRP_S
    c = np.where(inside, np.sin(2 * np.pi * np.cumsum(f) / FS)
                 * np.sin(np.pi * np.clip(t, 0, CHIRP_S) / CHIRP_S) ** 2, 0.0)
    return c * SNR / np.sqrt(np.sum(c ** 2))


def planted_f(t):
    return 40.0 + 160.0 * np.clip(t - T0 - CHIRP_AT, 0, CHIRP_S) / CHIRP_S


def settings(**kw):
    par = SimpleNamespace(window=1024, overlap=256, MinFrequency=16.0, **kw)
    return TrackSettings.from_parameters(par, FS), par


@pytest.fixture(scope="module")
def planted():
    n = int(SECONDS * FS)
    c = chirp(n)
    x = np.random.default_rng(3).standard_normal(n) + c
    return x, c


def trigger(gps, peak, f):
    """A trigger record without coefficients: its loudest tile is (gpsPeak,
    freqMean)."""
    return dict(gps=T0 + gps, n_coeff=1024, fs=FS, EnWDF=10.0, freqMin=f / 2, freqMax=f * 2,
                gpsPeak=T0 + peak, freqMean=f, sigma=1.0, snrPeak=5.0)


def run(planted, basis, triggers, **kw):
    x, _ = planted
    ts, par = settings(TrackBasis=basis, **kw)
    X, origin, offs, co, sg = stream_cells(x, T0, FS, ts.ridge)
    Z = background_z(X, ts.ridge)
    return reconstruct(triggers, Z, X, origin, FS, ts, "H1", co[0.0], sg[0.0], T0 + offs / FS,
                       stream=(T0, x)), ts, par


def test_the_settings_are_parameters_and_are_recorded():
    ts, par = settings(TrackLookback=2.0, TrackBasis="CosinePackets32")
    assert (ts.lookback, ts.horizon, ts.basis, ts.seed_states) == (2.0, 1.0, "CosinePackets32", 1)
    assert par.TrackHorizon == 1.0 and par.TrackPacketCost == "entropy"
    assert settings()[0].basis == "LocalCos128"
    assert (ts.overlap_policy, ts.event_instant) == ("drop", "peak")
    assert par.TrackOverlapPolicy == "drop" and par.TrackEventInstant == "peak"
    assert settings(TrackOverlapPolicy="supersede")[0].overlap_policy == "supersede"
    for bad in (dict(TrackBasis="Haar"), dict(TrackBasis="CosinePackets30"),
                dict(TrackPacketCost="l2"), dict(TrackHorizon=-1.0),
                dict(TrackOverlapPolicy="merge"), dict(TrackEventInstant="start")):
        with pytest.raises(ValueError):
            settings(**bad)


def test_a_seed_is_its_window_with_the_lookback_and_the_horizon():
    ts, _ = settings(TrackLookback=2.0, TrackHorizon=0.5)
    s = seed_window(trigger(81.0, 81.25, 100.0), ts, T0)
    slot = ts.ridge.slot_s
    assert s.slot_lo == int(round(79.0 / slot)) and s.slot_hi == int(round(82.0 / slot))
    assert s.peak_slot == int(81.25 / slot)
    edges = ts.ridge.edges()
    assert edges[s.peak_state] <= 100.0 < edges[s.peak_state + 1]


@pytest.mark.parametrize("basis", ["LocalCos128", "LocalCos64", "CosinePackets32"])
def test_the_seeded_track_follows_the_chirp_and_its_waveform_matches(planted, basis):
    """The track contains the seed's cell, follows the planted f(t), and its
    waveform overlaps the planted chirp by more than MIN_OVERLAP."""
    x, c = planted
    seed = trigger(81.5, 82.0, float(planted_f(T0 + 82.0)))
    (events, waveforms, tracks), ts, _ = run(planted, basis, [seed])
    track = tracks[0]
    s = seed_window(seed, ts, T0 + 0.5 * ts.ridge.overlap / FS)
    assert track.contains(s.peak_slot, s.peak_state, ts.seed_states)
    inside = (track.times() > T0 + CHIRP_AT + 0.2) & (track.times() < T0 + CHIRP_AT + CHIRP_S - 0.2)
    assert inside.sum() > 0.6 * (CHIRP_S - 0.4) / ts.ridge.slot_s
    err = np.abs(np.log2(track.frequencies()[inside] / planted_f(track.times()[inside])))
    assert np.median(err) < 0.25
    t0, y = waveforms[0]
    i = int(round((t0 - T0) * FS))
    overlap = np.dot(y, c[i:i + y.size]) / np.sqrt(np.sum(y ** 2) * np.sum(c ** 2))
    assert overlap > MIN_OVERLAP
    assert list(events.columns) == TRACK_EVENT_COLUMNS
    assert events["track_basis"][0] == basis and events["n_triggers"][0] == 1


def test_overlapping_seeds_make_one_track(planted):
    seeds = [trigger(81.0, 81.4, float(planted_f(T0 + 81.4))),
             trigger(81.375, 81.8, float(planted_f(T0 + 81.8)))]
    (events, _, tracks), *_ = run(planted, "LocalCos128", seeds)
    assert len(tracks) == 1 and events["track_n_seeds"][0] == 2


def test_the_event_instant_is_the_envelope_peak_or_the_track_end(planted):
    """"peak" (the default) puts gpsPeak at the waveform's envelope peak, inside
    the chirp; "end" at the end of the track's last slot."""
    seed = trigger(81.5, 82.0, float(planted_f(T0 + 82.0)))
    (peak, _, tracks), *_ = run(planted, "LocalCos64", [seed])
    (end, _, _), *_ = run(planted, "LocalCos64", [seed], TrackEventInstant="end")
    assert T0 + CHIRP_AT < peak["gpsPeak"][0] < T0 + CHIRP_AT + CHIRP_S
    assert peak["gpsPeak"][0] != end["gpsPeak"][0]
    assert end["gpsPeak"][0] == end["track_t_end"][0] == tracks[0].t_end
    assert peak["track_supersedes"][0] == [] and end["track_supersedes"][0] == []


# ================================================================ online batches
def two_batches(planted, policy, order=("short", "long")):
    """The planted chirp seeded in two drains of a TrackObserver: one seed
    early on the chirp (its slice ends before the chirp does), one near its
    end whose track runs back over the first track's cells. Returns the
    observer and the rows of each drain."""
    from wdf.analysis.track_reconstruction import calibrated
    from wdf.observers.TrackObserver import TrackObserver

    x, _ = planted
    ts, _ = settings(TrackBasis="LocalCos64", RidgeBasis="LocalCos64", TrackOverlapPolicy=policy)
    X, origin, offs, co, sg = stream_cells(x, T0, FS, ts.ridge)
    Z = background_z(X, ts.ridge)
    cal = calibrated(ts, Z)
    seeds = dict(short=trigger(80.5, 81.0, float(planted_f(T0 + 81.0))),
                 long=trigger(82.2, 82.6, float(planted_f(T0 + 82.6))))
    obs = TrackObserver(ts, FS, "H1")
    held = (T0 + offs / FS, co[0.0], sg[0.0])
    out = []
    for name in order:
        obs.queue = [seeds[name]]
        hi = seed_window(seeds[name], ts, origin).slot_hi
        rows, kept = obs.drain(origin, hi, [(k, Z[k], X[k]) for k in range(hi)], held, cal)
        out.append((rows, kept))
    return obs, out


def test_a_chirp_seeded_in_two_batches_is_dropped_or_superseded(planted):
    """"drop": the second, longer and higher-scoring track is discarded, the
    chirp's track stops a second short of its end. "supersede": it is emitted
    with the first track's seed and its id in track_supersedes, reaching the
    end of the chirp; the first row is untouched; the counters say so."""
    from wdf.analysis.track_reconstruction import drop_superseded

    end = T0 + CHIRP_AT + CHIRP_S
    drop, (d1, d2) = two_batches(planted, "drop")
    assert len(d1[0]) == 1 and d2 == ([], {})
    first = d1[0][0]
    assert first["track_t_end"] < end - 0.3 and first["track_supersedes"] == []
    assert (drop.absorbed, drop.superseded, drop.carried, drop.count) == (1, 0, 0, 1)

    sup, (s1, s2) = two_batches(planted, "supersede")
    assert s1[0] == [first] and s1[1].keys() == d1[1].keys()
    np.testing.assert_array_equal(s1[1][0][1], d1[1][0][1])
    assert len(s2[0]) == 1
    second = s2[0][0]
    assert second["cluster_id"] == 1 and second["track_supersedes"] == [0]
    assert second["track_score"] > first["track_score"]
    assert second["track_t_end"] > first["track_t_end"] + 0.5 and second["track_t_end"] >= end - 0.5
    assert second["track_t_start"] <= first["track_t_start"]
    assert second["track_n_seeds"] == 2 and sorted(second["track_seed_gps"]) == \
        sorted([first["track_seed_gps"][0], T0 + 82.2])
    assert second["seed_end"] == T0 + 82.2 + 1024 / FS and 1 in s2[1]
    assert (sup.absorbed, sup.superseded, sup.carried, sup.count) == (0, 1, 1, 2)
    table = pd.DataFrame(s1[0] + s2[0])
    assert drop_superseded(table)["cluster_id"].tolist() == [1]


def test_a_lower_scoring_overlap_is_discarded_under_supersede(planted):
    """Seeded the other way round, the long track comes first and the short
    one, of lower score, is absorbed by either policy."""
    for policy in ("drop", "supersede"):
        obs, (b1, b2) = two_batches(planted, policy, order=("long", "short"))
        assert len(b1[0]) == 1 and b2 == ([], {}) and obs.absorbed == 1 and obs.superseded == 0


def test_superseded_tracks_are_written_and_left_out_on_reading(planted, tmp_path):
    from wdf.analysis.track_reconstruction import load_tracks, save_tracks, superseded_ids

    _, (s1, s2) = two_batches(planted, "supersede")
    events = pd.DataFrame(s1[0] + s2[0], columns=TRACK_EVENT_COLUMNS)
    waveforms = {**s1[1], **s2[1]}
    save_tracks(events, waveforms, str(tmp_path / "e.csv"), str(tmp_path / "w.npz"))
    read, wf = load_tracks(str(tmp_path / "e.csv"), str(tmp_path / "w.npz"))
    assert read["cluster_id"].tolist() == [1] and list(wf) == [1]
    assert read["track_supersedes"][0] == [0]
    every, wf_all = load_tracks(str(tmp_path / "e.csv"), str(tmp_path / "w.npz"), superseded="keep")
    assert every["cluster_id"].tolist() == [0, 1] and sorted(wf_all) == [0, 1]
    assert superseded_ids(every) == {0}
    with pytest.raises(ValueError):
        load_tracks(str(tmp_path / "e.csv"), superseded="all")


def test_the_anchored_path_has_no_preferred_direction():
    """The penalty is symmetric, and a falling track, the rising one played
    backward in time over the same frequencies, gets the same path and score."""
    ts, _ = settings()
    rs = ts.ridge
    from dataclasses import replace

    pen = rs.penalty()
    np.testing.assert_allclose(pen, pen.T)
    rs = replace(rs, drift=1.0, xdrop=5.0)
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((200, rs.n_states))
    for k in range(60, 120):
        Z[k, 2 + (k - 60) // 8] += 6.0
    up = anchored_path(Z, 90, 2 + 30 // 8, rs)
    down = anchored_path(Z[::-1], 199 - 90, 2 + 30 // 8, rs)
    assert up[2] == pytest.approx(down[2])
    np.testing.assert_array_equal(up[0], 199 - down[0][::-1])
    np.testing.assert_array_equal(up[1], down[1][::-1])
    assert up[0][0] <= 62 and up[0][-1] >= 117


def test_the_track_penalty_follows_the_chirp_scaling():
    """The allowance grows with frequency as an inspiral's sweep, is bounded
    by TrackAllowanceMax, and beyond it a move costs TrackLambda per octave."""
    from wdf.analysis.track_reconstruction import chirp_allowance

    ts, par = settings(TrackMcRange=[1.0, 30.0])
    rs = ts.ridge
    assert (rs.lam, par.TrackLambda, par.TrackMcRange) == (3.7, 3.7, [1.0, 30.0])
    c = rs.allowances()
    assert np.all(np.diff(np.diag(c)) >= 0) and c.max() <= rs.allowance_max
    assert chirp_allowance(30.0, 30.0, rs.slot_s) < 0.1 < chirp_allowance(60.0, 30.0, rs.slot_s)
    assert chirp_allowance(30.0, 1.0, rs.slot_s) < chirp_allowance(30.0, 30.0, rs.slot_s)
    pen = rs.penalty()
    assert pen[0, 4] == pytest.approx(3.7 * (1.0 - c[0, 4]))
    with pytest.raises(ValueError, match="TrackMcRange"):
        settings(TrackMcRange=[30.0, 1.0])


#: Largest median and 90th percentile of |d log2 f| between a planted chirp's
#: track and its f(t), over the chirp's loud part, octaves (states are 0.25
#: octave wide).
WANDER_MEDIAN, WANDER_P90 = 0.15, 0.35


@pytest.mark.parametrize("at", [81.0, 82.0, 82.6])
def test_a_planted_chirp_track_does_not_wander(planted, at):
    """Seeded anywhere on the chirp, the track stays on its f(t) and does not
    run into the noise before it."""
    seed = trigger(at - 0.5, at, float(planted_f(T0 + at)))
    (_, _, tracks), ts, _ = run(planted, "LocalCos64", [seed], RidgeBasis="LocalCos64")
    t = tracks[0]
    loud = (t.times() > T0 + CHIRP_AT + 0.5) & (t.times() < T0 + CHIRP_AT + CHIRP_S - 0.5)
    err = np.abs(np.log2(t.frequencies()[loud] / planted_f(t.times()[loud])))
    reach = min(at + ts.horizon, CHIRP_AT + CHIRP_S - 0.5) - (CHIRP_AT + 0.5)
    assert loud.sum() > 0.8 * reach / ts.ridge.slot_s
    assert np.median(err) < WANDER_MEDIAN and np.percentile(err, 90) < WANDER_P90
    assert t.t_start > T0 + CHIRP_AT - 1.0 and t.t_end < T0 + CHIRP_AT + CHIRP_S + 1.0


# ================================================================ online
pytest.importorskip("py4tsa")

from test_online_worker import T0 as OT0, FS as OFS, SECONDS as OSECONDS, parameters, strain  # noqa: E402

from wdf.processes.wdfUnitOnline import ArraySource, wdfUnitOnline  # noqa: E402


def track_parameters(**kw):
    return parameters(WhiteningMode="zerophase", ZeroPhaseDuration=16.0, Reconstruction="track",
                      TrackBasis="LocalCos64", **kw)


@pytest.fixture(scope="module")
def trun():
    return wdfUnitOnline(track_parameters()).process(ArraySource(strain(), OT0, OFS), OT0 + 26.0,
                                                     OT0 + OSECONDS)


def test_the_track_needs_the_ridge_and_a_ring_long_enough():
    with pytest.raises(ValueError, match="Ridge"):
        wdfUnitOnline(track_parameters(Ridge=False))
    with pytest.raises(ValueError, match="ring"):
        wdfUnitOnline(track_parameters(TrackLookback=20.0))


def test_the_track_latency_is_declared(trun):
    par = trun.parameters
    assert par.TrackLatency_s == pytest.approx(par.Latency_s + par.TrackHorizon
                                               + par.RidgeSlot
                                               + (par.RidgeWindow - par.RidgeOverlap / 2) / 2048.0)
    assert trun.tracks and par.TrackCount == len(trun.tracks)
    assert par.TrackOverlapPolicy == "drop" and par.TrackSuperseded == 0
    assert par.TrackSeedsCarried == 0 and par.TrackSeedsAbsorbed >= 0
    for row in trun.tracks:
        assert row["latency"] <= par.TrackLatency_s + par.len + 1e-9
        assert row["emitted_at"] == pytest.approx(row["data_time"] + par.Latency_s)


def test_a_track_follows_the_online_chirp(trun):
    on = [r for r in trun.tracks if r["track_t_start"] < OT0 + 131.0 and r["track_t_end"] > OT0 + 128.0]
    assert on and max(r["track_t_end"] - r["track_t_start"] for r in on) > 1.5
    assert all(r["cluster_id"] in trun.track_waveforms for r in trun.tracks)


def test_altering_the_input_after_t_changes_no_track_before_t_minus_its_latency(trun):
    """No track whose seeds end at or before t - TrackLatency_s reads the input
    after t; and nothing emitted at or before t changes."""
    K = trun.parameters.TrackLatency_s
    t = OT0 + 145.0
    x = strain()
    i = int(round((t - OT0) * OFS))
    x[i:] = np.random.default_rng(9).standard_normal(x.size - i) * 3.0
    altered = wdfUnitOnline(track_parameters()).process(ArraySource(x, OT0, OFS), OT0 + 26.0,
                                                        OT0 + OSECONDS)
    assert trun.before(t) == altered.before(t)

    def early(result):
        rows = [{k: v for k, v in r.items() if k not in ("emitted_at", "latency", "data_time")}
                for r in result.tracks if r["seed_end"] <= t - K]
        return pd.DataFrame(rows)

    a, b = early(trun), early(altered)
    assert len(a) and any(a["track_t_end"] > OT0 + 128.0)
    pd.testing.assert_frame_equal(a, b)
    for cid in a["cluster_id"]:
        np.testing.assert_array_equal(trun.track_waveforms[cid][1], altered.track_waveforms[cid][1])
    assert trun.before(OT0 + OSECONDS) != altered.before(OT0 + OSECONDS)


def test_the_online_worker_writes_the_tracks(tmp_path):
    from wdf.analysis.track_reconstruction import load_tracks

    par = track_parameters(outdir=str(tmp_path) + "/", run="r", SearchName="trk")
    worker = wdfUnitOnline(par)
    result = worker.segmentProcess((OT0 + 26.0, OT0 + OSECONDS), ArraySource(strain(), OT0, OFS))
    events, waveforms = load_tracks(worker.dir + "TrackEvents-Searchtrk.csv",
                                    worker.dir + "TrackWaveforms-Searchtrk.npz")
    assert len(events) == len(result.tracks) and set(waveforms) == set(events["cluster_id"])
    assert isinstance(events["track_f"][0], list)


# ================================================================ offline
def test_the_offline_worker_reconstructs_tracks(tmp_outdir):
    from conftest import run_segment_process

    from wdf.processes.wdfUnitOffline import wdfUnitOffline

    common = dict(worker_class=wdfUnitOffline, Ridge=True, RidgeWindow=512, RidgeOverlap=256,
                  RidgeBackground=16.0, RidgeBackgroundMinimum=4.0, RidgeBasis="LocalCos16",
                  ZeroPhaseFilter="truncated", ZeroPhaseDuration=1.0, BandPassFilter="butterworth",
                  LowFrequencyCut=12.0)
    for sub in ("p", "t"):
        os.makedirs(os.path.join(tmp_outdir, sub), exist_ok=True)
    plain = run_segment_process(os.path.join(tmp_outdir, "p") + "/", **common)
    track = run_segment_process(os.path.join(tmp_outdir, "t") + "/", Reconstruction="track",
                                TrackLookback=0.5, TrackHorizon=0.25, **common)
    pd.testing.assert_frame_equal(plain, track)
    found = glob.glob(os.path.join(tmp_outdir, "t", "**", "TrackEvents.csv"), recursive=True)
    assert found and os.path.exists(found[0].replace("TrackEvents.csv", "TrackWaveforms.npz"))
    with pytest.raises(ValueError, match="Ridge"):
        wdfUnitOffline(_offline_par(Reconstruction="track"))


def _offline_par(**kw):
    from conftest import TEST_PARAMS

    from wdf.config.Parameters import Parameters

    return Parameters(**dict(TEST_PARAMS, sampling=4096, file="x.gwf", **kw))


def test_the_track_calibration_seeds_and_overlap_memory_are_parameters():
    s, par = settings()
    assert s.ridge.track_calibration_seeds == 500 and par.TrackCalibrationSeeds == 500
    assert s.overlap_memory == 256 and par.TrackOverlapMemory == 256
    s, _ = settings(TrackCalibrationSeeds=50, TrackOverlapMemory=8)
    assert s.ridge.track_calibration_seeds == 50 and s.overlap_memory == 8
    with pytest.raises(ValueError, match="TrackCalibrationSeeds"):
        settings(TrackCalibrationSeeds=0)
    with pytest.raises(ValueError, match="TrackOverlapMemory"):
        settings(TrackOverlapMemory=0)
