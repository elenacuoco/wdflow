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
    for bad in (dict(TrackBasis="Haar"), dict(TrackBasis="CosinePackets30"),
                dict(TrackPacketCost="l2"), dict(TrackHorizon=-1.0)):
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


def test_the_anchored_path_has_no_preferred_direction():
    """The same map mirrored in frequency gives the mirrored path and the
    same score: a falling track scores as a rising one."""
    ts, _ = settings()
    rs = ts.ridge
    from dataclasses import replace

    rs = replace(rs, drift=1.0, xdrop=5.0)
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((200, rs.n_states))
    for k in range(60, 120):
        Z[k, 2 + (k - 60) // 4] += 6.0
    up = anchored_path(Z, 90, 2 + 30 // 4, rs)
    down = anchored_path(Z[:, ::-1], 90, rs.n_states - 1 - (2 + 30 // 4), rs)
    assert up[2] == pytest.approx(down[2])
    np.testing.assert_array_equal(up[0], down[0])
    np.testing.assert_array_equal(up[1], rs.n_states - 1 - down[1])
    assert up[0][0] <= 62 and up[0][-1] >= 117


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
