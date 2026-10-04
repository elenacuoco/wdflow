"""The network driver: the paper path unchanged against a snapshot, and both
paths run from the configuration."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("py4tsa")

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
sys.path.insert(0, FIXTURES)

import make_paper_path_snapshot as snap  # noqa: E402

from wdf.config.Parameters import Parameters, network_path  # noqa: E402


def frozen(name):
    return pd.read_parquet(os.path.join(FIXTURES, name))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("paper"))
    return {ifo: snap.run_detector(ifo, out, **snap.FROZEN) for ifo in snap.DELAY}


def test_the_defaults_are_the_paper_path():
    assert network_path(Parameters()) == ("wavegram", "wavegram", "classic")


def test_a_mixed_path_or_a_gnn_without_model_is_refused():
    with pytest.raises(ValueError, match="do not mix"):
        network_path(Parameters(Reconstruction="track"))
    with pytest.raises(ValueError, match="GNNModel"):
        network_path(Parameters(Scorer="gnn"))
    with pytest.raises(ValueError, match="Coincidence"):
        network_path(Parameters(Coincidence="graph"))


def test_the_paper_path_triggers_are_the_snapshot(runs):
    """With the defaults the worker's triggers, as written and as held in
    memory, are those of the snapshot, exactly."""
    for ifo, (triggers, result, worker) in runs.items():
        pd.testing.assert_frame_equal(triggers, frozen(f"paper_path_triggers_{ifo}.parquet"),
                                      check_exact=True)
        pd.testing.assert_frame_equal(pd.DataFrame(result.triggers),
                                      frozen(f"paper_path_online_{ifo}.parquet"), check_exact=True)
        assert not hasattr(worker.par, "Reconstruction") and result.tracks == []


def test_the_paper_path_events_and_network_are_the_snapshot(runs):
    triggers = {ifo: r[0] for ifo, r in runs.items()}
    events, cand = snap.paper_network(triggers)
    for ifo in snap.DELAY:
        pd.testing.assert_frame_equal(events[ifo], frozen(f"paper_path_events_{ifo}.parquet"),
                                      check_exact=True)
    pd.testing.assert_frame_equal(cand, frozen("paper_path_network.parquet"), check_exact=True)


def test_the_driver_reproduces_the_catalogue_paper_path():
    """`run_network` with the defaults gives the network table the catalogue's
    sequence of calls gives, on the snapshot's triggers."""
    from wdf.analysis.network_driver import run_network

    triggers = {ifo: frozen(f"paper_path_triggers_{ifo}.parquet") for ifo in snap.DELAY}
    par = Parameters()
    result = run_network(par, triggers=triggers)
    expected = frozen("paper_path_network.parquet")
    assert len(expected) >= 1
    pd.testing.assert_frame_equal(result.candidates.drop(columns=["rank"]), expected,
                                  check_exact=True)
    assert (result.candidates["rank"] >= 1).all()
    for ifo in snap.DELAY:
        pd.testing.assert_frame_equal(result.events[ifo],
                                      frozen(f"paper_path_events_{ifo}.parquet"), check_exact=True)
    assert vars(par) == {}


def test_the_driver_draws_a_background_when_asked():
    from wdf.analysis.network_driver import run_network

    triggers = {ifo: frozen(f"paper_path_triggers_{ifo}.parquet") for ifo in snap.DELAY}
    result = run_network(Parameters(NetworkSlides=5, NetworkSlideStep=4.0), triggers=triggers)
    assert "far_hz" in result.candidates and result.background is not None


@pytest.fixture(scope="module")
def track_runs(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("track"))
    return {ifo: snap.run_detector(ifo, out, Reconstruction="track", TrackBasis="LocalCos64",
                                   RidgeBasis="LocalCos64", RidgeBackground=32.0,
                                   RidgeBackgroundMinimum=16.0) for ifo in snap.DELAY}


def test_the_track_path_runs_from_the_configuration(track_runs):
    """The workers write their track events; the driver reads them and the
    chirp is a candidate, at the planted lag, coherent."""
    import glob

    from wdf.analysis.network_driver import run_network
    from wdf.analysis.track_reconstruction import load_tracks

    tracks, waveforms = {}, {}
    for ifo, (_, result, worker) in track_runs.items():
        csv = glob.glob(worker.dir + "TrackEvents-*.csv")[0]
        tracks[ifo], waveforms[ifo] = load_tracks(csv, csv.replace("TrackEvents", "TrackWaveforms")
                                                  .replace(".csv", ".npz"))
        assert len(tracks[ifo]) == len(result.tracks) > 0
    par = Parameters(Reconstruction="track", Coincidence="track")
    cand = run_network(par, tracks=tracks, waveforms=waveforms).candidates
    on = cand[(cand["gps_candidate"] - snap.T0 - snap.CHIRP_AT).between(-1, snap.CHIRP_S + 1)]
    assert len(on)
    best = on.loc[on["track_coherence"].idxmax()]
    assert best["track_coherence"] > 0.5 and best["track_compatible"] == 1.0
    assert best["track_lag_s"] == pytest.approx(snap.DELAY["H1"] - snap.DELAY["L1"], abs=2e-3)
    assert best["dt_s"] == best["track_lag_s"] and "network_enwdf" in cand
    assert par.TrackCoherenceMin == "none" and par.TrackCoherenceMinUsed == 0.0
    assert par.TrackTimingJitter == 0.0035
    expected = cand.groupby("ifos_involved")["track_statistic"].rank(ascending=False, method="min")
    assert (cand["rank"] == expected.astype(int)).all()
    assert on.loc[on["track_statistic"].idxmax(), "rank"] == 1


def test_the_network_stream_holds_and_forgets():
    from wdf.analysis.network_driver import NetworkStream

    triggers = {ifo: frozen(f"paper_path_triggers_{ifo}.parquet") for ifo in snap.DELAY}
    stream = NetworkStream(Parameters(NetworkMemory=1e6), list(triggers))
    for ifo, t in triggers.items():
        stream.add(ifo, t)
    held = stream.candidates().candidates
    expected = frozen("paper_path_network.parquet")
    pd.testing.assert_frame_equal(held.drop(columns=["rank"]), expected, check_exact=True)
    short = NetworkStream(Parameters(NetworkMemory=1.0), list(triggers))
    for ifo, t in triggers.items():
        short.add(ifo, t)
    assert all(len(t) < len(triggers[i]) for i, t in short.tables.items())
