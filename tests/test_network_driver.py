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
    pd.testing.assert_frame_equal(result.candidates[expected.columns], expected, check_exact=True)
    expected_rank = result.candidates.groupby("ifos_involved")["network_morphology"].rank(
        ascending=False, method="min").astype(int)
    assert (result.candidates["rank"] == expected_rank).all()
    for ifo in snap.DELAY:
        pd.testing.assert_frame_equal(result.events[ifo],
                                      frozen(f"paper_path_events_{ifo}.parquet"), check_exact=True)
    assert set(vars(par)) - {"NetworkLagCut", "NetworkLagWindow", "NetworkLagTolerance", "NetworkLagScale",
                             "NetworkLagSearch", "NetworkGlitchVeto", "NetworkGlitchTolerance",
                             "NetworkGlitchFactor", "NetworkPenaltySpace"} == set()
    assert {"NetworkLagCut", "NetworkLagWindow", "NetworkLagTolerance", "NetworkLagScale",
                              "NetworkLagSearch", "NetworkGlitchVeto", "NetworkGlitchTolerance",
                              "NetworkGlitchFactor", "NetworkPenaltySpace"} <= set(vars(par))
    assert par.NetworkLagCut == "none" and par.NetworkGlitchVeto == "none"


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
    pd.testing.assert_frame_equal(held[expected.columns], expected, check_exact=True)
    short = NetworkStream(Parameters(NetworkMemory=1.0), list(triggers))
    for ifo, t in triggers.items():
        short.add(ifo, t)
    assert all(len(t) < len(triggers[i]) for i, t in short.tables.items())


def test_the_paper_path_ranks_on_the_morphology_by_default():
    """`network_morphology` is the default rank column of the wavegram path;
    `network_enwdf` stays selectable."""
    from wdf.analysis.network_driver import NetworkSettings, _rank

    assert NetworkSettings.from_parameters(Parameters()).rank_column == "network_morphology"
    chosen = NetworkSettings.from_parameters(Parameters(NetworkRankColumn="network_enwdf"))
    assert chosen.rank_column == "network_enwdf"
    table = pd.DataFrame(dict(ifos_involved=["H1,L1"] * 3, network_morphology=[1.0, 3.0, 2.0],
                              network_enwdf=[30.0, 10.0, 20.0]))
    assert _rank(table, "network_morphology")["rank"].tolist() == [3, 1, 2]
    assert _rank(table, "network_enwdf")["rank"].tolist() == [1, 3, 2]



def test_the_event_instant_is_a_choice():
    """`NetworkEventInstant`: the instant of gps_candidate and of dt_s; the extents per detector."""
    from wdf.analysis.network_driver import run_network

    triggers = {ifo: frozen(f"paper_path_triggers_{ifo}.parquet") for ifo in snap.DELAY}
    out = {}
    for mode, column in (("peak", "gpsPeak"), ("centroid", "gpsCentroid"), ("envelope", "gpsEnvelope"),
                         ("coarse", "gpsPeak")):
        res = run_network(Parameters(NetworkEventInstant=mode), triggers=triggers)
        c = res.candidates
        assert len(c) and (c["event_instant"] == column).all()
        for ifo in snap.DELAY:
            for key in ("gpsStart", "duration", "instant"):
                assert f"{key}_{ifo}" in c
        row = c.iloc[0]
        ev = {i: res.events[i].set_index("cluster_id") for i in snap.DELAY}
        t = {row[f"ifo_{s}"]: float(ev[row[f"ifo_{s}"]].loc[int(row[f"cluster_{s}"]), column]) for s in "ij"}
        assert row["gps_candidate"] == pytest.approx(0.5 * sum(t.values()))
        if mode != "coarse":
            assert row["dt_s"] == pytest.approx(t[row["ifo_i"]] - t[row["ifo_j"]])
        assert row[f"instant_{row['ifo_i']}"] == pytest.approx(t[row["ifo_i"]])
        out[mode] = c
    pd.testing.assert_frame_equal(out["peak"].drop(columns=["event_instant"]),
                                  out["coarse"].drop(columns=["event_instant"])[out["peak"].columns.drop("event_instant")],
                                  check_exact=False) if len(out["peak"]) == len(out["coarse"]) else None


def test_the_envelope_instant_lies_within_its_width_of_the_peak():
    from wdf.analysis.network_driver import run_network

    triggers = {ifo: frozen(f"paper_path_triggers_{ifo}.parquet") for ifo in snap.DELAY}
    res = run_network(Parameters(NetworkEventInstant="envelope", NetworkEnvelopeWidth=0.25), triggers=triggers)
    for ev in res.events.values():
        assert (abs(ev["gpsEnvelope"] - ev["gpsPeak"]) <= 0.125 + 1e-9).all()


def test_an_unknown_event_instant_is_refused():
    from wdf.analysis.network_driver import run_network

    triggers = {ifo: frozen(f"paper_path_triggers_{ifo}.parquet") for ifo in snap.DELAY}
    with pytest.raises(ValueError):
        run_network(Parameters(NetworkEventInstant="median"), triggers=triggers)
