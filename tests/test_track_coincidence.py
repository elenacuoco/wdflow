"""Stage 3 of the track path: track compatibility and waveform coherence on
the classical finder's admitted pairs, under time slides."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wdf.analysis.ridge_tracking import background_z, stream_cells
from wdf.analysis.robust_events import FARConfig, IndexedCoincidenceFinder, TimeSlideFAR
from wdf.analysis.track_coincidence import (TRACK_EDGE_FEATURES, TrackCoincidenceFinder,
                                            TrackCoincidenceSettings, waveform_coherence)
from wdf.analysis.track_reconstruction import TrackSettings, reconstruct

FS, T0, SECONDS = 2048.0, 1.0e9, 128
CHIRP_AT, CHIRP_S, SNR = 80.0, 3.0, 25.0
#: H1 - L1 arrival difference of the planted chirp, s.
LAG = -0.0073


def chirp(n, delay):
    t = np.arange(n) / FS - CHIRP_AT - delay
    inside = (t >= 0) & (t < CHIRP_S)
    f = 40.0 + 160.0 * np.clip(t, 0, CHIRP_S) / CHIRP_S
    c = np.where(inside, np.sin(2 * np.pi * np.cumsum(f) / FS)
                 * np.sin(np.pi * np.clip(t, 0, CHIRP_S) / CHIRP_S) ** 2, 0.0)
    return c * SNR / np.sqrt(np.sum(c ** 2))


def trigger(gps, peak, f):
    return dict(gps=T0 + gps, n_coeff=1024, fs=FS, EnWDF=10.0, freqMin=f / 2, freqMax=f * 2,
                gpsPeak=T0 + peak, freqMean=f, sigma=1.0, snrPeak=5.0)


@pytest.fixture(scope="module")
def detectors():
    """H1 and L1, white noise, the chirp in both (L1 later by -LAG); each
    detector also holds a track seeded in noise only, 16 s later."""
    n = int(SECONDS * FS)
    par = SimpleNamespace(window=1024, overlap=256, MinFrequency=16.0, TrackBasis="LocalCos64")
    ts = TrackSettings.from_parameters(par, FS)
    events, waveforms = {}, {}
    for k, (ifo, delay) in enumerate((("H1", 0.0), ("L1", -LAG))):
        x = np.random.default_rng(10 + k).standard_normal(n) + chirp(n, delay)
        X, origin, offs, co, sg = stream_cells(x, T0, FS, ts.ridge)
        Z = background_z(X, ts.ridge)
        seeds = [trigger(81.5, 82.0, 157.0), trigger(97.5, 98.0, 157.0)]
        ev, wf, _ = reconstruct(seeds, Z, X, origin, FS, ts, ifo, co[0.0], sg[0.0],
                                T0 + offs / FS, stream=(T0, x))
        events[ifo], waveforms[ifo] = ev.assign(ifo=ifo), wf
    return events, waveforms


def finder(detectors):
    events, waveforms = detectors
    return TrackCoincidenceFinder(IndexedCoincidenceFinder(), events, waveforms, FS,
                                  TrackCoincidenceSettings())


def test_the_coherence_reads_a_sub_sample_lag_and_its_sign():
    n = 4096
    t = np.arange(n) / FS
    a = np.sin(2 * np.pi * 120 * t) * np.exp(-((t - 1.0) / 0.1) ** 2)
    b = np.interp(t - 0.0037, t, a)
    lag, coherence, phase = waveform_coherence((T0, a), (T0, b), FS, 0.02)
    assert lag == pytest.approx(-0.0037, abs=2e-4) and coherence > 0.95
    lag2, _, phase2 = waveform_coherence((T0, -a), (T0, b), FS, 0.02)
    assert lag2 == pytest.approx(lag, abs=2e-4)
    assert abs(abs(phase2 - phase) - np.pi) < 0.1


def test_a_coincident_chirp_has_the_planted_lag_and_is_coherent(detectors):
    table = finder(detectors).find(detectors[0])
    assert set(TRACK_EDGE_FEATURES) <= set(table.columns)
    on = table[(table["gps_candidate"] - T0 - CHIRP_AT).between(0, CHIRP_S)]
    assert len(on) == 1
    row = on.iloc[0]
    assert row["track_lag_s"] == pytest.approx(LAG, abs=1e-3)
    assert row["track_coherence"] > 0.45 and row["track_compatible"] == 1.0
    assert row["track_time_overlap"] > 0.8


def test_slid_noise_is_incoherent_and_the_displacement_is_applied(detectors):
    """L1 moved 16 s back: its noise track meets H1's chirp. The finder moves
    L1's track and waveform with it: the pair is compared at a lag within the
    light travel time, and is incoherent."""
    events, _ = detectors
    slid = dict(events)
    l1 = events["L1"].copy()
    for column in ("gpsPeak", "gpsCentroid", "gpsStart"):
        l1[column] = l1[column] - 16.0
    slid["L1"] = l1
    table = finder(detectors).find(slid)
    on = table[(table["gps_candidate"] - T0 - CHIRP_AT).between(0, CHIRP_S)]
    assert len(on) == 1
    row = on.iloc[0]
    assert row["track_coherence"] < 0.2
    assert abs(row["track_lag_s"]) <= TrackCoincidenceSettings().window("H1", "L1") + 1e-9
    assert row["track_time_overlap"] > 0.0


def test_time_slides_run_on_the_track_finder(detectors):
    events, _ = detectors
    far = TimeSlideFAR(finder(detectors), FARConfig(n_slides=4, min_shift_s=4.0))
    bounds = {i: (T0, T0 + SECONDS) for i in events}
    background = far.background_distribution(events, bounds)
    zero = finder(detectors).find(events)
    ranked = far.rank_candidates(zero, background, SECONDS, score_column="track_coherence")
    assert "far_hz" in ranked
    if len(background):
        assert background["track_coherence"].max() < zero["track_coherence"].max()


def test_a_gnn_reads_the_track_features_through_its_edge_width(detectors):
    """The track features are appended to the graph's edge features: a model
    of that width scores the pairs, a model of the paper's width is refused."""
    pytest.importorskip("torch_geometric")
    from wdf.analysis.gnn import GNNCoincidenceScorer
    from wdf.analysis.network_graph import N_EDGE_FEATURES, TriggerGraphBuilder
    from wdf.analysis.track_coincidence import track_event_coefficients

    events, waveforms = detectors
    builder = TriggerGraphBuilder(ifos=list(events))
    maps = {i: track_event_coefficients(events[i], waveforms[i], FS, 1024) for i in events}
    node_dim = builder.build(events, maps).node_features.shape[1]

    def scored(width):
        model = GNNCoincidenceScorer(node_dim, cross_edge_dim=width, device="cpu")
        return TrackCoincidenceFinder(IndexedCoincidenceFinder(), events, waveforms, FS,
                                      scorer=model, builder=builder, coefficients=maps).find(events)

    table = scored(N_EDGE_FEATURES + len(TRACK_EDGE_FEATURES))
    assert {"gnn_logit", "track_coherence", "ifo_i", "cluster_j"} <= set(table.columns)
    with pytest.raises(ValueError, match="cross_edge_dim"):
        scored(N_EDGE_FEATURES)
