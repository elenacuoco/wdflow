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
CHIRP_AT, CHIRP_S, SNR = 80.0, 3.0, 40.0
#: H1 - L1 arrival difference of the planted chirp, s.
LAG = -0.0073


def chirp(n, delay, f0=40.0, f1=200.0):
    t = np.arange(n) / FS - CHIRP_AT - delay
    inside = (t >= 0) & (t < CHIRP_S)
    f = f0 + (f1 - f0) * np.clip(t, 0, CHIRP_S) / CHIRP_S
    c = np.where(inside, np.sin(2 * np.pi * np.cumsum(f) / FS)
                 * np.sin(np.pi * np.clip(t, 0, CHIRP_S) / CHIRP_S) ** 2, 0.0)
    return c * SNR / np.sqrt(np.sum(c ** 2))


def trigger(gps, peak, f):
    return dict(gps=T0 + gps, n_coeff=1024, fs=FS, EnWDF=10.0, freqMin=f / 2, freqMax=f * 2,
                gpsPeak=T0 + peak, freqMean=f, sigma=1.0, snrPeak=5.0)


def track_settings():
    par = SimpleNamespace(window=1024, overlap=256, MinFrequency=16.0, TrackBasis="LocalCos64",
                          RidgeBasis="LocalCos64")
    return TrackSettings.from_parameters(par, FS)


def detector(ifo, x, seeds, ts):
    X, origin, offs, co, sg = stream_cells(x, T0, FS, ts.ridge)
    Z = background_z(X, ts.ridge)
    ev, wf, _ = reconstruct(seeds, Z, X, origin, FS, ts, ifo, co[0.0], sg[0.0],
                            T0 + offs / FS, stream=(T0, x))
    return ev.assign(ifo=ifo), wf


def build(signals):
    """H1 and L1: white noise plus `signals[ifo]`, seeded on the chirp's
    loud part and once in noise only, 16 s later."""
    n = int(SECONDS * FS)
    ts = track_settings()
    events, waveforms = {}, {}
    for k, ifo in enumerate(("H1", "L1")):
        x = np.random.default_rng(10 + k).standard_normal(n) + signals[ifo](n)
        seeds = [trigger(81.5, 82.0, 157.0), trigger(97.5, 98.0, 157.0)]
        events[ifo], waveforms[ifo] = detector(ifo, x, seeds, ts)
    return events, waveforms


@pytest.fixture(scope="module")
def detectors():
    """The chirp in both detectors, L1 later by -LAG."""
    return build({"H1": lambda n: chirp(n, 0.0), "L1": lambda n: chirp(n, -LAG)})


#: A threshold for the tests that do not calibrate one.
COHERENCE_MIN = 0.3


def finder(detectors, gate="full", coherence_min=COHERENCE_MIN):
    events, waveforms = detectors
    return TrackCoincidenceFinder(IndexedCoincidenceFinder(), events, waveforms, FS,
                                  TrackCoincidenceSettings(coherence_min=coherence_min),
                                  gate=gate)


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


def on_chirp(table):
    return table[(table["gps_candidate"] - T0 - CHIRP_AT).between(0, CHIRP_S)]


def test_a_coincident_chirp_is_admitted_at_the_planted_lag(detectors):
    table = finder(detectors).find(detectors[0])
    assert set(TRACK_EDGE_FEATURES) <= set(table.columns)
    on = on_chirp(table)
    assert len(on) == 1 and len(table) == 1
    row = on.iloc[0]
    assert row["dt_s"] == row["track_lag_s"] == pytest.approx(LAG, abs=1e-3)
    assert row["track_coherence"] >= COHERENCE_MIN and row["track_compatible"] == 1.0
    assert row["track_statistic"] == pytest.approx(
        min(row["track_snr_i"], row["track_snr_j"]) * row["track_coherence"])
    assert "network_enwdf" in table and "dt_instant_s" in table


def test_the_noise_pair_is_formed_but_not_admitted(detectors):
    """The two noise tracks, 16 s after the chirp, overlap in time: the
    classical rule forms the pair; the track admission does not."""
    every = finder(detectors, gate="classic").find(detectors[0])
    noise = every[(every["gps_candidate"] - T0 - 97.0).between(0, 2)]
    assert len(noise)
    fails = ((noise["track_compatible"] < 1) | (noise["track_lag_physical"] < 1)
             | (noise["track_coherence"] < COHERENCE_MIN))
    assert fails.all()
    admitted = finder(detectors).find(detectors[0])
    assert len(admitted) < len(every)
    assert (admitted["gps_candidate"] - T0 - 97.0).between(0, 2).sum() == 0


@pytest.mark.parametrize("other", ["falling", "absent"])
def test_incompatible_or_incoherent_tracks_are_rejected(other):
    """L1 holds a chirp falling over the same band (incompatible f(t)), or no
    chirp at all (its track is noise): no pair is admitted on the chirp."""
    signal = {"falling": lambda n: chirp(n, -LAG, 200.0, 40.0), "absent": lambda n: 0.0}[other]
    events, waveforms = build({"H1": lambda n: chirp(n, 0.0), "L1": signal})
    every = finder((events, waveforms), gate="classic").find(events)
    assert len(on_chirp(every))
    assert on_chirp(finder((events, waveforms)).find(events)).empty


def test_slid_noise_is_incoherent_and_the_displacement_is_applied(detectors):
    """L1 moved 16 s back: its noise track meets H1's chirp. The finder moves
    L1's track and waveform with it: the pair is incoherent, and rejected."""
    events, _ = detectors
    slid = dict(events)
    l1 = events["L1"].copy()
    for column in ("gpsPeak", "gpsCentroid", "gpsStart"):
        l1[column] = l1[column] - 16.0
    slid["L1"] = l1
    row = on_chirp(finder(detectors, gate="classic").find(slid)).iloc[0]
    assert row["track_coherence"] < 0.2 and row["track_time_overlap"] > 0.0
    assert abs(row["track_lag_s"]) <= TrackCoincidenceSettings().lag_search_s + 1e-9
    assert on_chirp(finder(detectors).find(slid)).empty


def test_time_slides_run_on_the_track_finder(detectors):
    events, _ = detectors
    far = TimeSlideFAR(finder(detectors), FARConfig(n_slides=4, min_shift_s=4.0))
    bounds = {i: (T0, T0 + SECONDS) for i in events}
    background = far.background_distribution(events, bounds)
    zero = finder(detectors).find(events)
    ranked = far.rank_candidates(zero, background, SECONDS, score_column="track_statistic")
    assert "far_hz" in ranked and len(background) == 0


def test_the_finder_needs_a_threshold_or_a_calibration(detectors):
    with pytest.raises(ValueError, match="TrackCoherenceMin"):
        finder(detectors, coherence_min=None)


#: Length of the noise streams of the calibration tests, s.
NOISE_SECONDS = 300.0


@pytest.fixture(scope="module")
def noise():
    """H1 and L1 noise, a track seeded every second."""
    n = int(NOISE_SECONDS * FS)
    ts = track_settings()
    events, waveforms = {}, {}
    for k, ifo in enumerate(("H1", "L1")):
        x = np.random.default_rng(40 + k).standard_normal(n)
        seeds = [trigger(t, t + 0.25, 60.0 * 2 ** (j % 3))
                 for j, t in enumerate(np.arange(10.0, NOISE_SECONDS - 10.0, 1.0))]
        events[ifo], waveforms[ifo] = detector(ifo, x, seeds, ts)
    return events, waveforms


#: Slide step of the calibration tests, s: longer than the noise tracks.
STEP = 8.0
#: Slides of the calibration tests.
SLIDES = 30


#: A minimum common duration of one slot, so that slid noise tracks (about
#: 0.1 s long) can pass admission (a).
ONE_SLOT = 0.03125


@pytest.mark.parametrize("percentile", [90.0, 99.0])
def test_the_calibration_is_the_stated_percentile_of_slid_noise(noise, percentile):
    """The threshold is the percentile of |Gamma| over the slid pairs passing
    (a) and (b); on a second, independent set of slides about the same
    fraction lies above it."""
    from wdf.analysis.track_coincidence import calibrate_coherence_min

    events, waveforms = noise
    settings = TrackCoincidenceSettings(percentile=percentile, calibration_slides=SLIDES,
                                        calibration_minimum=20, min_common_s=ONE_SLOT)
    bounds = {i: (T0, T0 + NOISE_SECONDS) for i in events}
    value, record = calibrate_coherence_min(events, waveforms, FS, settings, bounds, STEP)
    assert record["population"] == "tracks" and record["pairs"] >= 20
    assert record["slides"] == SLIDES and record["percentile"] == percentile
    open_finder = TrackCoincidenceFinder(IndexedCoincidenceFinder(), events, waveforms, FS,
                                         TrackCoincidenceSettings(min_common_s=ONE_SLOT),
                                         gate="tracks")
    again = TimeSlideFAR(open_finder, FARConfig(n_slides=SLIDES, min_shift_s=STEP, seed=1))
    same = again.background_distribution(events, bounds)["track_coherence"].to_numpy(float)
    assert value == pytest.approx(np.percentile(same[np.isfinite(same)], percentile))
    other = TimeSlideFAR(open_finder, FARConfig(n_slides=SLIDES, min_shift_s=STEP, seed=7))
    fresh = other.background_distribution(events, bounds)["track_coherence"].to_numpy(float)
    above = float(np.mean(fresh[np.isfinite(fresh)] > value))
    assert above <= (100.0 - percentile) / 100.0 + 0.1


def test_with_too_few_compatible_noise_pairs_the_calibration_says_so(noise):
    """At the default TrackMinCommonDuration no slid noise pair is compatible:
    the calibration falls back to every slid pair and records it."""
    from wdf.analysis.track_coincidence import calibrate_coherence_min

    events, waveforms = noise
    settings = TrackCoincidenceSettings(calibration_slides=SLIDES)
    bounds = {i: (T0, T0 + NOISE_SECONDS) for i in events}
    value, record = calibrate_coherence_min(events, waveforms, FS, settings, bounds, STEP)
    assert record["population"] == "classic" and record["pairs"] > 100
    assert 0.0 < value < 1.0


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
                                      TrackCoincidenceSettings(coherence_min=0.0), scorer=model,
                                      builder=builder, coefficients=maps).find(events)

    table = scored(N_EDGE_FEATURES + len(TRACK_EDGE_FEATURES))
    assert {"gnn_logit", "track_coherence", "ifo_i", "cluster_j"} <= set(table.columns)
    with pytest.raises(ValueError, match="cross_edge_dim"):
        scored(N_EDGE_FEATURES)
