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
#: Start of the short sweep of the merger tests, s.
MERGER_AT = 60.0
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


#: The default: no coherence cut.
COHERENCE_MIN = 0.0


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
    fails = (noise["track_compatible"] < 1) | (noise["track_lag_physical"] < 1)
    assert fails.all()
    admitted = finder(detectors).find(detectors[0])
    assert len(admitted) < len(every)
    assert (admitted["gps_candidate"] - T0 - 97.0).between(0, 2).sum() == 0


def test_an_incompatible_track_is_rejected():
    """L1 holds a chirp falling over the same band: the f(t) disagree, and
    admission (a) rejects the pair."""
    events, waveforms = build({"H1": lambda n: chirp(n, 0.0),
                               "L1": lambda n: chirp(n, -LAG, 200.0, 40.0)})
    assert len(on_chirp(finder((events, waveforms), gate="classic").find(events)))
    assert on_chirp(finder((events, waveforms)).find(events)).empty


def test_an_incoherent_partner_ranks_far_below_the_coherent_one(detectors):
    """L1 without the chirp: its noise track may pass (a) and (b); with no
    coherence cut it is then admitted, and its track_statistic is a small
    fraction of the coherent pair's, so the time slides rank it as noise."""
    coherent = on_chirp(finder(detectors).find(detectors[0])).iloc[0]
    events, waveforms = build({"H1": lambda n: chirp(n, 0.0), "L1": lambda n: 0.0})
    alone = on_chirp(finder((events, waveforms)).find(events))
    assert alone.empty or alone["track_statistic"].max() < 0.1 * coherent["track_statistic"]
    assert alone.empty or alone["track_coherence"].max() < 0.2


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


@pytest.mark.parametrize("percentile", [90.0, 99.0])
def test_the_calibration_is_the_stated_percentile_of_slid_noise(noise, percentile):
    """The threshold is the percentile of |Gamma| over the slid pairs passing
    (a) and (b); on a second, independent set of slides about the same
    fraction lies above it."""
    from wdf.analysis.track_coincidence import calibrate_coherence_min

    events, waveforms = noise
    settings = TrackCoincidenceSettings(percentile=percentile, calibration_slides=SLIDES,
                                        calibration_minimum=20)
    bounds = {i: (T0, T0 + NOISE_SECONDS) for i in events}
    value, record = calibrate_coherence_min(events, waveforms, FS, settings, bounds, STEP)
    assert record["population"] == "tracks" and record["pairs"] >= 20
    assert record["slides"] == SLIDES and record["percentile"] == percentile
    open_finder = TrackCoincidenceFinder(IndexedCoincidenceFinder(), events, waveforms, FS,
                                         TrackCoincidenceSettings(), gate="tracks")
    again = TimeSlideFAR(open_finder, FARConfig(n_slides=SLIDES, min_shift_s=STEP, seed=1))
    same = again.background_distribution(events, bounds)["track_coherence"].to_numpy(float)
    assert value == pytest.approx(np.percentile(same[np.isfinite(same)], percentile))
    other = TimeSlideFAR(open_finder, FARConfig(n_slides=SLIDES, min_shift_s=STEP, seed=7))
    fresh = other.background_distribution(events, bounds)["track_coherence"].to_numpy(float)
    above = float(np.mean(fresh[np.isfinite(fresh)] > value))
    assert above <= (100.0 - percentile) / 100.0 + 0.1


def test_with_too_few_compatible_noise_pairs_the_calibration_says_so(noise):
    """With an admission no slid noise pair passes (an absolute floor of 1 s,
    longer than the noise tracks), the calibration falls back to every slid
    pair and records it."""
    from wdf.analysis.track_coincidence import calibrate_coherence_min

    events, waveforms = noise
    settings = TrackCoincidenceSettings(calibration_slides=SLIDES, min_common_s=1.0)
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


def merger(n, delay, at=MERGER_AT, seconds=0.2, f0=100.0, f1=300.0, snr=20.0):
    """A short sweep, as a compact binary's merger: `seconds` long."""
    t = np.arange(n) / FS - at - delay
    inside = (t >= 0) & (t < seconds)
    f = f0 + (f1 - f0) * np.clip(t, 0, seconds) / seconds
    c = np.where(inside, np.sin(2 * np.pi * np.cumsum(f) / FS)
                 * np.sin(np.pi * np.clip(t, 0, seconds) / seconds), 0.0)
    return c * snr / np.sqrt(np.sum(c ** 2))


def merger_pair(delay_l1, l1_signal=True, snr=20.0):
    n = int(SECONDS * FS)
    ts = track_settings()
    events, waveforms = {}, {}
    for k, ifo in enumerate(("H1", "L1")):
        x = np.random.default_rng(60 + k).standard_normal(n)
        if ifo == "H1" or l1_signal:
            x = x + merger(n, 0.0 if ifo == "H1" else delay_l1, snr=snr)
        seeds = [trigger(MERGER_AT - 0.3, MERGER_AT + 0.1, 200.0)]
        events[ifo], waveforms[ifo] = detector(ifo, x, seeds, ts)
    return events, waveforms


def test_a_short_merger_pair_is_admitted():
    """Two tracks of a 0.2 s sweep share less than 0.25 s; the relative rule
    admits them: more than half the shorter track and many cycles."""
    events, waveforms = merger_pair(0.005)
    for ev in events.values():
        assert (ev["track_t_end"] - ev["track_t_start"]).max() < 0.5
    table = finder((events, waveforms)).find(events)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["track_common_s"] < 0.25 and row["track_common_fraction"] >= 0.5
    assert row["track_common_cycles"] >= 2.0
    assert row["dt_s"] == pytest.approx(-0.005, abs=1e-3)


def test_a_merger_against_noise_is_rejected():
    events, waveforms = merger_pair(0.005, l1_signal=False)
    assert finder((events, waveforms)).find(events).empty


@pytest.mark.parametrize("h1_minus_l1", [0.0069, -0.0069])
def test_the_lag_sign_is_first_minus_second_detector(h1_minus_l1):
    """`dt_s` on the track path is t_H1 - t_L1 for ifos_involved "H1,L1", as
    the classical difference of instants: positive when L1 leads, as at
    GW150914."""
    events, waveforms = build({"H1": lambda n: chirp(n, h1_minus_l1),
                               "L1": lambda n: chirp(n, 0.0)})
    row = on_chirp(finder((events, waveforms)).find(events)).iloc[0]
    assert row["ifos_involved"] == "H1,L1"
    assert row["dt_s"] == pytest.approx(h1_minus_l1, abs=1e-3)
    one = pd.DataFrame(dict(ifo=["H1"], gpsPeak=[T0 + h1_minus_l1], gpsStart=[T0],
                            duration=[0.1], freqMin=[50.0], freqMax=[100.0], EnWDF=[10.0]))
    two = one.assign(ifo="L1", gpsPeak=T0)
    edge = IndexedCoincidenceFinder().candidate_edge_array(one, two)
    assert edge[0, 3] == pytest.approx(h1_minus_l1, abs=1e-6)


def test_a_coincident_pair_below_a_calibrated_threshold_is_admitted_by_default():
    """With no coherence cut (the default), a planted coincident merger whose
    |Gamma| is below a threshold calibrated at the 99th percentile of noise
    is admitted, ranked by track_statistic; the same threshold set as
    TrackCoherenceMin rejects it."""
    events, waveforms = merger_pair(0.005, snr=12.0)
    assert TrackCoincidenceSettings().coherence_min == 0.0
    table = TrackCoincidenceFinder(IndexedCoincidenceFinder(), events, waveforms, FS,
                                   TrackCoincidenceSettings()).find(events)
    assert len(table) == 1
    row = table.iloc[0]
    threshold = min(0.95, row["track_coherence"] + 0.05)
    assert row["track_statistic"] > 0 and row["track_coherence"] < threshold
    cut = TrackCoincidenceFinder(IndexedCoincidenceFinder(), events, waveforms, FS,
                                 TrackCoincidenceSettings(coherence_min=threshold)).find(events)
    assert cut.empty


def test_the_coherence_cut_is_read_from_the_parameters():
    from types import SimpleNamespace

    for given, value, written in ((None, 0.0, "none"), ("none", 0.0, "none"),
                                  ("calibrate", None, "calibrate"), (0.4, 0.4, 0.4)):
        par = SimpleNamespace(TrackCoherenceMin=given)
        assert TrackCoincidenceSettings.from_parameters(par).coherence_min == value
        assert par.TrackCoherenceMin == written
