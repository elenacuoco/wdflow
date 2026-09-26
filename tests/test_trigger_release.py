"""The single-threshold release, from two detectors' triggers to a rate.

Every check here is on triggers whose answer is known: a burst placed in both
detectors at a stated delay, among isolated triggers of noise placed at random.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("py4tsa")

from _synth import forward
from wdf.analysis.coefficients import from_dense
from wdf.analysis.metaparameters import meta_features
from wdf.analysis.robust_events import FARConfig
from wdf.analysis.trigger_release import (TriggerReleaseConfig, check_scorer,
                                          common_intervals, release)

FS, WINDOW, OVERLAP = 2048.0, 64, 16
GPS0, SPAN = 1000000000.0, 240.0
BURST, DELAY = GPS0 + 120.0, 0.005


def _row(index, value, gps, ifo, wave="DaubC12"):
    return dict(meta_features(index, value, WINDOW, FS, 1.0, gps=gps), gps=gps,
                EnWDF=float(np.linalg.norm(value)), sigma=1.0, wave=wave,
                n_coeff=WINDOW, fs=FS, wt_index=index, wt_value=value, ifo=ifo,
                stride=(WINDOW - OVERLAP) / FS)


def _triggers(ifo, onset, seed):
    """Isolated windows of noise, one every few seconds, and a burst at
    `onset` cut into the windows it touches and kept above a tenth of its
    largest coefficient, as a threshold would keep it."""
    rng = np.random.default_rng(seed)
    rows = []
    for gps in GPS0 + 2.0 + np.sort(rng.uniform(0.0, SPAN - 4.0, 40)):
        if abs(gps - BURST) < 2.0:
            continue
        index = np.sort(rng.choice(np.arange(8, WINDOW), 3, replace=False)).astype(np.uint16)
        rows.append(_row(index, rng.normal(0.0, 3.0, 3).astype(np.float32),
                         gps - (gps - GPS0) % ((WINDOW - OVERLAP) / FS), ifo))
    t = np.arange(int(0.2 * FS)) / FS
    burst = 40.0 * np.exp(-((t - 0.1) / 0.01) ** 2) * np.sin(2 * np.pi * 200.0 * t)
    step = WINDOW - OVERLAP
    first = int(round((onset - 0.1 - GPS0) * FS)) // step * step
    padded = np.zeros(first + len(burst) + 3 * WINDOW)
    at = int(round((onset - 0.1 - GPS0) * FS))
    padded[at:at + len(burst)] = burst
    for start in range(first - WINDOW, first + len(burst) + WINDOW, step):
        coefficients = forward(padded[start:start + WINDOW], "DaubC12")
        coefficients[np.abs(coefficients) < 0.1 * np.abs(coefficients).max()] = 0.0
        index, value = from_dense(coefficients)
        if len(index):
            rows.append(_row(index, value, GPS0 + start / FS, ifo))
    return pd.DataFrame(rows).sort_values("gps").reset_index(drop=True)


@pytest.fixture(scope="module")
def released():
    triggers = {"H1": _triggers("H1", BURST + DELAY, 1),
                "L1": _triggers("L1", BURST, 2)}
    spans = {ifo: [(GPS0, GPS0 + SPAN)] for ifo in triggers}
    config = TriggerReleaseConfig(window=WINDOW, overlap=OVERLAP,
                                  local_scale_neighbours=None,
                                  slides=FARConfig(n_slides=10, min_shift_s=5.0),
                                  ranking=("network_morphology", "network_enwdf"),
                                  minimum_interval_s=60.0)
    return release(triggers, spans, config)


def test_an_event_is_measured_on_its_own_reconstruction(released):
    stage = released.stages["H1"]
    for event in stage.events.itertuples():
        cluster = stage.coefficients(event.cluster_id)
        assert event.EnWDF == pytest.approx(cluster.enwdf(), rel=1e-9)
        # The instant is sought within one block of the ranked tile.
        assert abs(event.gpsEnvelope - event.gpsPeak) <= 0.5 * WINDOW / FS + 1e-9


def test_the_burst_is_the_first_pair_and_is_timed_at_its_delay(released):
    top = released.candidates.iloc[0]
    assert abs(top.gps_candidate - BURST) < 0.1
    assert top["rank"] == 1
    assert top.lag_from == "correlation"
    # `lag_s` is t_i - t_j, and the burst reaches H1 later.
    sign = 1.0 if top.ifo_i == "H1" else -1.0
    assert sign * top.lag_s == pytest.approx(DELAY, abs=1.0 / FS)
    assert bool(top.physical)


def test_the_rate_is_read_on_the_slides_of_the_common_stretch(released):
    assert released.intervals == [(GPS0, GPS0 + SPAN, ("H1", "L1"))]
    assert released.livetime_s == pytest.approx(10 * SPAN)
    assert released.observed_s == pytest.approx(SPAN)
    top = released.candidates.iloc[0]
    # A finite background cannot establish a zero rate.
    assert top.far_per_day_network_morphology >= 86400.0 / released.livetime_s - 1e-9
    assert set(released.background["slide_index"]) <= set(range(10))


def test_the_trace_says_where_a_transient_is_held(released):
    here = released.trace(BURST, 0.2).set_index("ifo")
    assert (here.triggers > 0).all() and (here.events > 0).all()
    assert (here.best_rank == 1).all()
    empty = released.trace(GPS0 + 1.0, 0.1).set_index("ifo")
    assert (empty.triggers == 0).all() and (empty.pairs == 0).all()
    assert len(released.at(BURST, 0.05)) >= 1
    listed = released.events_near(BURST, 0.2)
    assert set(listed.ifo) == {"H1", "L1"}
    covering = listed[(listed.start_s <= 0.05) & (listed.end_s >= -0.05)]
    assert set(covering.ifo) == {"H1", "L1"}
    assert covering.best_rank.min() == 1


def test_the_common_stretches_are_cut_where_a_detector_starts_or_stops():
    spans = {"H1": [(0.0, 100.0)], "L1": [(10.0, 40.0), (50.0, 120.0)]}
    assert common_intervals(spans) == [(10.0, 40.0, ("H1", "L1")),
                                       (50.0, 100.0, ("H1", "L1"))]
    assert common_intervals(spans, minimum_s=35.0) == [(50.0, 100.0, ("H1", "L1"))]


def test_a_model_fitted_on_another_representation_is_refused(released):
    class Layer:
        def __init__(self, n_in, n_out):
            self.in_features, self.out_features = n_in, n_out

    class Scorer:
        encoder = [Layer(642, 16)]
        edge_head = [Layer(2 * 16 + 11, 16)]
        profile_dim = 0

    class Graph:
        node_features = np.zeros((3, 578))
        cross_edge_features = np.zeros((2, 11))

    with pytest.raises(ValueError, match="another representation"):
        check_scorer(Scorer(), Graph())
    Graph.node_features = np.zeros((3, 642))
    check_scorer(Scorer(), Graph())
