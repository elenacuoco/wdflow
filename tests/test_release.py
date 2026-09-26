"""From each detector's triggers to the released candidates.

The chain is judged here on what it must do by construction: search time is
cut where a detector starts or stops, a pair's rank is read on its own events,
the released list is ordered by its rate, the second threshold is translated
from a single threshold the same stage applies, and a signal present in two
detectors reaches the list while the accidental rate stays with the slides.
"""
import numpy as np
import pandas as pd
import pytest

from wdf.analysis.release import (ReleaseConfig, common_intervals, rank,
                                  with_node_statistic)
from wdf.analysis.robust_events import FARConfig


def test_search_time_is_cut_where_a_detector_starts_or_stops():
    spans = {"H1": [(0.0, 100.0)], "L1": [(10.0, 60.0), (70.0, 100.0)],
             "V1": [(50.0, 90.0)]}
    found = common_intervals(spans)
    assert found == [(10.0, 50.0, ("H1", "L1")),
                     (50.0, 60.0, ("H1", "L1", "V1")),
                     (60.0, 70.0, ("H1", "V1")),
                     (70.0, 90.0, ("H1", "L1", "V1")),
                     (90.0, 100.0, ("H1", "L1"))]
    assert common_intervals(spans, minimum_s=15.0) == [
        (10.0, 50.0, ("H1", "L1")), (70.0, 90.0, ("H1", "L1", "V1"))]


def test_one_detector_alone_holds_no_stretch():
    assert common_intervals({"H1": [(0.0, 10.0)], "L1": [(20.0, 30.0)]}) == []


def test_a_pair_is_ranked_on_the_weaker_of_its_two_events():
    events = {"H1": pd.DataFrame(dict(EnWDF_significance=[1.0, 7.0])),
              "L1": pd.DataFrame(dict(EnWDF_significance=[4.0, 2.0, 9.0]))}
    pairs = pd.DataFrame(dict(node_i=[1, 0], node_j=[4, 3]))
    out = with_node_statistic(pairs, events, ["H1", "L1"])
    assert out.network_min_significance.tolist() == [7.0, 1.0]


def test_the_list_is_ordered_by_the_rate_of_its_first_statistic():
    candidates = pd.DataFrame(dict(a=[1.0, 5.0, 3.0], b=[9.0, 1.0, 2.0]))
    background = pd.DataFrame(dict(a=[0.5, 2.0, 4.0, 6.0], b=[1.0, 3.0, 5.0, 7.0]))
    background.attrs["total_livetime_s"] = 86400.0
    out = rank(candidates, background, 3600.0, ("a", "b"))
    assert out.a.tolist() == [5.0, 3.0, 1.0]
    # One accidental at or above 5, plus the one a finite background cannot
    # rule out, over one day.
    assert out.far_per_day_a.tolist() == pytest.approx([2.0, 3.0, 4.0])
    assert out.far_per_day_b.iloc[2] == pytest.approx(1.0)


def _triggers(rng, ifo, gps0, seconds, fs, window, overlap, signal_at=None,
              amplitude=0.0):
    """Every window of white noise as the search writes it, a sine-Gaussian
    added where asked, the coefficients kept above the universal threshold."""
    from _synth import triggers_from_signal

    samples = rng.normal(size=int(seconds * fs))
    if signal_at is not None:
        t = np.arange(len(samples)) / fs - (signal_at - gps0)
        samples += amplitude * np.exp(-(t / 0.02) ** 2) * np.sin(2 * np.pi * 180 * t)
    triggers = triggers_from_signal(samples, fs, window, overlap, gps0=gps0,
                                    ifo=ifo)
    floor = np.sqrt(2.0 * np.log(window))
    kept_index, kept_value, energy = [], [], []
    for index, value in zip(triggers.wt_index, triggers.wt_value):
        keep = np.abs(value) >= floor
        kept_index.append(np.asarray(index)[keep])
        kept_value.append(np.asarray(value)[keep])
        energy.append(float(np.sqrt(np.sum(np.asarray(value)[keep] ** 2))))
    out = triggers.assign(wt_index=kept_index, wt_value=kept_value, EnWDF=energy,
                          stride=(window - overlap) / fs)
    return out[out.EnWDF > 0].reset_index(drop=True)


def test_a_signal_in_two_detectors_is_released_above_the_accidentals():
    pytest.importorskip("py4tsa")
    from wdf.analysis.release import release

    rng = np.random.default_rng(11)
    fs, window, overlap, seconds, gps0 = 2048.0, 512, 32, 300.0, 1000.0
    signal_at = gps0 + 151.3
    triggers = {ifo: _triggers(rng, ifo, gps0, seconds, fs, window, overlap,
                               signal_at=signal_at + delay, amplitude=6.0)
                for ifo, delay in (("H1", 0.0), ("L1", 0.004))}
    spans = {ifo: [(gps0, gps0 + seconds)] for ifo in triggers}
    config = ReleaseConfig(reference_threshold=4.2, reported_thresholds=(5.0,),
                           slides=FARConfig(n_slides=20, min_shift_s=4.0),
                           calibration_min_count=50, minimum_interval_s=60.0)
    made = release(triggers, spans, config)

    counts = made.counts().set_index("ifo")
    for ifo in triggers:
        stage = made.stages[ifo]
        # The second threshold passes about as many events as the single
        # threshold it stands for builds.
        assert counts.loc[ifo, "events_passing_4.2"] == pytest.approx(
            counts.loc[ifo, "events_at_4.2"], rel=0.5)
        assert stage.events.passes_event_cut.equals(stage.events["passes_4.2"])
        assert np.isfinite(stage.events.EnWDF_significance).all()
    assert made.livetime_s == pytest.approx(20 * seconds, rel=0.01)

    best = made.candidates.iloc[0]
    assert abs(best.gps_candidate - signal_at) < 0.3
    assert best.physical
    assert best.far_per_day_network_min_significance < (
        made.candidates.far_per_day_network_min_significance.iloc[-1])
    assert set(made.candidates.ifos_involved) == {"H1,L1"}


def test_the_search_s_own_threshold_is_the_reference_population():
    """A single threshold writes exactly the windows at or above it, so the
    reference is read off the low-threshold triggers and not searched again."""
    pytest.importorskip("py4tsa")
    from wdf.analysis.release import reference_events

    rng = np.random.default_rng(12)
    triggers = _triggers(rng, "H1", 1000.0, 120.0, 2048.0, 512, 32)
    config = ReleaseConfig()
    everything = reference_events(triggers, 0.0, config)
    fewer = reference_events(triggers, 4.5, config)
    assert 0 < fewer < everything
    assert reference_events(triggers, 1e9, config) == 0
