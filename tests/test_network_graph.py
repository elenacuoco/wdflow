"""The network stage: what an edge between two detectors carries."""
import numpy as np
import pytest

from wdf.analysis.network_graph import TriggerGraphBuilder


def test_the_edge_time_is_read_on_the_reconstructions():
    """A tile centre cannot resolve the light travel time: in a dyadic
    transform a tile's length is tied to its band, so two detectors whose
    loudest tile falls on different rungs report instants displaced by the
    difference of two tile lengths. The pair's own waveforms say better."""
    fs = 1024.0
    n = 256
    t = np.arange(n) / fs
    wave = np.sin(2 * np.pi * 80.0 * t) * np.exp(-((t - 0.125) / 0.02) ** 2)
    # The same waveform in both, each placed where it sits inside its own
    # event. The second starts 4 ms after its instant and the first 20 ms
    # before its own, so the pair is 24 ms from the alignment the tiles claim.
    prepared = {
        "series": [(-0.020, wave), (0.004, wave)],
        "rates": [fs, fs],
        "reconstruction_offset": {},
    }
    tile_dt = np.array([0.020])
    builder = TriggerGraphBuilder()
    corrected = builder._timed_on_reconstruction(
        prepared, [0], [1], tile_dt, max_lag_s=0.05)

    assert corrected[0] == pytest.approx(-0.004, abs=1.5 / fs)
    # The correction is a property of the pair, so it is measured once and
    # kept on the builder, which outlives any one preparation.
    assert builder._reconstruction_offset
    again = builder._timed_on_reconstruction(
        prepared, [0], [1], tile_dt + 3.0, max_lag_s=0.05)
    assert again[0] == pytest.approx(corrected[0] + 3.0, abs=1e-9)


def test_a_pair_without_a_reconstruction_keeps_the_time_it_came_with():
    prepared = {"series": [None, None], "rates": [1024.0, 1024.0],
                "reconstruction_offset": {}}
    tile_dt = np.array([0.007])
    out = TriggerGraphBuilder()._timed_on_reconstruction(
        prepared, [0], [1], tile_dt, max_lag_s=0.05)
    assert out[0] == pytest.approx(0.007)


def test_a_different_event_set_is_prepared_again():
    """A slide moves events and leaves the set intact, so one preparation
    serves every slide of it. A background slid stretch by stretch is several
    sets, and each is its own preparation."""
    import pandas as pd

    whole = {"H1": pd.DataFrame(dict(cluster_id=[0, 1, 2])),
             "L1": pd.DataFrame(dict(cluster_id=[0, 1]))}
    stretch = {"H1": whole["H1"].iloc[:2], "L1": whole["L1"]}

    order = TriggerGraphBuilder.event_order(whole, ["H1", "L1"])
    assert order == [("H1", 0), ("H1", 1), ("H1", 2), ("L1", 0), ("L1", 1)]
    assert TriggerGraphBuilder.event_order(stretch, ["H1", "L1"]) != order
    # The detector order is part of it: the arrays are laid out that way.
    assert TriggerGraphBuilder.event_order(whole, ["L1", "H1"]) != order


def test_pairs_timed_on_tiles_say_so_once():
    """Degrading silently is the failure this warning exists to prevent: a
    pair timed on tile centres can carry a difference no signal can produce,
    and nothing downstream would show where it came from."""
    prepared = {"series": [None, None], "rates": [1024.0, 1024.0],
                "reconstruction_offset": {}}
    tile_dt = np.array([0.007])
    builder = TriggerGraphBuilder()
    with pytest.warns(RuntimeWarning, match="timed on their tile centres"):
        out = builder._timed_on_reconstruction(
            prepared, [0], [1], tile_dt, max_lag_s=0.05)
    assert out[0] == pytest.approx(0.007)
    # Said once for a run, not once per pair.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        builder._timed_on_reconstruction(
            prepared, [0], [1], tile_dt, max_lag_s=0.05)


def test_a_correction_that_cannot_be_measured_is_not_silently_zero():
    """The rate belongs to the event and not to its rendering. Taken from the
    wrong object it is NaN, every correction falls back to zero, and the stage
    reads as though it had timed the pairs when it had not."""
    fs = 1024.0
    wave = np.sin(2 * np.pi * 60.0 * np.arange(256) / fs)
    prepared = {
        "series": [(-0.020, wave), (0.004, wave)],
        "rates": [np.nan, np.nan],
        "reconstruction_offset": {},
    }
    out = TriggerGraphBuilder()._timed_on_reconstruction(
        prepared, [0], [1], np.array([0.020]), max_lag_s=0.05)
    # Without a rate nothing can be measured, so the tile difference stands.
    assert out[0] == pytest.approx(0.020)

    prepared["rates"] = [fs, fs]
    timed = TriggerGraphBuilder()._timed_on_reconstruction(
        prepared, [0], [1], np.array([0.020]), max_lag_s=0.05)
    assert timed[0] == pytest.approx(-0.004, abs=1.5 / fs)


def test_the_correlation_spans_the_events_and_not_the_gap_between_them():
    """A slide can put the two events of an accidental pair minutes apart.
    Correlating them on absolute time would lay out that whole gap, once per
    pair; each series is placed relative to its own event's instant instead,
    so the cost follows the events' own length."""
    fs = 1024.0
    wave = np.sin(2 * np.pi * 70.0 * np.arange(512) / fs)
    # The events are ten minutes apart, and each waveform is placed inside
    # its own event rather than on absolute time, so the gap never appears.
    far_apart = {
        "series": [(-0.25, wave), (-0.254, wave)],
        "rates": [fs, fs],
        "reconstruction_offset": {},
    }
    import time as _time
    began = _time.time()
    out = TriggerGraphBuilder()._timed_on_reconstruction(
        far_apart, [0], [1], np.array([-600.004]), max_lag_s=0.05)
    assert _time.time() - began < 1.0, "the gap was laid out"
    # The residual is the pair's own misalignment: the first series begins
    # four milliseconds later than the second relative to its own instant, so
    # it arrives that much after it.
    assert out[0] - (-600.004) == pytest.approx(+0.004, abs=1.5 / fs)


def test_a_pair_on_two_clocks_is_refused_and_not_answered():
    """The waveform is placed inside its own event; the event's instant is
    what a slide moves. Hand the estimator one series on absolute time and the
    other's instant displaced and the two sit a slide apart, which no lag
    inside the search can close. It has to keep the difference it came with
    and say so, rather than return the lag at the edge of the search --- and it
    must not lay the gap out to discover that."""
    import time as _time

    fs = 512.0
    wave = np.sin(2 * np.pi * 70.0 * np.arange(512) / fs)
    shift = 600.0
    mismatched = {
        "series": [(0.0, wave), (-shift, wave)],
        "rates": [fs, fs],
        "reconstruction_offset": {},
    }
    tile_dt = np.array([-shift])
    began = _time.time()
    with pytest.warns(RuntimeWarning, match="could not be timed"):
        out = TriggerGraphBuilder()._timed_on_reconstruction(
            mismatched, [0], [1], tile_dt, max_lag_s=0.05)
    assert _time.time() - began < 1.0, "the gap was laid out"
    assert out[0] == pytest.approx(-shift), "a junk correction was applied"


def test_detectors_at_different_rates_have_no_common_grid():
    """A lag is a number of samples of one grid. Two waveforms sampled
    differently do not share one, and correlating them as though they did
    would read the ratio of the rates as a delay."""
    fs = 1024.0
    wave = np.sin(2 * np.pi * 70.0 * np.arange(256) / fs)
    prepared = {"series": [(0.0, wave), (0.0, wave)],
                "rates": [fs, fs / 2.0], "reconstruction_offset": {}}
    with pytest.warns(RuntimeWarning, match="could not be timed"):
        out = TriggerGraphBuilder()._timed_on_reconstruction(
            prepared, [0], [1], np.array([0.003]), max_lag_s=0.05)
    assert out[0] == pytest.approx(0.003)
