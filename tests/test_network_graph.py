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
    # The same waveform in both, the second starting 4 ms later. That is the
    # arrival-time difference; the tile centres, below, disagree with it.
    prepared = {
        "series": [(0.0, wave), (0.004, wave)],
        "rates": [fs, fs],
        "prepared_gps": np.array([0.020, 0.000]),
        "reconstruction_offset": {},
    }
    tile_dt = np.array([0.020])
    corrected = TriggerGraphBuilder._timed_on_reconstruction(
        prepared, [0], [1], tile_dt, max_lag_s=0.05)

    assert corrected[0] == pytest.approx(-0.004, abs=1.5 / fs)
    # The correction is a property of the pair, so it is measured once.
    assert (0, 1) in prepared["reconstruction_offset"]
    again = TriggerGraphBuilder._timed_on_reconstruction(
        prepared, [0], [1], tile_dt + 3.0, max_lag_s=0.05)
    assert again[0] == pytest.approx(corrected[0] + 3.0, abs=1e-9)


def test_a_pair_without_a_reconstruction_keeps_the_time_it_came_with():
    prepared = {"series": [None, None], "rates": [1024.0, 1024.0],
                "prepared_gps": np.zeros(2), "reconstruction_offset": {}}
    tile_dt = np.array([0.007])
    out = TriggerGraphBuilder._timed_on_reconstruction(
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
                "prepared_gps": np.zeros(2), "reconstruction_offset": {}}
    tile_dt = np.array([0.007])
    with pytest.warns(RuntimeWarning, match="timed on their tile centres"):
        out = TriggerGraphBuilder._timed_on_reconstruction(
            prepared, [0], [1], tile_dt, max_lag_s=0.05)
    assert out[0] == pytest.approx(0.007)
    # Said once for a run, not once per pair.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        TriggerGraphBuilder._timed_on_reconstruction(
            prepared, [0], [1], tile_dt, max_lag_s=0.05)


def test_a_correction_that_cannot_be_measured_is_not_silently_zero():
    """The rate belongs to the event and not to its rendering. Taken from the
    wrong object it is NaN, every correction falls back to zero, and the stage
    reads as though it had timed the pairs when it had not."""
    fs = 1024.0
    wave = np.sin(2 * np.pi * 60.0 * np.arange(256) / fs)
    prepared = {
        "series": [(0.0, wave), (0.004, wave)],
        "rates": [np.nan, np.nan],
        "prepared_gps": np.array([0.020, 0.000]),
        "reconstruction_offset": {},
    }
    out = TriggerGraphBuilder._timed_on_reconstruction(
        prepared, [0], [1], np.array([0.020]), max_lag_s=0.05)
    # Without a rate nothing can be measured, so the tile difference stands.
    assert out[0] == pytest.approx(0.020)

    prepared["rates"] = [fs, fs]
    prepared["reconstruction_offset"] = {}
    timed = TriggerGraphBuilder._timed_on_reconstruction(
        prepared, [0], [1], np.array([0.020]), max_lag_s=0.05)
    assert timed[0] == pytest.approx(-0.004, abs=1.5 / fs)
