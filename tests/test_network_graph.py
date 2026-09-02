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
