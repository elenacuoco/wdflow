"""An event made of tiles: its labels, its path, its map and its waveform.

The detector stage groups tiles, not windows, so everything an event is asked
for has to be read off the tiles it owns and off no others: the regions two
windows both kept count once in its energy and twice in its stitching, a
window's other tiles belong elsewhere, and the path the event's ridge traces
decides which of its tiles are summed together, whichever way in time it
runs.
"""
import numpy as np
import pandas as pd
import pytest

from wdf.analysis.pixel_graph import (PixelGraphConfig, build_pixel_graph,
                                      cluster_events, cluster_wavegrams,
                                      follow_ridges, tile_labels)

STRIDE = (512 - 32) / 2048.0


def path_cloud(reverse=False):
    """A path over three windows, octave by octave, beside an off-path tile.

    The path climbs 64 -> 128 -> 256 Hz (or falls, with `reverse`), one
    window apart; the fourth tile sits with the middle window at 512 Hz, an
    empty octave above the path, and joins the cluster through the allowance.
    Tiles last one over the upper edge of their band, as the dyadic tiling
    makes them.
    """
    bands = [(64.0, 128.0), (128.0, 256.0), (256.0, 512.0)]
    if reverse:
        bands = bands[::-1]
    starts = [0.0, STRIDE, 2 * STRIDE, STRIDE + 0.004]
    bands = bands + [(512.0, 1024.0)]
    return pd.DataFrame(dict(
        trigger_index=[0, 1, 2, 1], stride=STRIDE, ifo="H1", scale=512.0,
        fs=2048.0, t_lo=starts,
        t_hi=[start + 1.0 / high for start, (_, high) in zip(starts, bands)],
        f_lo=[low for low, _ in bands], f_hi=[high for _, high in bands],
        energy=[9.0, 16.0, 25.0, 4.0], sigma=1.0,
        coefficient=[3, 9, 20, 400], value=[3.0, -4.0, 5.0, 2.0]))


ADMISSION = PixelGraphConfig(stride_tolerance=2.0, band_tolerance=2)


def test_the_ridge_keeps_the_path_and_regroups_what_it_leaves():
    graph = build_pixel_graph(path_cloud(), config=ADMISSION)
    assert len(np.unique(graph.components())) == 1
    labels = follow_ridges(graph)
    events = cluster_events(graph, labels=labels)
    assert len(events) == 2
    path = events.loc[events.n_pixels.idxmax()]
    assert path.n_pixels == 3
    assert path.EnWDF == pytest.approx(np.sqrt(9.0 + 16.0 + 25.0))
    # Nothing is dropped: the energy of the two events is the cluster's.
    assert (events.energy.sum()
            == pytest.approx(9.0 + 16.0 + 25.0 + 4.0))


def test_a_falling_path_is_followed_as_a_rising_one():
    rising = cluster_events(*_graph_and_path(path_cloud()))
    falling = cluster_events(*_graph_and_path(path_cloud(reverse=True)))
    assert sorted(rising.n_pixels) == sorted(falling.n_pixels)
    assert sorted(rising.EnWDF.round(9)) == sorted(falling.EnWDF.round(9))


def _graph_and_path(cloud):
    graph = build_pixel_graph(cloud, config=ADMISSION)
    return graph, follow_ridges(graph)


def test_a_cluster_of_one_tile_is_its_own_path():
    cloud = path_cloud().iloc[[0]]
    graph = build_pixel_graph(cloud, config=ADMISSION)
    assert follow_ridges(graph).tolist() == [0]


def test_a_repeated_region_takes_the_label_of_the_node_describing_it():
    cloud = path_cloud()
    twice = pd.concat([cloud, cloud.iloc[[1]].assign(trigger_index=7,
                                                     value=-3.5,
                                                     energy=12.25)],
                      ignore_index=True)
    graph = build_pixel_graph(twice, config=ADMISSION)
    labels = graph.components()
    tiles = tile_labels(twice, graph, labels)
    assert len(tiles) == len(twice)
    # The repeat and the tile it repeats are one region, so one event.
    assert tiles[1] == tiles[-1]


def test_a_tile_no_node_describes_belongs_to_no_event():
    cloud = path_cloud()
    graph = build_pixel_graph(cloud, significance=np.log1p(cloud.energy),
                              config=PixelGraphConfig(
                                  minimum_significance=np.log1p(5.0)))
    tiles = tile_labels(cloud, graph)
    assert (tiles[cloud.energy.to_numpy() < 5.0] == -1).all()
    assert (tiles[cloud.energy.to_numpy() >= 5.0] >= 0).all()


def test_the_map_carries_signed_amplitudes_the_network_can_render():
    from wdf.analysis.wavegram_match import render

    graph = build_pixel_graph(path_cloud(), config=ADMISSION)
    maps = cluster_wavegrams(graph, time_bins=16)
    tiles = maps[0].tiles
    assert len(tiles) == 6
    ordered = np.argsort(tiles[0])
    assert tiles[5][ordered][:3] == pytest.approx([3.0, -4.0, 2.0])
    grid = render(tiles, maps[0].bands, -0.1, 1.0, 0.01)
    assert np.isfinite(grid).all()


def test_a_cloud_without_signs_carries_no_amplitude_rather_than_a_magnitude():
    cloud = path_cloud().drop(columns=["value"])
    graph = build_pixel_graph(cloud, config=ADMISSION)
    tiles = cluster_wavegrams(graph, time_bins=16)[0].tiles
    assert np.isnan(tiles[5]).all()


def test_an_event_is_inverted_from_its_own_tiles_and_no_others():
    pytest.importorskip("py4tsa")
    from _synth import triggers_from_signal
    from wdf.analysis.cluster_coefficients import (
        ClusterCoefficients, iter_tile_cluster_coefficients)
    from wdf.analysis.scale import pixel_cloud

    rng = np.random.default_rng(4)
    fs, window, overlap = 2048.0, 512, 32
    triggers = triggers_from_signal(rng.normal(size=4 * window), fs, window,
                                    overlap)
    cloud = pixel_cloud(triggers)
    # Two events in one window: the first half of its coefficients, and the
    # rest of the stretch.
    first = (cloud.trigger_index == 0) & (cloud.coefficient < window // 2)
    labels = np.where(first, 0, 1)
    found = dict(iter_tile_cluster_coefficients(cloud, labels, triggers,
                                                window, overlap))
    assert set(found) == {0, 1}
    kept = found[0].coefficients
    assert kept.shape == (1, window)
    assert np.count_nonzero(kept[0, window // 2:]) == 0
    # One window, orthonormal basis: the waveform's norm is the tiles' norm.
    energy = float((cloud.value[first] ** 2).sum())
    assert found[0].enwdf() == pytest.approx(np.sqrt(energy), rel=1e-5)
    # All of a window's tiles give back what the trigger-level cluster gives.
    whole = ClusterCoefficients.from_triggers(triggers, fs, window, overlap)
    every = dict(iter_tile_cluster_coefficients(
        cloud, np.zeros(len(cloud), dtype=int), triggers, window, overlap))[0]
    assert every.enwdf() == pytest.approx(whole.enwdf(), rel=1e-5)


def test_a_cloud_of_two_window_lengths_is_refused():
    from wdf.analysis.cluster_coefficients import iter_tile_cluster_coefficients

    cloud = path_cloud().assign(scale=[512.0, 512.0, 1024.0, 512.0])
    with pytest.raises(ValueError, match="window lengths"):
        list(iter_tile_cluster_coefficients(
            cloud, np.zeros(4, dtype=int),
            pd.DataFrame(dict(gps=[0.0] * 8, wave="x", sigma=1.0,
                              n_coeff=512, fs=2048.0)), 512, 32))
