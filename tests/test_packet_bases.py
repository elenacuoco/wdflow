"""Triggers whose windows won a wavelet-packet basis, beside pyramid ones.

A trigger records the basis it won as `wave`, and its coefficient indices are
placed in the plane by that basis's layout. At 2048 Hz and 1024 samples,
index 256 is the first coefficient of the octave [256, 512) Hz for the
pyramid, 4 samples long, and the first coefficient of the band [256, 272) Hz
for Coif1P6, 64 samples long: the same index, two different tiles, one octave.
"""
import numpy as np
import pandas as pd
import pytest

from wdf.analysis.detector_graph import (band_grid, build_detector_graph, detector_events,
                                         event_coefficients, trigger_wavegrams)
from wdf.analysis.scale import ScaleCalibration, pixel_cloud, scale_maximum, unique_tiles
from wdf.analysis.wavegram_match import render

FS = 2048.0
N = 1024
PACKET = "Coif1P6"
PYRAMID = "DaubC12"


def mixed(waves, index=(256,), values=(3.0,), gps0=1000.0, sigma=1.0):
    """One trigger per basis named, each window after the last, all holding
    the same coefficients."""
    rows = []
    for k, wave in enumerate(waves):
        gps = gps0 + k * 0.75 * N / FS
        rows.append(dict(
            gps=gps, gpsStart=gps, gpsCentroid=gps, tSpread=0.01, gpsPeak=gps,
            duration=N / FS, EnWDF=float(np.linalg.norm(values)) / sigma, sigma=sigma,
            snrPeak=float(np.max(np.abs(values))) / sigma,
            freqMin=256.0, freqMean=300.0, freqMax=400.0,
            wave=wave, n_coeff=N, fs=FS, stride=0.75 * N / FS,
            wt_index=np.asarray(index, dtype=np.uint16),
            wt_value=np.asarray(values, dtype=np.float32), ifo="H1"))
    return pd.DataFrame(rows)


def octave_row(bands, lo=256.0, hi=512.0):
    return int(np.flatnonzero((bands[:, 0] == lo) & (bands[:, 1] == hi))[0])


def test_the_wave_column_carries_a_packet_name():
    triggers = mixed([PYRAMID, PACKET])
    assert triggers.wave.tolist() == [PYRAMID, PACKET]


def test_a_packet_tile_lands_on_the_octave_that_holds_it_in_the_trigger_wavegram():
    bands = band_grid([N], FS)
    grids = trigger_wavegrams(mixed([PYRAMID, PACKET]), bands, time_bins=8)
    row = octave_row(bands)
    for grid in grids.reshape(2, len(bands), 8):
        assert grid[row, 0] == pytest.approx(3.0)
        assert grid.sum() == pytest.approx(3.0)


def test_the_cloud_places_each_trigger_by_the_basis_it_won():
    cloud = pixel_cloud(mixed([PYRAMID, PACKET], gps0=1000.0).assign(gps=1000.0))
    pyramid, packet = cloud.iloc[0], cloud.iloc[1]
    assert (pyramid.f_lo, pyramid.f_hi) == (256.0, 512.0)
    assert pyramid.t_hi - pyramid.t_lo == pytest.approx(4 / FS)
    assert (packet.f_lo, packet.f_hi) == (256.0, 272.0)
    assert packet.t_hi - packet.t_lo == pytest.approx(64 / FS)


def test_a_packet_tile_and_an_octave_tile_that_share_a_corner_are_two_tiles():
    cloud = pixel_cloud(mixed([PYRAMID, PACKET]).assign(gps=1000.0))
    assert cloud.t_lo.nunique() == 1 and cloud.f_lo.nunique() == 1
    assert len(unique_tiles(cloud)) == 2


def test_the_calibration_keeps_a_packet_band_apart_from_its_octave():
    rng = np.random.default_rng(0)
    frames = [mixed([PYRAMID, PACKET], values=(v,), gps0=1000.0 + 10 * i)
              for i, v in enumerate(rng.uniform(1.0, 4.0, 50))]
    calibration = ScaleCalibration.fit(pixel_cloud(pd.concat(frames, ignore_index=True)))
    assert {(N, 256.0, 512.0), (N, 256.0, 272.0)} <= set(calibration.tables)


def test_the_cross_scale_maximum_reads_each_tile_in_its_own_band():
    """An octave tile and a packet tile of one window with one start: each is
    read at its own band, not at whichever of the two sorted last."""
    background = pixel_cloud(pd.concat(
        [mixed([PYRAMID], values=(1.0,)).assign(gps=1000.0 - 10 * i) for i in range(20)]
        + [mixed([PACKET], values=(8.0,)).assign(gps=1000.0 - 10 * i) for i in range(20)],
        ignore_index=True))
    calibration = ScaleCalibration.fit(background)
    pixels = pixel_cloud(pd.concat([mixed([PYRAMID], values=(4.0,)),
                                    mixed([PACKET], values=(4.0,))],
                                   ignore_index=True).assign(gps=1000.0))
    out = scale_maximum(pixels, calibration)
    assert out.significance[0] > out.significance[1]
    assert np.allclose(out.s_max, out.significance)
    assert (out.n_scales == 1).all()


def test_an_event_of_packet_triggers_is_assembled_and_described():
    triggers = mixed([PACKET, PACKET, PYRAMID], index=(256, 257), values=(3.0, 2.0))
    graph = build_detector_graph(triggers)
    events = detector_events(graph)
    assert len(events) >= 1
    assert np.isfinite(events.EnWDF).all()

    maps = event_coefficients(graph)
    tiles = np.concatenate([m.tiles[3] - m.tiles[2] for m in maps.values()])
    assert set(np.round(tiles, 9)) == {16.0, 256.0}
    row = octave_row(next(iter(maps.values())).bands)
    for rendered in maps.values():
        assert rendered.grid.sum() > 0.0
        assert rendered.grid[row].sum() == pytest.approx(rendered.grid.sum())


def test_the_packet_tiles_of_an_event_render_on_the_shared_ladder():
    graph = build_detector_graph(mixed([PACKET, PYRAMID]))
    rendered = event_coefficients(graph, labels=np.zeros(2, dtype=int))[0]
    t_lo, t_hi = rendered.tiles[0], rendered.tiles[1]
    grid = render(rendered.tiles, rendered.bands, t_lo.min(), t_hi.max(), 1.0 / FS)
    row = octave_row(rendered.bands)
    assert np.flatnonzero(np.abs(grid).sum(axis=1)).tolist() == [row]


def test_a_tile_in_no_row_of_the_ladder_is_still_an_error():
    cloud = tuple(np.array([v]) for v in (0.0, 0.1, 4096.0, 4112.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="lies in no row"):
        render(cloud, band_grid([N], FS), 0.0, 1.0, 0.1)


def test_the_pixel_map_of_an_event_keeps_its_packet_tiles():
    from wdf.analysis.pixel_graph import build_pixel_graph, cluster_wavegrams

    pixels = pixel_cloud(mixed([PYRAMID, PACKET]).assign(gps=1000.0))
    graph = build_pixel_graph(pixels)
    labels = np.zeros(len(graph.nodes), dtype=int)
    rendered = cluster_wavegrams(graph, labels, time_bins=64)[0]
    row = octave_row(rendered.bands)
    assert rendered.grid.sum() == pytest.approx(6.0)
    assert rendered.grid[row].sum() == pytest.approx(6.0)
