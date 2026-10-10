"""Waveform completion: the mask from tiles (bridged, dilated, with a ridge),
the inversion inside it in each basis, the pursuit, the settings, the strain
and the offline whitening of a stretch."""
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from wdf.analysis.reconstruction import waveform_overlap
from wdf.analysis.waveform_completion import (CompletionMask, CompletionSettings, build_mask, complete, energy_ridge,
                                              hole_fraction, masked_pursuit, place, whitened_stretch)

FS, T0 = 2048.0, 1.0e9
CHIRP_AT, CHIRP_S, SNR = 30.0, 3.0, 40.0


def chirp(n, f0=40.0, f1=200.0):
    t = np.arange(n) / FS - CHIRP_AT
    inside = (t >= 0) & (t < CHIRP_S)
    f = f0 + (f1 - f0) * np.clip(t, 0, CHIRP_S) / CHIRP_S
    c = np.where(inside, np.sin(2 * np.pi * np.cumsum(f) / FS)
                 * np.sin(np.pi * np.clip(t, 0, CHIRP_S) / CHIRP_S) ** 2, 0.0)
    return c * SNR / np.sqrt(np.sum(c ** 2)), t, f, inside


@pytest.fixture(scope="module")
def planted():
    n = int(64 * FS)
    c, t, f, inside = chirp(n)
    x = np.random.default_rng(5).standard_normal(n) + c
    return x, c, (T0 + CHIRP_AT + t[inside], f[inside])


#: Tiles along the chirp with holes between them: four pieces of it.
TILES = (T0 + CHIRP_AT + np.array([0.0, 0.5, 1.5, 2.6]),
         T0 + CHIRP_AT + np.array([0.3, 0.8, 1.9, 3.0]),
         np.array([35.0, 55.0, 110.0, 170.0]), np.array([56.0, 88.0, 176.0, 272.0]),
         np.array([30.0, 40.0, 60.0, 90.0]))


def settings(**kw):
    par = SimpleNamespace(MinFrequency=16.0, **kw)
    return CompletionSettings.from_parameters(par, FS), par


def packets_or_skip():
    tsa = pytest.importorskip("py4tsa.tsa")
    if not hasattr(tsa, "CosinePackets"):
        pytest.skip("the installed py4tsa has no cosine packets")


# ---------------------------------------------------------------- settings
def test_the_settings_are_parameters_and_are_recorded():
    s, par = settings(CompletionBasis="LocalCos64", CompletionThreshold=0.0, CompletionQs=[2, 4])
    assert (s.basis, s.threshold, s.qs) == ("LocalCos64", 0.0, (2.0, 4.0))
    assert s.low_frequency == 16.0 and s.slot_s == 64.0 / FS
    assert par.CompletionMethod == "mask" and par.CompletionDilationOctaves == 0.5
    assert par.CompletionQs == [2.0, 4.0] and par.CompletionLowFrequency == 16.0
    assert settings()[0].basis == "CosinePackets32"
    from wdf.analysis.waveform_completion import DEFAULTS
    d = CompletionSettings.defaults(FS, basis="LocalCos64")
    assert d.basis == "LocalCos64" and d.threshold == DEFAULTS["CompletionThreshold"]
    assert d.slot_s == DEFAULTS["CompletionSlotSamples"] / FS and d.low_frequency == 0.0
    other = CompletionSettings.from_parameters(SimpleNamespace(), FS, defaults=dict(DEFAULTS, CompletionPad=2.0))
    assert other.pad == 2.0
    assert set(DEFAULTS) == set(CompletionSettings.KEYS.values())


# without py4tsa a wavelet name is checked when the transform runs, not here
_NEEDS_CORE = pytest.mark.skipif(importlib.util.find_spec("py4tsa") is None,
                                 reason="wavelet names are checked against py4tsa")


@pytest.mark.parametrize("bad", [dict(CompletionMethod="fit"),
                                 pytest.param(dict(CompletionBasis="Fourier"), marks=_NEEDS_CORE),
                                 dict(CompletionBasis="CosinePackets30"), dict(CompletionPacketCost="l0"),
                                 dict(CompletionTileBand="ridge"), dict(CompletionSlot=0.0),
                                 dict(CompletionDilationTime=-1.0), dict(CompletionMaxAtoms=0),
                                 dict(CompletionLowFrequency=2000.0)])
def test_settings_that_cannot_run_are_refused(bad):
    with pytest.raises(ValueError):
        settings(**bad)


# ---------------------------------------------------------------- the mask
def test_the_mask_marks_boxes_and_reads_them_back():
    m = CompletionMask.empty(T0, T0 + 2.0, 16.0, 1024.0, 0.125, 0.25)
    m.add_boxes([T0 + 0.5], [T0 + 0.75], [64.0], [128.0])
    assert m.cells.sum() == 2 * 4           # two slots, one octave of quarter-octave cells
    assert m.contains([T0 + 0.6], [90.0])[0] and not m.contains([T0 + 0.6], [300.0])[0]
    assert not m.contains([T0 + 1.5], [90.0])[0]
    boxes = m.boxes()
    assert len(boxes) == 2 and boxes[0][2] == pytest.approx(64.0) and boxes[0][3] == pytest.approx(128.0)
    keep = m.keeps([T0 + 0.7, T0 + 1.0], [T0 + 0.8, T0 + 1.1], [100.0, 100.0], [110.0, 110.0])
    assert keep.tolist() == [True, False]
    assert m.area == pytest.approx(0.25 * 1.0)


def test_bridging_fills_a_hole_between_two_tiles_in_log_frequency():
    m = CompletionMask.empty(T0, T0 + 4.0, 16.0, 1024.0, 0.25, 0.25)
    m.add_boxes([T0 + 0.0, T0 + 3.0], [T0 + 0.25, T0 + 3.25], [32.0, 128.0], [45.0, 181.0])
    holes = int((~m.cells.any(axis=1)).sum())
    CompletionMask.bridge(m, 1.0)            # the hole is 2.75 s: longer than 1 s, nothing bridged
    assert int((~m.cells.any(axis=1)).sum()) == holes
    m.bridge(4.0)
    assert m.cells.any(axis=1).sum() == 13    # every slot from the first tile to the last
    assert m.contains([T0 + 1.6], [64.0])[0]  # midway: the geometric mean of the two bands
    assert not m.contains([T0 + 1.6], [32.0])[0] and not m.contains([T0 + 1.6], [150.0])[0]


def test_dilation_grows_the_mask_on_both_sides():
    m = CompletionMask.empty(T0, T0 + 2.0, 16.0, 1024.0, 0.125, 0.25)
    m.add_boxes([T0 + 1.0], [T0 + 1.125], [64.0], [128.0])
    m.dilate(0.25, 0.5)
    assert m.cells.any(axis=1).sum() == 5 and m.cells.any(axis=0).sum() == 8
    assert m.contains([T0 + 0.8, T0 + 1.3], [48.0, 160.0]).all()


def test_the_ridge_band_and_the_coverage():
    m = CompletionMask.empty(T0, T0 + 2.0, 16.0, 1024.0, 0.125, 0.125)
    times = T0 + np.linspace(0.2, 1.8, 20)
    freqs = np.geomspace(40.0, 400.0, 20)
    assert m.coverage(times, freqs) == 0.0
    m.add_ridge(times, freqs, 0.25)
    assert m.coverage(times, freqs) == 1.0
    assert m.coverage(times, freqs * 2.0 ** 0.5) < 0.5
    assert np.isnan(m.coverage([], []))


def test_build_mask_reads_the_tiles_and_pads_the_extent():
    s, _ = settings()
    m = build_mask(TILES, s, FS)
    assert m.t0 == pytest.approx(TILES[0][0] - s.pad) and m.t_end >= TILES[1][-1] + s.pad
    assert m.f_low == pytest.approx(16.0) and m.f_high == pytest.approx(0.5 * FS)
    assert m.cells.any(axis=1)[m.slot_range(TILES[1][0], TILES[0][1])[0]:m.slot_range(TILES[0][-1], TILES[1][-1])[1]].all()
    centroid = build_mask(TILES, settings(CompletionTileBand="centroid", CompletionBand=0.25)[0], FS)
    assert centroid.area < m.area


def test_the_energy_ridge_follows_a_known_chirp(planted):
    x, c, ridge = planted
    times, freqs = energy_ridge(c, T0, FS, m=64, f_low=16.0, floor=0.05)
    assert times.size and times.min() >= T0 + CHIRP_AT and times.max() <= T0 + CHIRP_AT + CHIRP_S
    expect = np.interp(times, ridge[0], ridge[1])
    assert np.median(np.abs(np.log2(freqs / expect))) < 0.25
    assert energy_ridge(np.zeros(10), T0, FS, 64, 16.0, 0.05)[0].size == 0


def test_build_mask_needs_tiles_or_a_track():
    s, _ = settings()
    with pytest.raises(ValueError):
        build_mask((np.zeros(0),) * 5, s, FS)
    m = build_mask((np.zeros(0),) * 5, s, FS, track=(T0 + np.array([1.0, 2.0]), np.array([50.0, 100.0])))
    assert m.contains([T0 + 1.5], [70.0])[0]


# ---------------------------------------------------------------- completion
def _overlap(result, c):
    i0 = int(round((result.t0 - T0) * FS))
    return waveform_overlap(result.waveform, c[i0:i0 + result.waveform.size])["overlap"]


@pytest.mark.parametrize("basis", ["CosinePackets32", "LocalCos64", "DaubC8"])
def test_the_completion_fills_the_holes_in_every_basis(planted, basis):
    if basis.startswith("CosinePackets"):
        packets_or_skip()
    else:
        pytest.importorskip("py4tsa")
    x, c, ridge = planted
    s, _ = settings(CompletionBasis=basis)
    full = complete((T0, x), FS, TILES, s, reference=ridge)
    assert full.waveform.size == int(np.ceil((full.mask.t_end - full.mask.t0) * FS))
    assert np.isfinite(full.waveform).all() and full.n_kept > 0 and full.n_total >= full.waveform.size
    assert full.coverage == 1.0
    assert _overlap(full, c) > (0.8 if basis != "DaubC8" else 0.6)   # octave-wide tiles keep more noise
    bare = complete((T0, x), FS, TILES, settings(CompletionBasis=basis, CompletionBridgeGap=0.0,
                                                 CompletionDilationTime=0.0)[0], reference=ridge)
    assert bare.coverage < 0.8
    assert hole_fraction(bare.waveform, place(T0, c, bare.t0, bare.waveform.size, FS)) > 0.1
    assert hole_fraction(full.waveform, place(T0, c, full.t0, full.waveform.size, FS)) < 0.05


def test_the_threshold_removes_noise_and_the_sparse_overlap_is_reported(planted):
    pytest.importorskip("py4tsa")
    x, c, ridge = planted
    with_threshold = complete((T0, x), FS, TILES, settings(CompletionBasis="LocalCos64", CompletionThreshold=1.0)[0])
    without = complete((T0, x), FS, TILES, settings(CompletionBasis="LocalCos64", CompletionThreshold=0.0)[0])
    assert _overlap(with_threshold, c) > _overlap(without, c)
    assert with_threshold.energy < without.energy
    sparse = (T0 + CHIRP_AT, c[int(CHIRP_AT * FS):int((CHIRP_AT + CHIRP_S) * FS)])
    r = complete((T0, x), FS, TILES, settings(CompletionBasis="LocalCos64")[0], sparse=sparse)
    assert 0.5 < r.sparse_overlap <= 1.0
    assert np.isnan(r.coverage)
    rec = r.record()
    assert rec["basis"] == "LocalCos64" and rec["method"] == "mask" and rec["n_samples"] == r.waveform.size


def test_the_pursuit_fits_atoms_inside_the_mask(planted):
    x, c, ridge = planted
    s, _ = settings(CompletionMethod="mp", CompletionMaxAtoms=40, CompletionFrequencyCount=20)
    r = complete((T0, x), FS, TILES, s, reference=ridge)
    assert r.method == "mp" and 0 < len(r.atoms) <= 40 and r.n_kept == len(r.atoms)
    assert _overlap(r, c) > 0.6
    for atom in r.atoms:
        assert r.mask.contains([atom.centre], [atom.frequency])[0]
        assert atom.snr >= s.stop_snr
    model, atoms, sigma = masked_pursuit(np.zeros(4096), T0, FS, r.mask, s)
    assert not atoms and not model.any() and sigma > 0


def test_a_stream_short_of_the_event_gives_zeros_not_nans(planted):
    pytest.importorskip("py4tsa")
    x, c, ridge = planted
    short = x[:int((CHIRP_AT + 1.0) * FS)]
    short[-10:] = np.nan
    r = complete((T0, short), FS, TILES, settings(CompletionBasis="LocalCos64")[0])
    assert np.isfinite(r.waveform).all()
    assert not r.waveform[int(1.6 * FS):].any()


# ---------------------------------------------------------------- strain and the stream
def test_the_strain_undoes_the_taps_and_the_stretch_is_whitened_as_the_worker_does():
    from scipy.signal import fftconvolve

    from wdf.analysis.hrss import ModelHistory
    pytest.importorskip("py4tsa")
    from wdf.processes.zero_phase_whitening import truncated_taps

    ar = np.array([1e-22, 1.2, -0.5])
    taps = truncated_taps(ar, 4.0, FS)
    n = 8192
    t = (np.arange(n) - n / 2) / FS
    h = 1e-21 * np.exp(-(t / 0.02) ** 2) * np.sin(2 * np.pi * 150.0 * t)
    white = fftconvolve(h, taps, mode="same")
    r = complete((T0, white), FS, (T0 + np.array([n / 2 / FS - 0.05]), T0 + np.array([n / 2 / FS + 0.05]),
                                   np.array([100.0]), np.array([220.0]), np.array([100.0])),
                 settings(CompletionBasis="LocalCos64", CompletionThreshold=0.0, CompletionDilationOctaves=0.0,
                          CompletionDilationTime=0.0)[0])
    back = r.strain(taps, (30.0, 900.0))
    assert back.size == r.waveform.size
    i0 = int(round((r.t0 - T0) * FS))
    assert waveform_overlap(back, h[i0:i0 + back.size])["overlap"] > 0.95

    history = ModelHistory([dict(switch_gps=T0, fit_start=T0 - 300, fit_end=T0, sigma=2.0, ar=ar),
                            dict(switch_gps=T0 + 2.0, fit_start=T0, fit_end=T0 + 2, sigma=4.0, ar=ar * 2)],
                           dict(WhiteningMode="zerophase", resampling=FS, ZeroPhaseDuration=4.0))
    k = taps.size // 2
    raw = np.random.default_rng(2).standard_normal(int(5 * FS))
    first, y, rate = whitened_stretch(raw, FS, T0 - 1.0, history)
    assert rate == FS and first == pytest.approx(T0 - 1.0 + k / FS) and y.size == raw.size - 2 * k
    expect0 = fftconvolve(raw, history.taps(0), mode="valid")
    expect1 = fftconvolve(raw, history.taps(1), mode="valid")
    cut = int(np.ceil((T0 + 2.0 - first) * FS - 1e-9))
    assert np.allclose(y[:cut], expect0[:cut]) and np.allclose(y[cut:], expect1[cut:])
    _, z, _ = whitened_stretch(raw, FS, T0 - 1.0, history, normalise=True)
    assert np.allclose(z[:cut], y[:cut] / 2.0) and np.allclose(z[cut:], y[cut:] / 4.0)
    with pytest.raises(ValueError):
        whitened_stretch(raw, 4096.0, T0, history)


# ---------------------------------------------------------------- the gate
def test_the_tile_coverage_reads_the_tiles_alone():
    from wdf.analysis.waveform_completion import tile_coverage

    s, _ = settings(CompletionSlot=0.25, CompletionOctaveStep=0.25)
    full = (T0 + np.array([0.0, 1.0]), T0 + np.array([1.0, 2.0]), np.array([40.0, 40.0]), np.array([80.0, 80.0]))
    assert tile_coverage(full, s, FS) == 1.0
    stairs = (T0 + np.array([0.0, 1.0]), T0 + np.array([1.0, 2.0]), np.array([40.0, 80.0]), np.array([80.0, 160.0]))
    assert tile_coverage(stairs, s, FS) == 1.0          # each time cell full over its own band
    split = (T0 + np.array([0.0, 0.0]), T0 + np.array([2.0, 2.0]), np.array([40.0, 160.0]), np.array([80.0, 320.0]))
    assert 0.6 < tile_coverage(split, s, FS) < 0.8      # an octave missing between two bands of an octave
    holed = (T0 + np.array([0.0, 1.75]), T0 + np.array([0.25, 2.0]), np.array([40.0, 80.0]), np.array([80.0, 160.0]))
    assert 0.0 < tile_coverage(holed, s, FS) < 0.5
    reverse = (holed[0], holed[1], holed[2][::-1], holed[3][::-1])
    assert tile_coverage(reverse, s, FS) == tile_coverage(holed, s, FS)    # symmetric in sweep direction
    nobridge = settings(CompletionSlot=0.25, CompletionOctaveStep=0.25, CompletionBridgeGap=0.0)[0]
    assert tile_coverage(holed, nobridge, FS) == 1.0     # nothing between the tiles to fill
    assert np.isnan(tile_coverage((np.zeros(0),) * 4, s, FS))


def test_the_gate_returns_the_sparse_reconstruction_above_the_coverage(planted):
    pytest.importorskip("py4tsa")
    from wdf.analysis.waveform_completion import tile_coverage

    x, c, ridge = planted
    sparse = (T0 + CHIRP_AT, c[int(CHIRP_AT * FS):int((CHIRP_AT + CHIRP_S) * FS)].copy())
    base = settings(CompletionBasis="LocalCos64")[0]
    covered = tile_coverage(TILES, base, FS)
    assert 0.0 < covered < 1.0
    above = complete((T0, x), FS, TILES, settings(CompletionBasis="LocalCos64",
                                                  CompletionMaxTileCoverage=covered - 0.01)[0], sparse=sparse)
    assert not above.completed and above.method == "sparse" and above.n_kept == 0
    assert above.t0 == sparse[0] and np.array_equal(above.waveform, sparse[1])
    assert above.tile_coverage == covered and above.record()["completed"] is False
    at = complete((T0, x), FS, TILES, settings(CompletionBasis="LocalCos64",
                                               CompletionMaxTileCoverage=covered)[0], sparse=sparse)
    assert at.completed and at.method == "mask" and at.tile_coverage == covered
    without_sparse = complete((T0, x), FS, TILES, settings(CompletionBasis="LocalCos64",
                                                           CompletionMaxTileCoverage=0.0)[0])
    assert settings(CompletionBasis="LocalCos64", CompletionMaxTileCoverage=1.0)[0].max_tile_coverage == 1.0
    assert without_sparse.completed           # nothing to fall back on: the completion runs
    from wdf.analysis.waveform_completion import DEFAULTS
    assert base.max_tile_coverage == DEFAULTS["CompletionMaxTileCoverage"]
    with pytest.raises(ValueError):
        settings(CompletionMaxTileCoverage=1.5)
