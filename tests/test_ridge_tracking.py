"""The ridge: cells, background, the local alignment offline and causal, and
the energy along the path, on white noise with a planted chirp."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wdf.analysis.ridge_tracking import (CausalBackground, RidgeSettings, StreamingRidge,
                                         background_z, box_scores, calibrate_drift, cell_sums,
                                         combine_min, offline_ridge, ridge_waveform, standardise,
                                         stream_cells)

FS, T0, SECONDS = 2048.0, 1.0e9, 128
CHIRP_AT, CHIRP_S, SNR = 80.0, 3.0, 15.0


def settings(**kw):
    par = SimpleNamespace(window=1024, overlap=256, MinFrequency=16.0, **kw)
    return RidgeSettings.from_parameters(par, FS), par


def chirp(n):
    t = np.arange(n) / FS - CHIRP_AT
    inside = (t >= 0) & (t < CHIRP_S)
    f = 40.0 + 160.0 * np.clip(t, 0, CHIRP_S) / CHIRP_S
    c = np.where(inside, np.sin(2 * np.pi * np.cumsum(f) / FS)
                 * np.sin(np.pi * np.clip(t, 0, CHIRP_S) / CHIRP_S) ** 2, 0.0)
    return c * SNR / np.sqrt(np.sum(c ** 2))


@pytest.fixture(scope="module")
def stream():
    n = int(SECONDS * FS)
    noise = np.random.default_rng(0).standard_normal(n)
    return noise, chirp(n)


@pytest.fixture(scope="module")
def maps(stream):
    noise, c = stream
    s, _ = settings()
    return s, stream_cells(noise + c, T0, FS, s)


def test_the_settings_are_parameters_and_are_recorded():
    s, par = settings(RidgeLambda=0.4, RidgeBasis="LocalCos64")
    assert s.lam == 0.4 and s.basis == "LocalCos64" and s.window == 1024
    assert par.RidgeDrift == "calibrate" and par.RidgeLowFrequency == 16.0
    assert par.RidgeHighFrequency == FS / 2
    assert s.n_states == 24 and s.edges()[0] == 16.0 and s.edges()[-1] == pytest.approx(1024.0)
    pen = s.penalty()
    np.testing.assert_allclose(pen, pen.T)
    assert pen[0, 1] == 0.0 and pen[0, 4] == pytest.approx(s.lam * (1.0 - s.allowance))


def test_a_tile_gives_all_its_energy_to_the_cells():
    s, _ = settings()
    tiles = dict(t_lo=np.array([T0 + 0.01]), t_hi=np.array([T0 + 0.2]), f_lo=np.array([40.0]),
                 f_hi=np.array([90.0]), sweep=np.zeros(1), tc=np.array([T0]), E=np.array([3.0]))
    X = cell_sums(tiles, T0, 32, s)
    assert X.sum() == pytest.approx(3.0)
    assert X[0].sum() > 0 and X[7:].sum() == 0.0


def test_noise_is_standardised(maps):
    s, (X, origin, *_ ) = maps
    Z = background_z(X, s)
    quiet = Z[: int((CHIRP_AT - 10) / s.slot_s)]
    assert abs(np.nanmean(quiet)) < 0.1
    assert np.nanstd(quiet) == pytest.approx(1.0, abs=0.15)


def test_the_causal_background_is_the_streaming_one(maps):
    s, (X, *_ ) = maps
    batch = background_z(X, s, causal=True)
    bg = CausalBackground(s)
    rows = np.array([bg.update(row) for row in X])
    np.testing.assert_allclose(rows, batch, equal_nan=True)
    assert np.isnan(rows[: s.slots(s.background_min_s)]).all()


def test_the_offline_ridge_finds_the_chirp(maps, stream):
    s, (X, origin, offsets, coeffs, sigmas) = maps
    found, used = offline_ridge(background_z(X, s), origin, s, X)
    assert used.drift is not None and used.xdrop > 0
    best = found[0]
    assert T0 + CHIRP_AT - 0.5 < best.t_end < T0 + CHIRP_AT + CHIRP_S + 0.5
    assert best.t_end - best.t_start > 1.0
    assert best.f_high[-1] > best.f_low[0]
    assert best.score > 1.5 * found[1].score
    start, y, energy = ridge_waveform(best, coeffs[0.0], sigmas[0.0], T0 + offsets / FS, FS,
                                      origin, used)
    noise, c = stream
    i = int(round((start - T0) * FS))
    match = np.dot(y, c[i:i + y.size]) / np.sqrt(energy * np.sum(c ** 2))
    assert energy > 0.5 * SNR ** 2 and match > 0.3


def test_the_causal_ridge_finds_the_chirp_with_its_latency(maps):
    s, (X, origin, *_ ) = maps
    b, x = calibrate_drift(background_z(X, s), s)
    s = replace(s, drift=b, xdrop=x)
    Z = background_z(X, s, causal=True)
    ridge = StreamingRidge(s, origin)
    found = []
    for k in range(len(Z)):
        found += ridge.update(Z[k], X[k])
        if k == int((CHIRP_AT + 1.0 - (origin - T0)) / s.slot_s):
            running = ridge.running()
            assert running is not None and running.t_end == pytest.approx(origin + (k + 1) * s.slot_s)
    found += ridge.flush()
    best = max(found, key=lambda c: c.score)
    assert T0 + CHIRP_AT - 0.5 < best.t_end < T0 + CHIRP_AT + CHIRP_S + 0.5
    assert 0.0 <= best.latency < 2.0 and best.emitted_at >= best.t_end


def test_the_causal_ridge_uses_only_the_past(maps):
    """Changing the rows after slot K changes nothing emitted at or before it."""
    s, (X, origin, *_ ) = maps
    b, x = calibrate_drift(background_z(X, s), s)
    s = replace(s, drift=b, xdrop=x)
    K = int((CHIRP_AT + 1.0 - (origin - T0)) / s.slot_s)

    def emitted(Xin):
        bg, ridge, out = CausalBackground(s), StreamingRidge(s, origin), []
        for row in Xin:
            out += ridge.update(bg.update(row), row)
        return [c.row() for c in out if c.emitted_at <= origin + (K + 1) * s.slot_s]

    altered = X.copy()
    altered[K + 1:] = np.random.default_rng(5).permutation(altered[K + 1:]) * 3.0
    assert emitted(X) == emitted(altered)
    assert emitted(X)


def test_the_network_takes_the_smaller_z():
    a, b = np.array([[1.0, 5.0]]), np.array([[3.0, 2.0]])
    np.testing.assert_array_equal(combine_min(a, b), [[1.0, 2.0]])


def test_box_scores_are_standardised(maps):
    s, (X, *_ ) = maps
    Z = background_z(X, s)
    b, x = calibrate_drift(Z, s)
    boxes = box_scores(Z, (0.5, 2.0), replace(s, drift=b))
    st = standardise(boxes)
    assert set(st) == {0.5, 2.0} and all(sd > 0 for _, sd in st.values())


def test_a_dechirped_map_runs():
    s, _ = settings(RidgeDechirpAlphas=[-128.0, 0.0, 128.0])
    x = np.random.default_rng(1).standard_normal(int(16 * FS))
    X, origin, offsets, coeffs, sigmas = stream_cells(x, T0, FS, s)
    assert set(coeffs) == {-128.0, 0.0, 128.0} and X.shape[1] == s.n_states


@pytest.mark.parametrize("alphas", [(), (-128.0, 0.0, 128.0)])
def test_the_streaming_cells_are_the_batch_cells(alphas):
    from wdf.analysis.ridge_tracking import StreamingCells

    s, _ = settings(RidgeDechirpAlphas=list(alphas))
    x = np.random.default_rng(2).standard_normal(int(20 * FS))
    X, origin, offsets, coeffs, sigmas = stream_cells(x, T0, FS, s)
    cells = StreamingCells(s, T0, FS)
    rows = [cells.push(x[a:a + 3001]) for a in range(0, x.size, 3001)]
    rows = np.vstack(rows)
    assert cells.origin == origin
    np.testing.assert_allclose(rows, X[: len(rows)], atol=1e-10)
    assert len(rows) == len(X)
    starts, held, scales = cells.held(alphas[0] if alphas else 0.0)
    np.testing.assert_allclose(held[-1], coeffs[alphas[0] if alphas else 0.0][-1])
