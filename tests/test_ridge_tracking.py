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


# ================================================================ RidgeMaxDown
def planted(slots, states, z=4.0, shape=(400, 24), seed=3):
    """A quiet map (z about -1) with z on the given cells."""
    Z = -1.0 + 0.1 * np.random.default_rng(seed).standard_normal(shape)
    Z[np.asarray(slots), np.asarray(states)] = z
    return Z


def fixed(**kw):
    s, par = settings(**kw)
    return replace(s, drift=0.5, xdrop=4.0), par


def test_max_down_unset_is_the_symmetric_penalty():
    s, par = settings()
    assert s.max_down is None and par.RidgeMaxDown is None
    np.testing.assert_array_equal(s.penalty(), s.penalty().T)


def test_max_down_forbids_only_the_larger_moves_down():
    sym, _ = settings()
    s, par = settings(RidgeMaxDown=1)
    assert s.max_down == 1 and par.RidgeMaxDown == 1
    P, Q = s.penalty(), sym.penalty()
    frm, to = np.indices(P.shape)
    allowed = frm - to <= 1
    np.testing.assert_array_equal(P[allowed], Q[allowed])
    assert np.all(np.isinf(P[~allowed]))
    with pytest.raises(ValueError):
        settings(RidgeMaxDown=-1)


def test_a_rising_track_is_followed_unchanged_with_max_down():
    slots = np.arange(100, 120)
    states = np.arange(2, 22)
    Z = planted(slots, states)
    (a, _), (b, _) = (offline_ridge(Z, T0, fixed()[0]), offline_ridge(Z, T0, fixed(RidgeMaxDown=1)[0]))
    np.testing.assert_array_equal(a[0].slots, b[0].slots)
    np.testing.assert_array_equal(a[0].states, b[0].states)
    assert a[0].score == b[0].score
    assert set(zip(slots, states)) <= set(zip(b[0].slots.tolist(), b[0].states.tolist()))


def test_a_falling_track_is_not_followed_more_than_max_down():
    slots = np.arange(200, 210)
    states = np.arange(22, 2, -2)        # down two states a slot
    Z = planted(slots, states)
    free, _ = offline_ridge(Z, T0, fixed()[0])
    assert set(zip(slots, states)) <= set(zip(free[0].slots.tolist(), free[0].states.tolist()))
    limited, _ = offline_ridge(Z, T0, fixed(RidgeMaxDown=1)[0])
    for c in limited:
        assert np.all(np.diff(c.states) >= -1)
    best = max(len(set(zip(slots, states)) & set(zip(c.slots.tolist(), c.states.tolist())))
               for c in limited)
    assert best == 1


def test_the_backward_extension_respects_max_down():
    # A loud cell high in frequency the slot before a run at state 5: going
    # back in time it is a move up, forward a move down by 10 states.
    Z = planted(np.arange(10, 20), np.full(10, 5))
    Z[9, 15] = 20.0
    for max_down, taken in ((None, True), (1, False)):
        s = fixed(**({} if max_down is None else dict(RidgeMaxDown=max_down)))[0]
        ridge = StreamingRidge(s, T0)
        for row in Z[:20]:
            ridge.update(row)
        ext = ridge._extend_back(10, 5)
        assert ((9, 15) in [(k, st) for k, st, _ in ext]) is taken
    # Mirrored: a loud cell low in frequency before the run is a move up, allowed.
    Z = planted(np.arange(10, 20), np.full(10, 15))
    Z[9, 5] = 20.0
    ridge = StreamingRidge(fixed(RidgeMaxDown=1)[0], T0)
    for row in Z[:20]:
        ridge.update(row)
    assert (9, 5) in [(k, st) for k, st, _ in ridge._extend_back(10, 15)]


def test_the_calibration_and_the_boxes_run_with_max_down():
    from wdf.analysis.ridge_tracking import _greedy_gains

    Z = np.random.default_rng(4).standard_normal((2000, 24))
    s, _ = settings(RidgeMaxDown=1)
    b, x = calibrate_drift(Z, s)
    assert np.isfinite(b) and np.isfinite(x)
    boxes = box_scores(Z, [0.5], replace(s, drift=b, xdrop=x))
    assert np.all(np.isfinite(boxes[0.5][1]))
    # Backward greedy gains price the move earlier -> later.
    Zs = np.full((4, 24), -5.0)
    Zs[3, 2], Zs[2, 20] = 1.0, 10.0
    G = _greedy_gains(Zs, np.array([3]), 1, s.penalty(), 0.0, direction=-1)
    assert G[0, 0] < 10.0                # 20 -> 2 is forbidden
    Zs[2, 2] = -4.0
    assert _greedy_gains(Zs, np.array([3]), 1, s.penalty(), 0.0, direction=-1)[0, 0] == -4.0
