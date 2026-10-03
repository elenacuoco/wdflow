"""M4 glitch subtraction: whitened sine-Gaussian matching pursuit.

A known coloured process (a red resonance, its prediction-error filter exact)
is whitened with the truncated zero-phase filter the worker runs; a loud
sine-Gaussian glitch must be removed to the noise level, pure noise must give
no atom, a chirp away from the glitch must be left untouched, and the
subtraction in the conditioned domain must equal the subtraction of the
atoms' whitened images in the whitened domain.
"""
import os

import numpy as np
import pytest
from scipy.signal import fftconvolve, lfilter

from types import SimpleNamespace

from wdf.processes.glitch_subtraction import (
    Atom,
    GlitchModel,
    M4Settings,
    SineGaussianBank,
    bank_frequencies,
    fit_glitches,
    fit_seeds,
    glitch_seeds,
    matching_pursuit,
    protect_coincident,
    whitened_census,
)

FS = 2048.0
#: The 3 Oct settings: bank from 8 Hz, census from 16 Hz, extent from 64 Hz.
SETTINGS = M4Settings.from_parameters(SimpleNamespace(LowFrequencyCut=8.0, MinFrequency=16.0))
START = 1000000000.0
SECONDS = 160
R = 0.995
#: A red process with a resonance at 60 Hz: x = lfilter([1], COLOUR, white).
COLOUR = np.convolve([1.0, -2 * R * np.cos(2 * np.pi * 60.0 / FS), R * R], [1.0, -0.9])


@pytest.fixture(scope="module")
def whitening():
    from wdf.processes.zero_phase_whitening import TruncatedWhitening
    ar = np.concatenate([[1.0], -COLOUR[1:]])
    w = TruncatedWhitening(ar, int(FS), 0, duration=4.0, sampling=FS)
    return w.taps, w.sigma


@pytest.fixture(scope="module")
def bank(whitening):
    return SineGaussianBank(*whitening, FS, SETTINGS.frequencies(FS), qs=SETTINGS.qs)


def _noise(seed):
    rng = np.random.default_rng(seed)
    return lfilter([1.0], COLOUR, rng.standard_normal(int(SECONDS * FS)))


def _sine_gaussian(centre, frequency, q, amplitude, phase=0.3):
    t = START + np.arange(int(SECONDS * FS)) / FS - centre
    tau = q / (2 * np.pi * frequency) / np.sqrt(2)
    return amplitude * np.exp(-t ** 2 / (2 * tau ** 2)) * np.cos(2 * np.pi * frequency * t + phase)


def _glitch(bank, whitened_snr=150.0, centre=START + 70.0123):
    """Two off-grid sine-Gaussians scaled to a whitened SNR (a blip-like glitch)."""
    g = _sine_gaussian(centre, 137.0, 9.0, 1.0) + _sine_gaussian(centre + 0.004, 310.0, 3.0, 0.6, 1.1)
    snr = np.sqrt(np.sum(bank.whiten(g) ** 2))
    return g * whitened_snr / snr


def _chirp(bank, start, whitened_snr=10.0, f0=40.0, f1=250.0, length=1.5):
    t = START + np.arange(int(SECONDS * FS)) / FS - start
    inside = (t >= 0) & (t < length)
    f = f0 + (f1 - f0) * np.clip(t, 0, length) / length
    phase = 2 * np.pi * np.cumsum(f) / FS
    c = np.where(inside, np.sin(phase) * np.sin(np.pi * np.clip(t, 0, length) / length) ** 2, 0.0)
    return c * whitened_snr / np.sqrt(np.sum(bank.whiten(c) ** 2))


def _interior(bank):
    k = (bank.taps.size - 1) // 2 + int(2 * FS)
    return slice(k, int(SECONDS * FS) - k)


def test_glitch_removed_to_noise_level(whitening, bank):
    noise = _noise(1)
    x = noise + _glitch(bank)
    model = fit_glitches(x, START, FS, *whitening, SETTINGS, bank=bank)
    assert len(model.seeds) == 1 and model.seeds[0]["fitted"]
    assert abs(model.seeds[0]["centre"] - (START + 70.0123)) < 0.05
    assert 1 <= len(model) <= 100
    residual = bank.whiten(model.subtract(x, START, FS))
    white_noise = bank.whiten(noise)
    near = np.abs(START + np.arange(x.size) / FS - (START + 70.0123)) < 0.5
    # before: the glitch at whitened SNR 150; after: the noise's own level
    assert np.max(np.abs(bank.whiten(x)[near])) > 20
    assert np.sqrt(np.mean((residual - white_noise)[near] ** 2)) < 0.3
    assert np.max(np.abs(residual[near])) < 5.0
    assert np.std(residual[_interior(bank)]) == pytest.approx(np.std(white_noise[_interior(bank)]), rel=0.02)


def test_nothing_fitted_on_pure_noise(whitening, bank):
    for seed in (2, 3):
        model = fit_glitches(_noise(seed), START, FS, *whitening, SETTINGS, bank=bank)
        assert len(model) == 0 and model.seeds == []
    # a pursuit forced on pure noise stops at once at the default SNR 6
    x = _noise(4)
    atoms, local = matching_pursuit(x, START, bank, START + 80.0, 0.3)
    assert atoms == [] and not local.any()


def test_chirp_far_from_glitch_is_untouched(whitening, bank):
    chirp_start = START + 100.0
    x = _noise(5) + _glitch(bank) + _chirp(bank, chirp_start)
    model = fit_glitches(x, START, FS, *whitening, SETTINGS, bank=bank)
    assert len(model) >= 1
    t = START + np.arange(x.size) / FS
    span = (t >= chirp_start - 2.0) & (t < chirp_start + 3.5)
    cleaned = model.subtract(x, START, FS)
    np.testing.assert_array_equal(cleaned[span], x[span])
    for atom in model.atoms:
        assert abs(atom.centre - (START + 70.0)) < 1.0


def test_conditioned_subtraction_equals_whitened_subtraction(whitening, bank):
    taps, sigma = whitening
    x = _noise(6) + _glitch(bank)
    model = fit_glitches(x, START, FS, taps, sigma, SETTINGS, bank=bank)
    assert len(model) >= 1
    # the atoms' whitened images, built atom by atom in the whitened domain
    images = np.zeros(x.size)
    for atom in model.atoms:
        images += bank.whiten(atom.evaluate(START, x.size))
    lhs = bank.whiten(model.subtract(x, START, FS))
    rhs = bank.whiten(x) - images
    np.testing.assert_allclose(lhs[_interior(bank)], rhs[_interior(bank)], atol=1e-9)


def test_model_is_independent_of_the_block_grid(whitening, bank):
    x = _noise(7) + _glitch(bank)
    model = fit_glitches(x, START, FS, *whitening, SETTINGS, bank=bank)
    whole = model.evaluate(START, FS, x.size)
    pieces, at = [], 0
    for size in (int(69.9 * FS), 333, int(0.25 * FS), x.size):
        size = min(size, x.size - at)
        m = model.evaluate(START + at / FS, FS, size)
        pieces.append(np.zeros(size) if m is None else m)
        at += size
        if at >= x.size:
            break
    np.testing.assert_allclose(np.concatenate(pieces), whole, atol=1e-12)


def test_extent_ignores_the_low_octave(whitening, bank):
    """The seed's half-width follows the glitch, not the 16-32 Hz whitening memory."""
    x = _noise(8) + _glitch(bank)
    w = bank.whiten(x)
    k = _interior(bank)
    seeds = glitch_seeds(w[k], START + k.start / FS, FS, **SETTINGS.census())
    assert len(seeds) == 1
    assert seeds[0]["half_width"] < 0.3
    assert seeds[0]["last"] - seeds[0]["first"] >= seeds[0]["extent"]


def test_atom_rejects_another_rate():
    atom = Atom(centre=START, frequency=100.0, q=4.0, alpha=1.0, beta=0.0, snr=10.0, rate=FS, half_length=40)
    with pytest.raises(ValueError):
        GlitchModel([atom]).evaluate(START - 1.0, 4096.0, 8192)


def test_bank_refuses_a_phased_filter():
    with pytest.raises(ValueError):
        SineGaussianBank(np.array([1.0, 0.5, 0.0]), 1.0, FS, (100.0,))


def test_the_bank_follows_the_rate():
    assert bank_frequencies(2048.0, 8.0) == pytest.approx(tuple(np.geomspace(8.0, 860.0, 40)))
    at_4096 = bank_frequencies(4096.0, 8.0)
    assert at_4096[0] == 8.0 and at_4096[-1] == pytest.approx(1720.0)
    with pytest.raises(ValueError):
        bank_frequencies(2048.0, 900.0)


def test_the_settings_are_parameters_and_are_recorded():
    par = SimpleNamespace(LowFrequencyCut=12.0, GlitchQs=[2.0, 8.0], GlitchFlag=5.0,
                          GlitchMergeGap=0.5)
    s = M4Settings.from_parameters(par)
    assert s.qs == (2.0, 8.0) and s.flag == 5.0 and s.merge_s == 0.5
    assert s.census_low == 12.0 and s.extent_low == 48.0
    assert par.GlitchProtectCoincident is True and par.GlitchTopFraction == pytest.approx(860 / 1024)
    assert par.GlitchCensusLow == 12.0 and par.GlitchQs == [2.0, 8.0]


def _seed(first, last):
    return dict(centre=0.5 * (first + last), first=first, last=last, half_width=0.1)


def test_coincident_seeds_are_protected():
    """H1 and L1 within their 10 ms light travel time: a possible signal."""
    seeds = {"H1": [_seed(100.000, 100.050), _seed(200.0, 200.05)],
             "L1": [_seed(100.060, 100.100)],
             "V1": [_seed(300.0, 300.1)]}
    protect_coincident(seeds, jitter_s=0.0)
    assert seeds["H1"][0]["protected"] and seeds["L1"][0]["protected"]
    assert seeds["H1"][0]["coincident_with"] == ["L1"]
    assert not seeds["H1"][1]["protected"] and not seeds["V1"][0]["protected"]
    far = {"H1": [_seed(100.0, 100.05)], "L1": [_seed(100.08, 100.1)]}
    protect_coincident(far, jitter_s=0.0)
    assert not far["H1"][0]["protected"]
    protect_coincident(far, jitter_s=0.03)
    assert far["H1"][0]["protected"]


def test_a_protected_seed_is_not_subtracted(whitening, bank):
    x = _noise(9) + _glitch(bank)
    seeds = whitened_census(x, START, FS, bank, SETTINGS)
    assert len(seeds) == 1
    seeds[0]["protected"] = True
    model = fit_seeds(x, START, FS, bank, seeds, SETTINGS)
    assert len(model) == 0 and seeds[0]["fitted"] is False
    np.testing.assert_array_equal(model.subtract(x, START, FS), x)


def test_the_octaves_reach_down_to_the_census_low():
    from wdf.processes.census import octave_bands
    assert octave_bands(2048.0, 16.0)[0] == (16.0, 32.0)
    assert octave_bands(2048.0, 16.0)[-1] == (512.0, 1024.0)
    assert octave_bands(2048.0, 20.0)[0] == (20.0, 32.0)
