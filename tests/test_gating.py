"""Gates on the whitened stream: the census of its transients, and the weights
that take the loud ones out without leaving an edge."""
import numpy as np
import pytest

from wdf.processes.gating import (gate_weights, merged, octave_bands,
                                  robust_sigma, transients)

RATE = 2048.0


def stream(seconds=300, seed=0, glitches=()):
    """Unit white noise plus a Gaussian-windowed sinusoid per
    `(centre_s, amplitude, frequency, width_s)`."""
    n = int(seconds * RATE)
    x = np.random.default_rng(seed).standard_normal(n)
    for centre, amplitude, frequency, width in glitches:
        t = np.arange(n) / RATE - centre
        x += amplitude * np.exp(-(t / width) ** 2) * np.sin(2 * np.pi * frequency * t)
    return x


def test_the_octaves_halve_down_from_the_nyquist_frequency():
    assert octave_bands(RATE, 16.0) == ((16.0, 32.0), (32.0, 64.0), (64.0, 128.0),
                                        (128.0, 256.0), (256.0, 512.0), (512.0, 1024.0))
    assert octave_bands(RATE, 40.0)[0] == (40.0, 64.0)
    with pytest.raises(ValueError):
        octave_bands(RATE, 1024.0)


def test_gaussian_noise_holds_no_transient():
    assert len(transients(stream(), RATE, octave_bands(RATE, 16.0))) == 0
    assert robust_sigma(stream()) == pytest.approx(1.0, rel=0.01)


def test_each_transient_is_found_with_its_whole_excursion():
    """Every sample above the flag belongs to a transient, and the height is
    read in the band where the transient stands highest: a narrow-band glitch
    is far taller in its octave than in the broadband stream."""
    x = stream(glitches=[(100.0, 200.0, 40.0, 0.2), (200.0, 12.0, 300.0, 0.01)])
    found = transients(x, RATE, octave_bands(RATE, 16.0))

    assert len(found) == 2
    centres = 0.5 * (found.start + found.stop) / RATE
    assert centres == pytest.approx([100.0, 200.0], abs=0.05)
    assert found.peak[0] > 2 * np.abs(x).max()
    assert 6.0 < found.peak[1] < 50.0
    assert np.all(found.stop > found.start)


def test_the_weights_zero_the_gate_and_taper_its_edges():
    gates = np.array([[10.0, 11.0]])
    times = np.array([9.0, 9.75, 9.875, 10.0, 10.5, 11.0, 11.125, 11.25, 12.0])
    weights = gate_weights(times, gates, taper_s=0.25)

    assert weights == pytest.approx([1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0])
    fine = np.linspace(9.0, 12.0, 30001)
    assert np.max(np.abs(np.diff(gate_weights(fine, gates, 0.25)))) < 1e-3


def test_overlapping_gates_are_one_gate():
    assert np.array_equal(merged(np.array([[5.0, 6.0], [1.0, 2.0], [1.5, 3.0]])),
                          np.array([[1.0, 3.0], [5.0, 6.0]]))
    assert gate_weights(np.array([2.5]), np.array([[1.0, 2.0], [3.0, 4.0]]), 0.25)[0] == 1.0


def test_a_gated_stream_has_no_edge_left():
    """The gate is the last operation before the search: what remains around
    it is the noise, tapered, and not a transient of its own."""
    x = stream(glitches=[(100.0, 400.0, 40.0, 0.3)])
    bands = octave_bands(RATE, 16.0)
    found = transients(x, RATE, bands)
    gates = np.column_stack([found.start, found.stop])[found.peak >= 50.0] / RATE
    gated = x * gate_weights(np.arange(x.size) / RATE, gates, 0.25)

    assert gates.shape[0] == 1
    assert len(transients(gated, RATE, bands)) == 0
