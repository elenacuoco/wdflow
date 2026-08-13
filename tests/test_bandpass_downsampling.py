"""The conditioning front end: zero phase, continuous across blocks, and an
anti-alias that stops what would otherwise fold back into the analysed band."""
import numpy as np
import pytest
from types import SimpleNamespace
from scipy.signal import sosfiltfilt

# The conditioning stage is built on the compiled core.
pytest.importorskip("pytsa")

from wdf.processes.BandPassDownSampling import (BandPassDownSampling,
                                                settling_length)

SAMPLING, FACTOR = 16384, 8
RESAMPLING = SAMPLING // FACTOR          # 2048 Hz, final Nyquist 1024 Hz


def parameters(low_cut=12.0):
    return SimpleNamespace(sampling=SAMPLING, resampling=RESAMPLING,
                           ResamplingFactor=FACTOR, LowFrequencyCut=low_cut)


class _Block:
    """The slice of the SeqView interface the filter uses."""

    def __init__(self, samples, start):
        self._samples, self._start = np.asarray(samples, dtype=float), float(start)

    def GetSize(self):
        return len(self._samples)

    def GetStart(self):
        return self._start

    def GetY(self, _channel, i):
        return self._samples[i]


def read_back(view):
    return np.array([view.GetY(0, i) for i in range(view.GetSize())])


def stream(filt, samples, block=SAMPLING, t0=0.0):
    """Push `samples` through one block at a time, as the worker does.

    A block comes back only once its lookahead has been read, so there are
    fewer outputs than reads.
    """
    out, starts = [], []
    for first in range(0, len(samples) - block + 1, block):
        view = filt.Process(_Block(samples[first:first + block],
                                   t0 + first / SAMPLING))
        if view is not None:
            out.append(read_back(view))
            starts.append(view.GetStart())
    return out, starts


def noise(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


# ----------------------------------------------------------------- continuity

def test_the_stream_has_no_seam_at_the_block_joins():
    """A filter applied to each block on its own leaves a step at every join,
    and a step is short in time and broad in frequency -- it manufactures
    triggers in the finest wavelet scales at a fixed rate."""
    filt = BandPassDownSampling(parameters())
    blocks, _ = stream(filt, noise(SAMPLING * 12))

    stack = np.vstack(blocks[1:])
    rms = np.sqrt((stack ** 2).mean(axis=0))
    n = stack.shape[1]
    middle = np.median(rms[n // 4: 3 * n // 4])

    assert np.median(rms[:32]) / middle < 1.5
    assert np.median(rms[-32:]) / middle < 1.5
    assert max(rms[:8].max(), rms[-8:].max()) / middle < 2.0


def test_a_block_matches_the_whole_stream_filtered_at_once():
    """The point of the lookahead. Compared per sample rather than by an
    aggregate: the whitening applies its largest gain at the band edges, where
    the residual lives, so an error invisible in the RMS is not invisible
    downstream. The edges are held to the same bound as the interior."""
    filt = BandPassDownSampling(parameters())
    samples = noise(SAMPLING * 12, seed=1)
    reference = sosfiltfilt(filt.sos, samples)[::FACTOR]

    blocks, starts = stream(filt, samples)

    for block, start in list(zip(blocks, starts))[1:]:
        offset = int(round(start * RESAMPLING))
        expected = reference[offset:offset + len(block)]
        error = np.abs(block - expected) / np.std(expected)
        assert error.max() < 1e-5


def test_the_latency_is_declared():
    """A block is held until its lookahead arrives; how much is held is not a
    secret the caller has to infer."""
    filt = BandPassDownSampling(parameters())
    assert filt.latency_s == 0.0
    stream(filt, noise(SAMPLING * 6))
    assert filt.latency_s >= filt.padlen / SAMPLING


def test_the_timestamps_carry_the_time_of_the_samples_held():
    filt = BandPassDownSampling(parameters())
    _, starts = stream(filt, noise(SAMPLING * 8), t0=1000.0)

    assert starts[0] == pytest.approx(1000.0)
    for k in range(1, len(starts)):
        assert starts[k] == pytest.approx(starts[k - 1] + 1.0)


# ------------------------------------------------------------------ zero phase

def test_a_transient_is_not_displaced_in_time():
    """Every parameter the search reports is read off the reconstruction, so a
    filter that moves the transient corrupts all of them."""
    filt = BandPassDownSampling(parameters())
    n = SAMPLING * 8
    samples = np.zeros(n)
    centre = int(4.5 * SAMPLING)
    t = (np.arange(n) - centre) / SAMPLING
    samples += np.exp(-(t / 0.01) ** 2) * np.sin(2 * np.pi * 200.0 * t)

    blocks, starts = stream(filt, samples)

    loudest = int(np.argmax([np.abs(b).max() for b in blocks]))
    peak = starts[loudest] + int(np.argmax(np.abs(blocks[loudest]))) / RESAMPLING
    assert peak == pytest.approx(centre / SAMPLING, abs=2e-3)


# ------------------------------------------------------------------ the band

def test_a_tone_above_the_new_nyquist_does_not_fold_back():
    """What the anti-alias is for: without enough attenuation before the
    decimated Nyquist, a tone above it reappears inside the analysed band."""
    filt = BandPassDownSampling(parameters())
    n = SAMPLING * 12
    t = np.arange(n) / SAMPLING
    intruder = 1500.0                      # above the final Nyquist of 1024 Hz
    blocks, _ = stream(filt, np.sin(2 * np.pi * intruder * t))
    settled = np.concatenate(blocks[2:])

    spectrum = np.abs(np.fft.rfft(settled * np.hanning(len(settled))))
    freq = np.fft.rfftfreq(len(settled), 1.0 / RESAMPLING)
    folded = abs(intruder - RESAMPLING)    # where it would land: 548 Hz

    assert spectrum[np.abs(freq - folded) < 5.0].max() / len(settled) < 1e-3


def test_a_tone_inside_the_band_survives():
    filt = BandPassDownSampling(parameters())
    n = SAMPLING * 12
    t = np.arange(n) / SAMPLING
    blocks, _ = stream(filt, np.sin(2 * np.pi * 200.0 * t))

    assert np.std(np.concatenate(blocks[2:])) == pytest.approx(np.sqrt(0.5), rel=0.05)


def test_a_tone_below_the_high_pass_is_removed():
    filt = BandPassDownSampling(parameters(low_cut=12.0))
    n = SAMPLING * 12
    t = np.arange(n) / SAMPLING
    blocks, _ = stream(filt, np.sin(2 * np.pi * 3.0 * t))

    assert np.std(np.concatenate(blocks[2:])) < 0.05


# ------------------------------------------------------------------ contracts

def test_the_settling_length_is_measured_not_assumed():
    """A steep filter close to Nyquist rings far longer than its order says,
    and the floor is set by what survives the whitening, not by what looks
    negligible in the conditioned data."""
    filt = BandPassDownSampling(parameters())
    assert filt.padlen == settling_length(filt.sos, SAMPLING)
    assert settling_length(filt.sos, SAMPLING, floor=1e-5) < filt.padlen


def test_band_edges_that_cross_are_refused():
    """The high-pass edge above the anti-alias edge leaves no pass band, and
    the filter design refuses it rather than producing an empty one."""
    with pytest.raises(ValueError):
        BandPassDownSampling(parameters(low_cut=2000.0))


def test_the_estimation_branch_returns_the_block_it_was_given():
    """The autoregressive fit is handed one complete stretch and needs it back
    immediately; there is nothing to wait for."""
    filt = BandPassDownSampling(parameters(), estimation=True)
    view = filt.Process(_Block(noise(SAMPLING * 4), 0.0))

    assert view is not None
    assert view.GetSize() == SAMPLING * 4 // FACTOR
