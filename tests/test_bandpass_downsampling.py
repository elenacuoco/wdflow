"""The conditioning front end: zero phase, continuous across blocks, and an
anti-alias that stops what would otherwise fold back into the analysed band."""
import numpy as np
import pytest
from types import SimpleNamespace
from wdf.filtering import sosfiltfilt

# The conditioning stage is built on the compiled core.
pytest.importorskip("py4tsa")

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
    downstream. The edges are held to the same bound as the interior.

    The bound is below what a single-precision handover alone would leave,
    2^-24 of each sample's size, so it also holds the block to the double
    precision it is computed in."""
    filt = BandPassDownSampling(parameters())
    samples = noise(SAMPLING * 12, seed=1)
    reference = sosfiltfilt(filt.sos, samples)[::FACTOR]

    blocks, starts = stream(filt, samples)

    for block, start in list(zip(blocks, starts))[1:]:
        offset = int(round(start * RESAMPLING))
        expected = reference[offset:offset + len(block)]
        error = np.abs(block - expected) / np.std(expected)
        assert error.max() < 1e-9


def test_the_conditioned_block_is_handed_over_in_double_precision():
    """The view the whitening reads stores double precision, so the samples it
    receives are the samples the filter computed, bit for bit. A narrowing on
    the way adds a white rounding noise that, in a band the conditioning has
    emptied, is all the band holds, and the whitening lifts it back up."""
    filt = BandPassDownSampling(parameters(), estimation=True)
    samples = noise(SAMPLING * 4, seed=7) * 1e-21
    expected = sosfiltfilt(filt.sos, samples)[::FACTOR]

    assert np.array_equal(read_back(filt.Process(_Block(samples, 0.0))), expected)


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


def test_a_block_that_does_not_decimate_whole_is_refused():
    """The decimation restarts at each block's first sample, so a block that
    does not hold a whole number of decimated samples moves the phase of
    everything after it. The stream carries no sign of it, so it is refused."""
    filt = BandPassDownSampling(parameters())
    samples = noise(6 * SAMPLING)
    odd = SAMPLING + 1
    with pytest.raises(ValueError, match="decimation phase"):
        for first in range(0, len(samples) - odd + 1, odd):
            filt.Process(_Block(samples[first:first + odd], first / SAMPLING))


def test_the_phase_is_the_same_whatever_the_block_length():
    """Two readings of one stretch, in blocks of different lengths: the joins
    and the decimation phase are the only things that differ between them, so
    the samples they emit for the same instants must agree to rounding."""
    samples = noise(40 * SAMPLING, seed=5)
    long_blocks, long_starts = stream(BandPassDownSampling(parameters()),
                                      samples, block=4 * SAMPLING)
    short_blocks, short_starts = stream(BandPassDownSampling(parameters()),
                                        samples, block=3 * SAMPLING)
    long_stream = np.concatenate(long_blocks)
    short_stream = np.concatenate(short_blocks)

    # Both start where their own first emitted block starts; align on time.
    shift = int(round((short_starts[0] - long_starts[0]) * RESAMPLING))
    n = min(len(long_stream) - max(shift, 0), len(short_stream) + min(shift, 0))
    left = long_stream[max(shift, 0):max(shift, 0) + n]
    right = short_stream[max(-shift, 0):max(-shift, 0) + n]

    assert n > 10 * RESAMPLING
    assert np.max(np.abs(left - right)) < 1e-6 * np.std(left)


def test_the_estimation_branch_returns_the_block_it_was_given():
    """The autoregressive fit is handed one complete stretch and needs it back
    immediately; there is nothing to wait for."""
    filt = BandPassDownSampling(parameters(), estimation=True)
    view = filt.Process(_Block(noise(SAMPLING * 4), 0.0))

    assert view is not None
    assert view.GetSize() == SAMPLING * 4 // FACTOR


# ------------------------------------------------------------------ settling

def _with_notch(bandwidth, low_cut=12.0):
    """The band-pass with a notch at 60 Hz of the given width stacked on."""
    from scipy.signal import iirnotch, tf2sos
    b, a = iirnotch(60.0, 60.0 / bandwidth, fs=SAMPLING)
    return np.vstack([tf2sos(b, a), BandPassDownSampling(parameters(low_cut)).sos])


def test_the_settling_is_followed_to_its_end():
    """A narrow notch rings for tens of seconds, far past any window chosen in
    advance. The length returned is where the response really falls below the
    floor, and it stays below over a stretch several times as long."""
    from scipy.signal import sosfilt
    sos = _with_notch(0.3)
    n = settling_length(sos, SAMPLING)
    impulse = np.zeros(4 * n)
    impulse[0] = 1.0
    response = np.abs(sosfilt(sos, impulse))

    assert n > 8 * SAMPLING
    assert response[n - 1] > 1e-12 * response.max()
    assert response[n:].max() <= 1e-12 * response.max()


def test_a_filter_that_does_not_settle_within_the_limit_is_refused():
    """Reporting the limit instead would put the unsettled transient into
    every emitted block."""
    with pytest.raises(ValueError, match="still rings"):
        settling_length(_with_notch(0.3), SAMPLING, limit_s=10.0)


def test_a_stretch_read_with_its_context_is_conditioned_as_the_stream():
    """The stretch a noise model is fitted on is filtered as the stream is
    filtered there, edges included, when it is read with the settling of real
    data on each side. Filtered alone, its edges carry the filter's start."""
    filt = BandPassDownSampling(parameters(), estimation=True)
    samples = noise(SAMPLING * 30, seed=11)
    reference = sosfiltfilt(filt.sos, samples)[::FACTOR]
    context = int(np.ceil(filt.padlen / FACTOR)) * FACTOR
    first, length = 10 * SAMPLING, 8 * SAMPLING

    view = filt.condition_stretch(
        _Block(samples[first - context:first + length + context],
               (first - context) / SAMPLING), context)
    kept = read_back(view)
    expected = reference[first // FACTOR:(first + length) // FACTOR]

    assert view.GetStart() == pytest.approx(first / SAMPLING)
    assert np.max(np.abs(kept - expected)) / np.std(expected) < 1e-9

    alone = read_back(BandPassDownSampling(parameters(), estimation=True).Process(
        _Block(samples[first:first + length], first / SAMPLING)))
    assert np.max(np.abs(alone - expected)) / np.std(expected) > 1e-3


def test_a_context_shorter_than_the_settling_is_refused():
    filt = BandPassDownSampling(parameters(), estimation=True)
    with pytest.raises(ValueError, match="settles over"):
        filt.condition_stretch(_Block(noise(SAMPLING * 20), 0.0), filt.padlen - 1)
