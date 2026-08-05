import numpy as np
import pytest

from wdf.observers.ParameterEstimationObserver import get_most_important_frequencies

FS = 2048.0
N = 512


def test_normal_signal_returns_ordered_band():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=N)
    freqMean, freqMin, freqMax, freqPeak = get_most_important_frequencies(signal, FS)
    assert 0.0 <= freqMin <= freqMean <= freqMax <= FS / 2
    assert freqMin <= freqPeak <= freqMax


def test_all_zero_signal_collapses_the_band():
    """A signal with no energy occupies no band, rather than the whole of it."""
    signal = np.zeros(N)
    freqMean, freqMin, freqMax, freqPeak = get_most_important_frequencies(signal, FS)
    assert freqMin == pytest.approx(0.0)
    assert freqMax == pytest.approx(0.0)
    assert freqMean == pytest.approx(0.0)
    assert freqPeak == pytest.approx(0.0)


def test_single_nonzero_sample_does_not_raise():
    signal = np.zeros(N)
    signal[10] = 1.0
    freqMean, freqMin, freqMax, freqPeak = get_most_important_frequencies(signal, FS)
    assert freqMin <= freqPeak <= freqMax
