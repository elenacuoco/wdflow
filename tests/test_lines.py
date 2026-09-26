"""Spectral lines: found against the local floor, and cut down to it."""
import numpy as np
import pytest
from scipy.signal import sosfreqz, tf2zpk

from wdf.filtering import sosfiltfilt
from wdf.processes.lines import median_spectrum, notch_sections, spectral_lines

FS = 4096.0
SECONDS = 256


def stretch(lines=(), seed=0, seconds=SECONDS):
    """Unit white noise plus a sinusoid per `(frequency, amplitude)`."""
    t = np.arange(int(seconds * FS)) / FS
    x = np.random.default_rng(seed).standard_normal(t.size)
    for frequency, amplitude in lines:
        x += amplitude * np.sin(2 * np.pi * frequency * t)
    return x


def found(x, threshold=5.0):
    frequency, psd = median_spectrum(x, FS)
    return spectral_lines(frequency, psd, 6.0, 1126.4, threshold=threshold)


def test_the_lines_above_the_threshold_are_found_and_nothing_else():
    """Noise alone has no bin five times above its own median floor."""
    lines = found(stretch([(60.0, 0.5), (331.3, 0.3), (900.25, 1.0), (500.0, 0.02)]))

    assert lines.shape == (3, 3)
    assert np.allclose(lines[:, 0], [60.0, 331.3, 900.25], atol=1.0 / 16)
    assert np.all(lines[:, 2] >= 5.0)
    assert found(stretch(seed=3)).shape == (0, 3)


def test_the_threshold_is_a_height_over_the_floor():
    x = stretch([(200.0, 0.3)])
    height = found(x, threshold=1.5)[0, 2]
    assert found(x, threshold=height * 0.99).shape[0] == 1
    assert found(x, threshold=height * 1.01).shape[0] == 0


def test_a_transient_in_the_stretch_makes_no_line():
    """The floor and the lines are read on the median of the periodograms, which
    the few segments holding a transient do not move."""
    x = stretch(seed=4)
    t = (np.arange(x.size) - x.size // 2) / FS
    x += 200.0 * np.exp(-(t / 0.05) ** 2) * np.sin(2 * np.pi * 150.0 * t)
    assert found(x).shape == (0, 3)


def test_the_notch_takes_the_line_to_the_floor_and_no_further():
    """Run both ways, the gain at the centre is the inverse of the height: the
    line comes out at the floor. The zeros are inside the unit circle, so the
    floor under the line is kept, and far from the line the gain is one."""
    lines = np.array([[60.0, 0.3, 40.0], [512.7, 0.5, 120.0]])
    for row, section in zip(lines, notch_sections(lines, FS)):
        _, h = sosfreqz(section[None, :], worN=[row[0], row[0] + 50 * row[1]], fs=FS)
        zeros, poles, _ = tf2zpk(section[:3], section[3:])
        assert np.abs(h[0]) ** 2 * row[2] == pytest.approx(1.0, rel=1e-9)
        assert np.abs(h[1]) ** 2 == pytest.approx(1.0, abs=5e-3)
        assert np.abs(zeros).max() < 1.0
        assert np.abs(poles).max() < 1.0


def test_a_notched_line_stands_at_the_floor():
    x = stretch([(60.0, 0.5), (331.3, 0.3)])
    lines = found(x)
    notched = sosfiltfilt(notch_sections(lines, FS), x)
    frequency, psd = median_spectrum(notched, FS)
    after = spectral_lines(frequency, psd, 6.0, 1126.4, threshold=1.5)

    assert lines.shape[0] == 2
    assert after.shape[0] == 0 or after[:, 2].max() < 2.0


@pytest.mark.parametrize("row", [[0.0, 0.3, 10.0], [2100.0, 0.3, 10.0],
                                 [60.0, 0.0, 10.0], [60.0, 0.3, 0.5]])
def test_a_line_that_cannot_be_notched_is_refused(row):
    with pytest.raises(ValueError):
        notch_sections(np.array([row]), FS)


def test_no_lines_make_no_sections():
    assert notch_sections(np.zeros((0, 3)), FS).shape == (0, 6)
