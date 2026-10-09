"""The background frames may carry the instrumental transients too.

A background written from stationary noise alone measures the rate at which
noise produces a candidate. A detector's background is not that: it is its
noise and the transients the instrument makes, and its foreground is the same
noise, the same transients and the signals. These check that the background is
written that way and that asking for it does not change the foreground.
"""
import os

import numpy as np
import pandas as pd
import pytest

gwpy = pytest.importorskip("gwpy")

from wdf.mock.dataset import generate_dataset

COMMON = dict(
    duration=64.0,
    start_gps=1400000000.0,
    sample_rate=1024,
    n_cbc=2,
    n_glitch=3,
    n_ccsn=0,
    snr_range=(20.0, 40.0),
    seed=11,
    edge_pad=4.0,
    minimum_injection_gap=2.0,
    frame_length=64.0,
    strict=False,
)


def frames_of(where, kind):
    """Every frame of one kind, by its path relative to the set."""
    found = []
    for root, _, names in os.walk(where):
        for name in names:
            path = os.path.join(root, name)
            if name.endswith(".gwf") and kind in os.path.relpath(path, where):
                found.append(os.path.relpath(path, where))
    return sorted(found)


def test_background_carries_its_own_transients(tmp_path):
    plain = tmp_path / "plain"
    glitchy = tmp_path / "glitchy"
    generate_dataset(plain, **COMMON)
    foreground = generate_dataset(glitchy, background_transients=True, **COMMON)

    written = glitchy / "background_injections.parquet"
    assert written.is_file()
    background = pd.read_parquet(written)
    glitches = foreground[foreground.category.astype(str) == "glitch"]
    assert len(background) == len(glitches)
    # Only the instrument's transients: a signal in the background would make a
    # candidate found there a signal, which is what the background denies.
    assert set(background.category.astype(str)) == {"glitch"}
    # The same transients the foreground carries, in the same detectors at the
    # same times, so the two frame kinds differ by the signals alone.
    assert (sorted(np.round(background.gps, 6))
            == sorted(np.round(glitches.gps, 6)))
    assert (background.detector.astype(str).value_counts().to_dict()
            == glitches.detector.astype(str).value_counts().to_dict())

    # The foreground is the same set of injections either way: the background's
    # transients are written into the background and nowhere else.
    plain_foreground = pd.read_parquet(plain / "injections.parquet")
    pd.testing.assert_frame_equal(plain_foreground, foreground)


def test_the_foreground_does_not_inherit_them(tmp_path):
    """The noise is redrawn rather than held twice, so the foreground frames
    must be what they are without transients in the background.

    The comparison allows a difference of 1e-9 of the frame's peak: the same
    seed does not give the same samples bit for bit in two calls, because
    pycbc's FFTW plans and numpy's vectorised loops depend on the memory
    alignment of the arrays, which changes the last bits (1e-14 of the
    noise). A transient written into the foreground would differ by the
    order of the noise itself."""
    plain = tmp_path / "plain"
    glitchy = tmp_path / "glitchy"
    generate_dataset(plain, **COMMON)
    generate_dataset(glitchy, background_transients=True, **COMMON)

    from gwpy.timeseries import TimeSeries

    def samples(where, name):
        ifo = os.path.basename(name).split("-", 1)[0]
        return np.asarray(TimeSeries.read(os.path.join(where, name),
                                          f"{ifo}:MOCK-STRAIN"))

    for name in frames_of(plain, "FOREGROUND"):
        expected = samples(plain, name)
        np.testing.assert_allclose(samples(glitchy, name), expected, rtol=0,
                                   atol=1e-9 * np.max(np.abs(expected)))

    # And the background differs from it somewhere, since a transient was
    # written into it and into nothing else.
    def apart(name):
        a, b = samples(plain, name), samples(glitchy, name)
        return np.max(np.abs(a - b)) > 1e-3 * np.max(np.abs(a))

    assert any(apart(name) for name in frames_of(plain, "BACKGROUND"))


def test_asking_for_none_writes_the_stationary_background(tmp_path):
    where = tmp_path / "plain"
    generate_dataset(where, **COMMON)
    assert not (where / "background_injections.parquet").exists()
