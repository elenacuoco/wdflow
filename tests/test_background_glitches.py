"""The background frames may carry the instrumental transients too.

A background written from stationary noise alone measures the rate at which
noise produces a candidate. A detector's background is not that: it is what the
instrument does. These check that the second kind can be written, that it is
independent of the foreground's, and that asking for it does not change the
foreground.
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
    foreground = generate_dataset(glitchy, n_background_glitch=3, **COMMON)

    written = glitchy / "background_injections.parquet"
    assert written.is_file()
    background = pd.read_parquet(written)
    assert len(background) == 3
    # Only the instrument's transients: a signal in the background would make a
    # candidate found there a signal, which is what the background denies.
    assert set(background.category.astype(str)) == {"glitch"}
    # Placed independently of the foreground's, or the zero-lag comparison
    # would be one realisation against itself.
    assert not set(np.round(background.gps, 3)) & set(
        np.round(foreground[foreground.category.astype(str) == "glitch"].gps, 3))

    # The foreground is the same set of injections either way: the background's
    # transients are written into the background and nowhere else.
    plain_foreground = pd.read_parquet(plain / "injections.parquet")
    pd.testing.assert_frame_equal(plain_foreground, foreground)


def test_the_foreground_does_not_inherit_them(tmp_path):
    """The noise is redrawn rather than held twice, so the foreground frames
    must be bit-for-bit what they are without a glitchy background."""
    plain = tmp_path / "plain"
    glitchy = tmp_path / "glitchy"
    generate_dataset(plain, **COMMON)
    generate_dataset(glitchy, n_background_glitch=3, **COMMON)

    from gwpy.timeseries import TimeSeries

    def samples(where, name):
        ifo = os.path.basename(name).split("-", 1)[0]
        return np.asarray(TimeSeries.read(os.path.join(where, name),
                                          f"{ifo}:MOCK-STRAIN"))

    for name in frames_of(plain, "FOREGROUND"):
        assert np.array_equal(samples(plain, name), samples(glitchy, name))

    # And the background differs from it somewhere, since a transient was
    # written into it and into nothing else.
    assert any(not np.array_equal(samples(plain, name), samples(glitchy, name))
               for name in frames_of(plain, "BACKGROUND"))


def test_asking_for_none_writes_the_stationary_background(tmp_path):
    where = tmp_path / "plain"
    generate_dataset(where, **COMMON)
    assert not (where / "background_injections.parquet").exists()
