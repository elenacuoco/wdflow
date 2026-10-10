"""M4 coincidence protection: effective when the detectors run together, and
stated OFF in a single-detector run."""
import glob
import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("py4tsa")
pytest.importorskip("gwpy")

from wdf.config.Parameters import Parameters  # noqa: E402
from wdf.processes.wdfUnitOffline import (SINGLE_DETECTOR_PROTECTION, offline_network,  # noqa: E402
                                          wdfUnitOffline)

FS, GPS0, SECONDS = 4096, 1000000000.0, 100
CHIRP_AT, GLITCH_AT = 60.0, 40.0
DELAY = {"H1": 0.0, "L1": 0.005}


def frames(folder):
    """H1 and L1 frames: white noise, a loud short chirp in both (L1 5 ms
    later), a sine-Gaussian glitch in H1 only."""
    from gwpy.timeseries import TimeSeries

    paths = {}
    for k, ifo in enumerate(("H1", "L1")):
        rng = np.random.default_rng(20 + k)
        t = np.arange(SECONDS * FS) / FS
        x = rng.standard_normal(t.size)
        u = t - CHIRP_AT - DELAY[ifo]
        inside = (u >= 0) & (u < 0.25)
        f = 60.0 + 400.0 * np.clip(u, 0, 0.25) / 0.25
        x += np.where(inside, 25.0 * np.sin(2 * np.pi * np.cumsum(f) / FS)
                      * np.sin(np.pi * np.clip(u, 0, 0.25) / 0.25), 0.0)
        if ifo == "H1":
            g = t - GLITCH_AT
            tau = 8.0 / (2 * np.pi * 250.0) / np.sqrt(2)
            x += 40.0 * np.exp(-g ** 2 / (2 * tau ** 2)) * np.cos(2 * np.pi * 250.0 * g)
        paths[ifo] = os.path.join(folder, f"{ifo}.gwf")
        TimeSeries(x, t0=GPS0, sample_rate=FS, name=f"{ifo}:TEST",
                   channel=f"{ifo}:TEST").write(paths[ifo])
    return paths


def parameters(ifo, path, outdir):
    return Parameters(window=512, overlap=128, threshold=0.1, channel=f"{ifo}:TEST", itf=ifo,
                      run="m4", len=2.0, gps=GPS0, ARorder=30, learn=20, preWhite=2,
                      ResamplingFactor=2, LowFrequencyCut=12.0, sampling=FS, file=path,
                      outdir=outdir, ZeroPhaseFilter="truncated", ZeroPhaseDuration=1.0,
                      BandPassFilter="butterworth", GlitchSubtraction=True,
                      AREstimationOffset=5.0, SearchName="m4")


def table(worker, name):
    path = glob.glob(f"{worker.par.dir}{name}-Searchm4.csv")[0]
    return pd.read_csv(path) if os.path.getsize(path) > 1 else pd.DataFrame()


def near(frame, column, at, within=0.5):
    if frame.empty:
        return frame
    return frame[(frame[column] - GPS0 - at).abs() < within]


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    return frames(str(tmp_path_factory.mktemp("frames")))


def test_together_the_coincident_chirp_is_kept_and_the_glitch_subtracted(paths, tmp_path):
    out = offline_network({i: parameters(i, p, str(tmp_path) + "/") for i, p in paths.items()},
                          (GPS0, GPS0 + SECONDS))
    for ifo, w in out["workers"].items():
        assert w.par.GlitchProtection == "on"
        assert w.par.GlitchProtectionDetectors == ["L1" if ifo == "H1" else "H1"]
        seeds, atoms = table(w, "GlitchSeeds"), table(w, "GlitchAtoms")
        chirp = near(seeds, "centre", CHIRP_AT)
        assert len(chirp) and chirp["protected"].all()
        assert near(atoms, "centre", CHIRP_AT).empty
    h1 = out["workers"]["H1"]
    glitch = near(table(h1, "GlitchSeeds"), "centre", GLITCH_AT)
    assert len(glitch) and not glitch["protected"].any()
    assert len(near(table(h1, "GlitchAtoms"), "centre", GLITCH_AT)) > 0


def test_alone_the_protection_is_off_and_said_so(paths, tmp_path):
    worker = wdfUnitOffline(parameters("H1", paths["H1"], str(tmp_path) + "/"))
    worker.segmentProcess((GPS0, GPS0 + SECONDS))
    assert worker.par.GlitchProtection == SINGLE_DETECTOR_PROTECTION
    with open(glob.glob(f"{worker.par.dir}parametersUsed-*Searchm4.json")[0]) as fh:
        used = json.load(fh)
    assert used["GlitchProtection"].startswith("OFF")
    assert len(near(table(worker, "GlitchAtoms"), "centre", CHIRP_AT)) > 0
    assert len(near(table(worker, "GlitchAtoms"), "centre", GLITCH_AT)) > 0
