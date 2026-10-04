"""The offline worker: wdfUnitDSWorker plus M4 and the segment's ridge."""
import glob
import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("py4tsa")

from conftest import GPS0, NOISE_GWF, TEST_PARAMS, run_segment_process

from wdf.processes.wdfUnitOffline import offline_network, wdfUnitOffline

TRUNCATED = dict(ZeroPhaseFilter="truncated", ZeroPhaseDuration=1.0, BandPassFilter="butterworth",
                 LowFrequencyCut=12.0)


def _sub(tmp_outdir, name):
    path = os.path.join(tmp_outdir, name) + os.sep
    os.makedirs(path, exist_ok=True)
    return path


@pytest.mark.parametrize("extra", [{}, TRUNCATED])
def test_without_m4_it_is_the_ds_worker(tmp_outdir, extra):
    """M4 off and the same parameters: the same triggers as wdfUnitDSWorker,
    with or without the ridge, which reads the stream and changes nothing."""
    old = run_segment_process(_sub(tmp_outdir, "ds"), **extra)
    new = run_segment_process(_sub(tmp_outdir, "off"), worker_class=wdfUnitOffline, **extra)
    ridge = run_segment_process(_sub(tmp_outdir, "ridge"), worker_class=wdfUnitOffline, Ridge=True,
                                RidgeWindow=512, RidgeOverlap=256, RidgeBackground=16.0,
                                RidgeBackgroundMinimum=4.0, **extra)
    for other in (new, ridge):
        assert len(other) == len(old)
        pd.testing.assert_frame_equal(other.drop(columns=["wt_index", "wt_value"]),
                                      old.drop(columns=["wt_index", "wt_value"]))
    assert glob.glob(os.path.join(tmp_outdir, "ridge", "**", "RidgeCandidates.csv"), recursive=True)
    assert glob.glob(os.path.join(tmp_outdir, "ridge", "**", "RidgeCells.npz"), recursive=True)


def test_m4_runs_and_records_its_settings(tmp_outdir):
    triggers = run_segment_process(tmp_outdir, worker_class=wdfUnitOffline, GlitchSubtraction=True,
                                   SearchName="offline", **TRUNCATED)
    folder = glob.glob(os.path.join(tmp_outdir, "*", "H1", "*"))[0]
    names = os.listdir(folder)
    assert "GlitchAtoms-Searchoffline.csv" in names and "GlitchCensus-Searchoffline.csv" in names
    with open(os.path.join(folder, "parametersUsed-Win64-Searchoffline.json")) as fh:
        used = json.load(fh)
    assert used["GlitchSubtraction"] is True and used["GlitchProtectCoincident"] is True
    assert used["GlitchProtection"].startswith("OFF") and used["GlitchProtectionDetectors"] == []
    assert used["GlitchQs"] == [1.0, 2.0, 4.0, 8.0, 16.0] and used["GlitchCensusLow"] == 12.0
    assert len(triggers) > 0


def test_m4_needs_zero_phase_taps(tmp_outdir):
    with pytest.raises(ValueError, match="taps|truncated"):
        run_segment_process(tmp_outdir, worker_class=wdfUnitOffline, GlitchSubtraction=True,
                            ZeroPhaseFilter="causal")


def test_the_offline_worker_refuses_the_causal_band_pass():
    from wdf.config.Parameters import Parameters
    par = Parameters(**dict(TEST_PARAMS, sampling=16384, BandPassPhase="causal"))
    with pytest.raises(ValueError, match="zero-phase"):
        wdfUnitOffline(par)


def test_the_network_protects_and_combines(tmp_outdir):
    """Two detectors on the same noise: each gets the other's census, and the
    network ridge is the minimum of their z maps."""
    from wdf.config.Parameters import Parameters

    pars = {}
    for ifo in ("H1", "L1"):
        cfg = dict(TEST_PARAMS, **TRUNCATED, file=NOISE_GWF, outdir=tmp_outdir, itf=ifo,
                   sampling=16384, GlitchSubtraction=True, Ridge=True, SearchName="offline",
                   RidgeBackground=16.0, RidgeBackgroundMinimum=4.0, RidgeWindow=512,
                   RidgeOverlap=256)
        pars[ifo] = Parameters(**cfg)
        pars[ifo].resampling = 16384 / TEST_PARAMS["ResamplingFactor"]
    out = offline_network(pars, (GPS0, GPS0 + 90.0))
    for ifo, w in out["workers"].items():
        other = "L1" if ifo == "H1" else "H1"
        assert set(w.par.GlitchCoincidenceSeeds) == {other}
    assert out["network"] is not None
    assert glob.glob(os.path.join(tmp_outdir, "**", "network", "**",
                                  "NetworkRidgeCandidates-Searchoffline.csv"), recursive=True)
