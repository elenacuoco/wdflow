"""What an offline segment records: the span actually searched, and the
values set during the run, in parametersUsed."""
import glob
import json
import os

import pytest

pytest.importorskip("py4tsa")

from conftest import GPS0, run_segment_process  # noqa: E402

from wdf.processes.wdfUnitOffline import wdfUnitOffline  # noqa: E402


def used(folder):
    path = glob.glob(os.path.join(folder, "**", "parametersUsed-*.json"), recursive=True)[0]
    with open(path) as fh:
        return json.load(fh)


def test_the_searched_span_and_the_tail_are_recorded(tmp_outdir):
    triggers = run_segment_process(tmp_outdir)
    par = used(tmp_outdir)
    end = GPS0 + 90.0
    assert par["gpsEndSegment"] == end
    assert par["gpsEndSearched"] <= par["gpsEnd"] < end
    assert par["UnsearchedTail_s"] == pytest.approx(end - par["gpsEndSearched"])
    parts = par["ReadAheadParts_s"]
    assert set(parts) == {"blocks", "bandpass", "whitening", "whitening_lag"}
    assert par["UnsearchedTail_s"] <= sum(parts.values()) + 1.0 / par["resampling"]
    window_s = par["window"] / par["resampling"]
    assert (triggers["gps"] + window_s).max() <= par["gpsEndSearched"] + 1e-6


def test_values_set_during_the_run_reach_parameters_used(tmp_outdir):
    run_segment_process(tmp_outdir, worker_class=wdfUnitOffline, Ridge=True, RidgeWindow=512,
                        RidgeOverlap=256, RidgeBackground=16.0, RidgeBackgroundMinimum=4.0,
                        ZeroPhaseFilter="truncated", ZeroPhaseDuration=1.0,
                        BandPassFilter="butterworth", LowFrequencyCut=12.0)
    par = used(tmp_outdir)
    for key in ("RidgeDriftUsed", "RidgeXDropUsed", "RidgeCandidates", "LowPassOrder",
                "AliasAttenuationReached", "gpsEndSearched"):
        assert key in par, key
    assert par["window"] == 64
