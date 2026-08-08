import glob
import json
import os

import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NOISE_GWF = os.path.join(FIXTURES_DIR, "test_noise.gwf")
GPS0 = 1000000000.0

TEST_PARAMS = dict(
    window=64,
    overlap=16,
    threshold=0.1,
    channel="H1:TEST-STRAIN",
    itf="H1",
    run="offLine",
    len=2.0,
    gps=GPS0,
    ARorder=30,
    learn=20,
    preWhite=2,
    ResamplingFactor=4,
    LowFrequencyCut=10.0,
    FilterOrder=4,
    nproc=1,
)


def run_segment_process(tmp_outdir, whitening_extra_size=None):
    """Runs the real wdfUnitDSWorker.segmentProcess over the small synthetic
    noise fixture and returns the resulting trigger DataFrame. Shared by the
    golden-output regression test and the len-equivalence test.
    """
    from wdf.config.Parameters import Parameters
    from wdf.processes.wdfUnitDSWorker import wdfUnitDSWorker
    from pytsa.tsa import FrameIChannel, SeqView_double_t as SV

    cfg = dict(TEST_PARAMS)
    cfg.update(file=NOISE_GWF, segments=[[GPS0, GPS0 + 90.0]], outdir=tmp_outdir, dir=tmp_outdir,
               ID="golden_test")
    filejson = os.path.join(tmp_outdir, "inputWDF.json")
    with open(filejson, "w") as fh:
        json.dump(cfg, fh)

    par = Parameters()
    par.load(filejson)
    if whitening_extra_size is not None:
        par.WhiteningExtraSize = whitening_extra_size

    strInfo = FrameIChannel(par.file, par.channel, 1.0, par.gps)
    info = SV()
    strInfo.GetData(info)
    par.sampling = int(round(1.0 / info.GetSampling()))
    par.resampling = int(par.sampling / par.ResamplingFactor)
    del strInfo, info

    worker = wdfUnitDSWorker(par)
    for segment in par.segments:
        worker.segmentProcess(segment)

    import pandas as pd
    parquets = glob.glob(os.path.join(par.outdir, par.run, "H1", "*", "*.parquet"))
    assert parquets, "no trigger parquet file produced"
    return pd.read_parquet(parquets[0])


@pytest.fixture
def tmp_outdir(tmp_path):
    outdir = str(tmp_path) + os.sep
    return outdir
