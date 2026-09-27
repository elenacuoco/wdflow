"""The stretch the noise model is fitted on is conditioned with real context."""
import json
import os

import numpy as np
import pytest
from conftest import GPS0, NOISE_GWF, TEST_PARAMS

from wdf.config.Parameters import Parameters
from wdf.processes.BandPassDownSampling import (
    BandPassDownSampling,
    SV_to_array,
    discard_edges,
)
from wdf.structures.array2SeqView import array2SeqView

SAMPLING = 4096


def front_end_parameters():
    return Parameters(sampling=SAMPLING, ResamplingFactor=2, LowFrequencyCut=6.0,
                      FilterOrder=10, resampling=SAMPLING // 2)


def conditioned(x, first_s, last_s):
    """`x` between two times, band-passed and decimated in one shot."""
    part = x[int(first_s * SAMPLING):int(last_s * SAMPLING)]
    view = array2SeqView(first_s, SAMPLING, part.size).Fill(first_s, part)
    return BandPassDownSampling(front_end_parameters(), estimation=True).Process(view)


def strain_like(seconds, seed=0):
    """White noise under a component far below the band, as strain is under
    its microseism: the band-pass's edge transient is driven by what it cuts."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLING)) / SAMPLING
    return rng.standard_normal(t.size) + 1e3 * np.sin(2 * np.pi * 0.3 * t + 0.4)


def test_the_edges_dropped_leave_the_middle_at_its_own_time():
    x = np.arange(1000.0)
    view = array2SeqView(10.0, 100.0, x.size).Fill(10.0, x)
    kept = discard_edges(view, 2.0)
    assert kept.GetStart() == pytest.approx(12.0)
    assert np.array_equal(SV_to_array(kept), x[200:800])
    with pytest.raises(ValueError):
        discard_edges(view, 5.0)


def test_context_removes_the_edge_transient():
    """A stretch conditioned with real data around it is the stream itself.

    Reference: the same times conditioned inside a much longer stretch. With
    real context read and dropped, the fitted samples equal it to the filter's
    settling floor; conditioned alone, the stretch starts and ends on the
    filter's transient from an assumed boundary.
    """
    x = strain_like(300.0)
    first, last, context = 100.0, 160.0, 20.0
    rate = SAMPLING // 2
    reference = SV_to_array(conditioned(x, 0.0, 300.0))[int(first * rate):int(last * rate)]
    with_context = SV_to_array(discard_edges(
        conditioned(x, first - context, last + context), context))
    alone = SV_to_array(conditioned(x, first, last))

    scale = np.std(reference)
    edges = np.r_[0:rate, reference.size - rate:reference.size]
    assert np.abs(with_context - reference).max() < 1e-6 * scale
    assert np.abs(alone[edges] - reference[edges]).max() > 1e-2 * scale


def worker(tmp_path, **overrides):
    from py4tsa.tsa import FrameIChannel
    from py4tsa.tsa import SeqView_double_t as SV

    from wdf.processes.wdfUnitDSWorker import wdfUnitDSWorker

    cfg = dict(TEST_PARAMS)
    cfg.update(file=NOISE_GWF, segments=[[GPS0, GPS0 + 90.0]], outdir=str(tmp_path) + os.sep)
    cfg.update(overrides)
    path = tmp_path / "inputWDF.json"
    path.write_text(json.dumps(cfg))
    par = Parameters()
    par.load(str(path))
    reader, info = FrameIChannel(par.file, par.channel, 1.0, par.gps), SV()
    reader.GetData(info)
    par.sampling = round(1.0 / info.GetSampling())
    par.resampling = int(par.sampling / par.ResamplingFactor)
    return wdfUnitDSWorker(par)


def test_the_worker_fits_on_the_stretch_it_was_asked_for(tmp_path):
    """`learn` seconds from `AREstimationOffset`, labelled with their own time,
    and equal to those times conditioned inside the whole frame."""
    from py4tsa.tsa import FrameIChannel
    from py4tsa.tsa import SeqView_double_t as SV

    job = worker(tmp_path, AREstimationOffset=50.0, ARFitContext=20.0)
    stretch = job._learn_stretch(GPS0, GPS0 + 90.0)
    rate = job.par.resampling
    assert stretch.GetStart() == pytest.approx(GPS0 + 50.0)
    assert stretch.GetSize() == int(job.learn * rate)
    assert job.par.ARFitContext == 20.0

    whole, raw = FrameIChannel(job.par.file, job.par.channel, 100.0, GPS0), SV()
    whole.GetData(raw)
    reference = SV_to_array(BandPassDownSampling(job.par, estimation=True).Process(raw))
    reference = reference[int(50 * rate):int(50 * rate) + stretch.GetSize()]
    fitted = SV_to_array(stretch)
    assert np.abs(fitted - reference).max() < 1e-6 * np.std(reference)


def test_a_context_shorter_than_the_settling_is_refused(tmp_path):
    job = worker(tmp_path, ARFitContext=0.01)
    with pytest.raises(ValueError, match="settling"):
        job._learn_stretch(GPS0, GPS0 + 90.0)
