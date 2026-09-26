"""What the worker fits the noise model on, and what the search starts on.

The model is fitted on a stretch conditioned with real data on both sides, and
the search starts only once both filters have settled on real data, whatever
the configuration asks for.
"""
import numpy as np
import pytest

pytest.importorskip("py4tsa")

from conftest import GPS0, NOISE_GWF, TEST_PARAMS
from wdf.config.Parameters import Parameters
from wdf.processes.BandPassDownSampling import (BandPassDownSampling,
                                                SV_to_array)
from wdf.processes.wdfUnitDSWorker import wdfUnitDSWorker
from wdf.processes.zero_phase_whitening import ZeroPhaseWhitening

SAMPLING, FACTOR = 16384, 4


def worker(**changes):
    """A worker on the synthetic frame, with `changes` to the test run."""
    par = Parameters()
    par.__dict__.update(TEST_PARAMS, file=NOISE_GWF, sampling=SAMPLING,
                        resampling=SAMPLING // FACTOR)
    par.__dict__.update(changes)
    return wdfUnitDSWorker(par)


def read(gps, seconds):
    from py4tsa.tsa import FrameIChannel, SeqView_double_t as SV
    view = SV()
    FrameIChannel(NOISE_GWF, TEST_PARAMS["channel"], seconds, gps).GetData(view)
    return view


# ------------------------------------------------------------ the fit stretch

def test_the_fit_stretch_is_where_the_offset_puts_it():
    w = worker(learn=20, AREstimationOffset=50.0)
    assert w._fit_start(GPS0, GPS0 + 90.0) == GPS0 + 50.0
    assert w._fit_start(GPS0, GPS0 + 90.0, context_s=5.0) == GPS0 + 50.0


def test_the_fit_stretch_moves_inward_for_its_context():
    """The offset is kept unless the context would fall outside the segment,
    and then the stretch moves just far enough."""
    w = worker(learn=20, AREstimationOffset=0.0)
    assert w._fit_start(GPS0, GPS0 + 90.0, context_s=5.0) == GPS0 + 5.0
    short = worker(learn=20, AREstimationOffset=50.0)
    assert short._fit_start(GPS0, GPS0 + 40.0) == GPS0 + 20.0
    assert short._fit_start(GPS0, GPS0 + 40.0, context_s=5.0) == GPS0 + 15.0


def test_a_segment_too_short_for_the_stretch_and_its_context_is_refused():
    with pytest.raises(ValueError, match="shorter than"):
        worker(learn=20)._fit_start(GPS0, GPS0 + 25.0, context_s=5.0)


def test_the_fit_stretch_is_conditioned_as_the_stream_is():
    """Filtered alone, the stretch's edges would carry the filter's start
    and the model would be fitted on it as though it were the noise."""
    w = worker(learn=20, AREstimationOffset=50.0)
    stretch = w._learn_stretch(GPS0, GPS0 + 90.0)
    ds = BandPassDownSampling(w.par, estimation=True)
    context = 10 * SAMPLING
    wide = ds.condition_stretch(read(GPS0 + 40.0, 40.0), context)
    kept, expected = SV_to_array(stretch), SV_to_array(wide)

    assert stretch.GetStart() == pytest.approx(GPS0 + 50.0)
    assert w.par.AREstimationStart == GPS0 + 50.0
    assert np.max(np.abs(kept - expected)) / np.std(expected) < 1e-9


# --------------------------------------------------------------- the warm-up

def test_the_search_starts_once_both_filters_have_settled():
    """A band-pass that rings for longer than `preWhite` seconds lengthens the
    warm-up instead of handing its start to the search: the first searched
    block starts after the conditioning's settling and the whitening's order."""
    w = worker(LowFrequencyCut=2.0, FilterOrder=10, preWhite=2)
    w.par.SqrtWhiteningOrder = 256
    ds = BandPassDownSampling(w.par)
    ar = np.zeros(31)
    ar[0] = 1.0
    whitening = ZeroPhaseWhitening(ar, int(w.par.resampling), 0, order=256)
    needed = ds.padlen / SAMPLING + whitening.latency / w.par.resampling

    streaming = w._prime(GPS0, GPS0 + 90.0, ds, whitening)
    first = next(w._whitened(streaming, GPS0 + 90.0, ds, whitening))

    assert needed > 2
    assert w.par.preWhite == int(np.ceil(needed))
    assert first.GetStart() == pytest.approx(GPS0 + w.par.preWhite)


# ------------------------------------------------------------ the saved model

@pytest.mark.parametrize("change", [
    dict(LowFrequencyCut=12.0), dict(FilterOrder=6), dict(AREstimationOffset=40.0),
    dict(learn=16), dict(ARorder=20)], ids=lambda c: next(iter(c)))
def test_a_saved_model_is_found_only_under_its_own_conditioning(tmp_path, monkeypatch, change):
    """The model depends on everything that shapes the samples it is fitted
    on. A model saved under another conditioning, reloaded silently, whitens
    the stream with the spectrum of other data."""
    import h5py
    import json
    directory = str(tmp_path) + "/"
    first = worker(learn=20, AREstimationOffset=50.0)
    first._noise_model(GPS0, GPS0 + 90.0, directory)
    with h5py.File(first.par.ARfile, "r") as fh:
        stored = json.loads(fh.attrs["conditioning"])
    assert stored["start"] == GPS0 + 50.0

    other = worker(**dict(dict(learn=20, AREstimationOffset=50.0), **change))
    other._noise_model(GPS0, GPS0 + 90.0, directory)
    assert other.par.ARfile != first.par.ARfile

    def refit(*_):
        raise AssertionError("a model saved under the same conditioning was fitted again")

    monkeypatch.setattr(wdfUnitDSWorker, "_learn_stretch", refit)
    same = worker(learn=20, AREstimationOffset=50.0)
    same._noise_model(GPS0, GPS0 + 90.0, directory)
    assert same.par.ARfile == first.par.ARfile
    assert same.par.sigma == first.par.sigma
    assert same.par.AREstimationStart == GPS0 + 50.0
