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


# ------------------------------------------------------------------ the lines

def _line_strain(frequency):
    """Unit noise at the frames' rate with one line, for `_raw_stretch`."""
    def raw(self, start, seconds):
        t = np.arange(int(seconds * SAMPLING)) / SAMPLING
        return (np.random.default_rng(int(start)).standard_normal(t.size)
                + (0.2 * np.sin(2 * np.pi * frequency * t) if frequency else 0.0))
    return raw


def test_each_segment_is_notched_at_the_lines_of_its_own_fit_stretch(monkeypatch):
    """The lines are found per segment and not carried over to the next."""
    w = worker(learn=20, AREstimationOffset=50.0)
    monkeypatch.setattr(wdfUnitDSWorker, "_raw_stretch", _line_strain(331.3))
    lines = w._segment_lines(GPS0, GPS0 + 90.0)
    assert len(lines) == 1 and abs(lines[0][0] - 331.3) <= 1.0 / 16

    w.par.LineNotches = lines
    monkeypatch.setattr(wdfUnitDSWorker, "_raw_stretch", _line_strain(None))
    assert w._segment_lines(GPS0 + 100.0, GPS0 + 190.0) == []


def test_lines_named_in_the_configuration_are_used_as_they_are(monkeypatch):
    def unread(*_):
        raise AssertionError("the strain was searched for lines the configuration names")

    monkeypatch.setattr(wdfUnitDSWorker, "_raw_stretch", unread)
    named = [[60.0, 0.3, 40.0], [120.0, 0.3, 12.0]]
    assert worker(LineNotches=named)._segment_lines(GPS0, GPS0 + 90.0) == named
    assert worker(LineThreshold=0)._segment_lines(GPS0, GPS0 + 90.0) == []


# ------------------------------------------------------------------ the gates

def test_a_declared_gate_takes_its_stretch_out_of_the_search(tmp_path):
    """The gates are applied to the whitened blocks the search reads: no
    window inside a gate reaches the threshold, while the same stretch of the
    ungated run fires as often as the noise does anywhere."""
    from conftest import run_segment_process

    def inside(gates):
        # A gate this long covers more of the short fixture than the check
        # allows; what is tested here is how the gate is applied.
        df = run_segment_process(str(tmp_path / str(len(gates))) + "/",
                                 changes=dict(Gates=gates, ValidateConditioning=False))
        return int(((df.gps > GPS0 + 40.1) & (df.gps < GPS0 + 41.9)).sum())

    assert inside([]) > 20
    assert inside([[GPS0 + 40.0, GPS0 + 42.0]]) == 0


def test_the_gates_are_the_declared_ones_and_the_loud_transients():
    w = worker(Gates=[[GPS0 + 10.0, GPS0 + 11.0]])
    ds = BandPassDownSampling(w.par)
    rate = w.par.resampling
    x = np.random.default_rng(3).standard_normal(int(60 * rate))
    t = np.arange(x.size) / rate - 30.0
    x += 500.0 * np.exp(-(t / 0.05) ** 2) * np.sin(2 * np.pi * 200.0 * t)

    from wdf.processes.gating import octave_bands, transients
    found = transients(x, rate, octave_bands(rate, ds.search_low_frequency))
    gates = w._gates(GPS0, found)
    assert gates.shape == (2, 2)
    assert gates[0].tolist() == [GPS0 + 10.0, GPS0 + 11.0]
    assert GPS0 + 29.8 < gates[1, 0] < GPS0 + 30.0 < gates[1, 1] < GPS0 + 30.2

    w.par.GateThreshold = 0
    assert w._gates(GPS0, found).tolist() == [[GPS0 + 10.0, GPS0 + 11.0]]
    assert w._gates(GPS0, None).tolist() == [[GPS0 + 10.0, GPS0 + 11.0]]


# ------------------------------------------------------------------ the check

def test_a_segment_that_fails_the_check_is_not_searched(tmp_path):
    """The check runs before the first search; a failure stops the segment
    with the detector, the band and the criterion, writes no trigger and does
    not mark the segment done, and the report is kept beside it."""
    import glob
    import json
    from conftest import run_segment_process
    from wdf.processes.validation import ConditioningRejected

    outdir = str(tmp_path) + "/"
    with pytest.raises(ConditioningRejected, match="H1 fails .*transients and gates cover"):
        run_segment_process(outdir, changes=dict(Gates=[[GPS0 + 30.0, GPS0 + 32.5]]))

    segment = glob.glob(outdir + "offLine/H1/*/")[0]
    assert not glob.glob(segment + "*.parquet")
    assert not glob.glob(segment + "ProcessEnded.check")
    with open(segment + "conditioning-check.json") as handle:
        assert json.load(handle)["passed"] is False


def test_a_segment_can_be_checked_without_being_searched(tmp_path):
    import glob
    w = worker(outdir=str(tmp_path) + "/")
    report = w.validate((GPS0, GPS0 + 90.0))

    assert report.passed, report.message()
    assert report.bands[0][0] == BandPassDownSampling(w.par).search_low_frequency
    assert report.range_mpc > 0.0
    assert glob.glob(str(tmp_path) + "/offLine/H1/*/conditioning-check.json")
    assert not glob.glob(str(tmp_path) + "/offLine/H1/*/*.parquet")


def test_a_report_reads_back_as_it_was_written(tmp_path):
    from wdf.processes.validation import ValidationReport
    report = worker(outdir=str(tmp_path) + "/").validate((GPS0, GPS0 + 90.0))
    again = ValidationReport.from_dict(report.to_dict())
    assert again.to_dict() == report.to_dict()
    assert again.passed == report.passed


# ------------------------------------------------------ a stretch of the stream

def test_a_stretch_of_the_stream_is_the_stream_the_segment_search_reads(tmp_path):
    """Both filters settle before the stretch starts, so once they have the
    stretch is the whole segment's stream at the same instants: the band-pass
    runs with real data on both sides of every block and the whitening is a
    finite filter with a look-ahead of its own order. The gates are applied
    and the scale divided out as the check applies and divides them."""
    from wdf.processes.gating import gate_weights

    segment = (GPS0, GPS0 + 90.0)
    gates = np.array([[GPS0 + 44.0, GPS0 + 44.5]])
    whole = worker(outdir=str(tmp_path) + "/", learn=20, AREstimationOffset=50.0)
    _, directory = whole._segment_directory(GPS0)
    whole.par.LineNotches = whole._segment_lines(*segment)
    build = whole._noise_model(*segment, directory)
    t0, stream, _ = whole._whitened_segment(*segment, build)
    rate = whole.par.resampling
    times = t0 + np.arange(stream.size) / rate
    expected = stream * gate_weights(times, gates, 0.25) / whole.par.sigma

    part = worker(outdir=str(tmp_path) + "/", learn=20, AREstimationOffset=50.0)
    start, samples = part.whitened_stretch(segment, GPS0 + 40.0, GPS0 + 48.0, gates)
    first = int(round((start - t0) * rate))
    same = expected[first:first + samples.size]

    assert start == pytest.approx(GPS0 + 40.0, abs=1.0 / rate)
    assert samples.size == pytest.approx(8.0 * rate, abs=2)
    # The band-pass agrees to 1e-10 of the noise whatever the grid; the
    # whitening's lattice arithmetic leaves differences of order 1e-6 of the
    # noise at the joins of either grid, which is rounding and not a start.
    assert np.max(np.abs(samples - same)) / np.std(same) < 1e-5
    with pytest.raises(ValueError, match="not inside"):
        part.whitened_stretch(segment, GPS0 - 5.0, GPS0 + 5.0)


def test_a_later_start_is_checked_as_the_shorter_segment_would_be(tmp_path):
    """Checking the tail of the segment's stream from a later start's warm-up
    is checking the shorter segment: it keeps the fit stretch, so its lines
    and model, and both filters forget where they were started."""
    whole = worker(outdir=str(tmp_path / "whole") + "/", learn=20, AREstimationOffset=50.0)
    tried = whole.validate_starts((GPS0, GPS0 + 90.0), [GPS0 + 10.0, GPS0 + 20.0],
                                  stop_at_first=False)
    assert [start for start, _ in tried] == [GPS0 + 10.0, GPS0 + 20.0]
    for later, report in tried:
        shorter = worker(outdir=str(tmp_path / str(int(later - GPS0))) + "/", learn=20,
                         AREstimationOffset=GPS0 + 50.0 - later)
        direct = shorter.validate((later, GPS0 + 90.0))
        assert report.start == pytest.approx(direct.start)
        assert report.stop == pytest.approx(direct.stop)
        for name in ("power", "kurtosis", "gaussian", "thirds"):
            assert np.allclose(getattr(report, name), getattr(direct, name), rtol=1e-5,
                               atol=1e-6), name
        assert report.transient_fraction == pytest.approx(direct.transient_fraction,
                                                          abs=1e-9)
        assert report.passed == direct.passed

    first = whole.validate_starts((GPS0, GPS0 + 90.0), [GPS0 + 10.0, GPS0 + 20.0])
    assert len(first) == (1 if tried[0][1].passed else 2)
    with pytest.raises(ValueError, match="outside the segment"):
        whole.validate_starts((GPS0, GPS0 + 90.0), [GPS0 + 55.0])


def test_every_window_is_the_search_at_the_same_positions(tmp_path):
    """The windows read again between two instants sit on the search's own
    grid and carry the statistic, the basis and the coefficients the search
    computed there; the search wrote those above its threshold, and every one
    of them is among them."""
    import glob
    import json
    from conftest import run_segment_process

    outdir = str(tmp_path) + "/"
    written = run_segment_process(outdir)
    with open(glob.glob(outdir + "offLine/H1/*/conditioning-check.json")[0]) as handle:
        gates = json.load(handle)["gates"]
    again = worker(outdir=outdir, segments=[[GPS0, GPS0 + 90.0]])
    start, stop = GPS0 + 30.0, GPS0 + 34.0
    every = again.every_window((GPS0, GPS0 + 90.0), start, stop, gates)
    span = 64 / (SAMPLING / FACTOR)
    inside = written[(written.gps + span >= start) & (written.gps < stop)]

    assert len(inside) > 100 and len(every) >= len(inside)
    matched = inside.merge(every, on="gps", suffixes=("", "_again"))
    assert len(matched) == len(inside)
    assert np.allclose(matched.EnWDF, matched.EnWDF_again, rtol=1e-5)
    assert (matched.wave == matched.wave_again).mean() > 0.99
    assert np.allclose(np.diff(np.sort(every.gps)), every.stride.iloc[0], atol=1e-9)
