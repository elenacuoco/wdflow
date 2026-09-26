"""Science time, the fit stretch, the check and the jobs, before any search runs.

A stretch the detector was not observing must never be searched nor counted
as livetime, the noise model must be fitted where the noise is stationary, and
no segment is searched unless every segment of the stretch has passed its
check: these are read off the data-quality mask, off the periodograms and off
the check's reports, and the functions that read them are checked here on
arrays whose answer is known.
"""
import os

import numpy as np
import pytest

from wdf.processes.gating import octave_bands
from wdf.processes.network_search import (CONDITIONING_KEYS, Job, SearchConfig,
                                          StretchRejected, _signature,
                                          contamination, frame_index,
                                          mask_segments, merge_segments,
                                          worker_parameters, write_frame_list)


def test_science_is_where_every_named_bit_is_set():
    mask = [0, 1, 3, 3, 2, 1, 1, 0, 1]
    assert mask_segments(mask, 100.0, 1.0, bits=1) == [
        (101.0, 104.0), (105.0, 107.0), (108.0, 109.0)]
    assert mask_segments(mask, 100.0, 1.0, bits=3) == [(102.0, 104.0)]
    assert mask_segments([0, 0], 0.0, 1.0, bits=1) == []


def test_stretches_that_touch_are_one():
    assert merge_segments([(5.0, 8.0), (0.0, 2.0), (2.0, 4.0), (7.0, 9.0)]) == [
        (0.0, 4.0), (5.0, 9.0)]


def test_a_stationary_stretch_scores_one_and_a_transient_lifts_it():
    rng = np.random.default_rng(0)
    rate = 2048.0
    quiet = rng.normal(size=int(300 * rate))
    loud = quiet.copy()
    # A glitch is broadband: an impulse lifts every bin of the periodograms
    # of the segments holding it, and none of the others.
    loud[len(loud) // 2] += 2000.0
    bands = octave_bands(rate, 16.0)
    assert np.max(contamination(quiet, rate, bands)) < 1.2
    assert np.max(contamination(loud, rate, bands)) > 3.0


def test_the_frames_are_indexed_from_their_names(tmp_path):
    for gps in (1000004096, 1000000000):
        open(tmp_path / f"H-H1_TEST-{gps}-4096.gwf", "w").close()
    open(tmp_path / "manifest.json", "w").close()
    index = frame_index(str(tmp_path))
    assert [gps for _, gps, _ in index] == [1000000000, 1000004096]
    path = write_frame_list(index, 1000004000.0, 1000005000.0,
                            str(tmp_path / "list.ffl"))
    lines = open(path).read().splitlines()
    assert len(lines) == 2 and lines[0].endswith("1000000000 4096 0 0")
    with pytest.raises(ValueError):
        write_frame_list(index, 2e9, 2e9 + 1, str(tmp_path / "none.ffl"))


def _config(**changes):
    return SearchConfig(frames={"H1": "x", "L1": "y"},
                        channels={"H1": "H1:STRAIN", "L1": "L1:STRAIN"},
                        quality={"H1": "H1:DQ", "L1": "L1:DQ"}, **changes)


def test_the_worker_is_told_what_the_search_was_configured_with():
    config = _config(search_low_frequency={"L1": 32.0}, threshold=5.0)
    job = Job(ifo="H1", segment=(100.0, 1100.0), frame_list="H1.ffl",
              outdir=os.path.join("out", ""), run="block")
    par = worker_parameters(config, job, 4096.0, fit_offset=300.0)
    assert par.threshold == pytest.approx(5.0)
    assert par.ARorder == par.SqrtWhiteningOrder == par.WhiteningExtraSize == 3000
    assert par.AREstimationOffset == 300.0
    assert (par.window, par.overlap) == (512, 32)
    assert par.segments == [[100.0, 1100.0]]
    assert (par.LineThreshold, par.GateThreshold, par.GateTaper) == (5.0, 50.0, 0.25)
    # The detector's own low frequency reaches the detector it names only.
    assert not hasattr(par, "SearchLowFrequency")
    other = worker_parameters(config, Job("L1", (100.0, 1100.0), "L1.ffl",
                                          os.path.join("out", ""), "block"),
                              4096.0, fit_offset=300.0)
    assert other.SearchLowFrequency == 32.0
    assert job.directory("H1:STRAIN").endswith(os.path.join(
        "out", "block", "H1", "H1:STRAIN_100"))


def test_a_check_is_read_back_only_under_its_own_conditioning():
    """Every entry the conditioning depends on is in the signature, so a check
    made under another low frequency or another fit stretch is not taken for
    this one."""
    job = Job(ifo="L1", segment=(100.0, 1100.0), frame_list="L1.ffl",
              outdir=os.path.join("out", ""), run="conditioning")
    first = worker_parameters(_config(search_low_frequency={"L1": 32.0}), job,
                              4096.0, fit_offset=300.0)
    same = worker_parameters(_config(search_low_frequency={"L1": 32.0}), job,
                             4096.0, fit_offset=300.0)
    lower = worker_parameters(_config(search_low_frequency={"L1": 16.0}), job,
                              4096.0, fit_offset=300.0)
    moved = worker_parameters(_config(search_low_frequency={"L1": 32.0}), job,
                              4096.0, fit_offset=600.0)
    signature = _signature(first, CONDITIONING_KEYS)
    assert signature == _signature(same, CONDITIONING_KEYS)
    assert signature != _signature(lower, CONDITIONING_KEYS)
    assert signature != _signature(moved, CONDITIONING_KEYS)


def test_a_rejected_stretch_names_every_detector_band_and_criterion():
    class Failed:
        def __init__(self, text):
            self.text = text

        def message(self):
            return self.text

    error = StretchRejected([Failed("L1 fails: 16-32 Hz: window kurtosis"),
                             Failed("V1 fails: transients and gates cover")],
                            table="the whole check")
    assert "L1 fails: 16-32 Hz" in str(error) and "V1 fails" in str(error)
    assert len(error.reports) == 2
    assert error.table == "the whole check"
