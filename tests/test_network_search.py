"""Science time, the fit stretch and the jobs, before any search runs.

A stretch the detector was not observing must never be searched nor counted
as livetime, and the noise model must be fitted where the noise is
stationary: these are read off the data-quality mask and off the periodograms,
and the functions that read them are checked here on arrays whose answer is
known.
"""
import os

import numpy as np
import pytest

from wdf.processes.network_search import (Job, SearchConfig, contamination,
                                          frame_index, mask_segments,
                                          merge_segments, octaves,
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


def test_the_octaves_reach_from_half_the_rate_to_the_low_edge():
    bands = octaves(2048.0, 6.0)
    assert bands[-1] == (512.0, 1024.0)
    assert bands[0][0] == 6.0
    assert all(high == 2.0 * low for low, high in bands[1:])


def test_a_stationary_stretch_scores_one_and_a_transient_lifts_it():
    rng = np.random.default_rng(0)
    rate = 2048.0
    quiet = rng.normal(size=int(300 * rate))
    loud = quiet.copy()
    # A glitch is broadband: an impulse lifts every bin of the periodograms
    # of the segments holding it, and none of the others.
    loud[len(loud) // 2] += 2000.0
    bands = octaves(rate, 16.0)
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


def test_the_worker_is_told_what_the_search_was_configured_with():
    config = SearchConfig(frames={"H1": "x"}, channels={"H1": "H1:STRAIN"},
                          quality={"H1": "H1:DQ"})
    job = Job(ifo="H1", segment=(100.0, 1100.0), frame_list="H1.ffl",
              outdir=os.path.join("out", ""), run="search")
    par = worker_parameters(config, job, 4096.0, fit_offset=300.0)
    assert par.threshold == pytest.approx(2.12)
    assert par.ARorder == par.SqrtWhiteningOrder == par.WhiteningExtraSize == 3000
    assert par.AREstimationOffset == 300.0
    assert (par.window, par.overlap) == (512, 32)
    assert par.segments == [[100.0, 1100.0]]
    assert job.directory("H1:STRAIN").endswith(os.path.join(
        "out", "search", "H1", "H1:STRAIN_100"))
