"""The submission format a mock data challenge fixes, and what it must refuse."""
import os

import numpy as np
import pandas as pd
import pytest

from wdf.analysis.submission import (
    MISSING, SECONDS_PER_YEAR, SUBMISSION_COLUMNS, false_alarm_rate_hz,
    resolvable_threshold, submission_name, submission_triggers,
    write_submission,
)


def _candidates(n=50, seed=0):
    rng = np.random.default_rng(seed)
    start = 1400000000.0 + np.arange(n) * 100.0
    return pd.DataFrame(dict(
        gpsStart=start, duration=rng.uniform(0.1, 1.0, n),
        gps_candidate=start + 0.5, statistic=rng.exponential(2.0, n) + 5.0,
        freqMin=rng.uniform(20.0, 60.0, n), freqMax=rng.uniform(200.0, 900.0, n),
    ))


def test_a_rate_the_background_cannot_reach_is_refused():
    """Three years of slides cannot state one per ten years: 0.3 accidentals.

    Quoting the rate anyway would put a number in a submission that nothing
    measured, which is the one thing a challenge's own threshold must not
    invite.
    """
    three_years = 3.17 * SECONDS_PER_YEAR
    with pytest.raises(ValueError, match="reaches no such rate"):
        resolvable_threshold(1.0 / (10.0 * SECONDS_PER_YEAR), three_years)

    # Ten years of background reach it, and the check passes.
    assert resolvable_threshold(1.0 / (10.0 * SECONDS_PER_YEAR),
                                12.0 * SECONDS_PER_YEAR) > 1.0


def test_a_rate_that_is_not_a_rate_is_refused_as_one():
    """A zero or negative threshold fails the check the loudest way possible.

    The refusal has to reach the caller as the refusal the API documents, not
    as whatever the arithmetic of describing it happens to raise.
    """
    for far_hz in (0.0, -1.0):
        with pytest.raises(ValueError, match="positive"):
            resolvable_threshold(far_hz, SECONDS_PER_YEAR)
    with pytest.raises(ValueError, match="positive"):
        resolvable_threshold(1.0 / SECONDS_PER_YEAR, 0.0)


def test_a_missing_ranking_statistic_is_refused_as_the_api_says():
    candidates = _candidates()
    with pytest.raises(ValueError, match="rank on"):
        submission_triggers(candidates, np.arange(100.0), 1e6,
                            "no_such_column", 1e-4)


def test_the_rate_is_counted_and_never_extrapolated():
    background = np.arange(100.0)
    livetime = 1000.0
    # Above everything: one such event over the whole livetime, not zero.
    louder = false_alarm_rate_hz([200.0], background, livetime)[0]
    assert louder == pytest.approx(1.0 / livetime)
    # Half way up: half the background stands above it.
    middle = false_alarm_rate_hz([50.0], background, livetime)[0]
    assert middle == pytest.approx(50.0 / livetime)


def test_every_column_is_present_and_absence_is_the_agreed_value():
    """A blank would shift every later column, and zero is a real declination."""
    candidates = _candidates()
    background = np.random.default_rng(1).exponential(2.0, 10000) + 5.0
    table = submission_triggers(
        candidates, background, livetime_s=1e6, statistic="statistic",
        far_threshold_hz=1e-2,
        columns=dict(gps_start="gpsStart", gps_peak="gps_candidate",
                     frequency_start="freqMin", frequency_end="freqMax"))

    assert list(table.columns) == SUBMISSION_COLUMNS
    assert table.notna().all().all()
    # Nothing provides a sky position here, so both columns carry the value the
    # challenge reserves for it.
    assert (table.right_ascension == MISSING).all()
    assert (table.declination == MISSING).all()
    assert (table.far_hz <= 1e-2).all()
    assert table.far_hz.is_monotonic_increasing


def test_the_file_names_follow_the_challenge_and_refuse_what_it_forbids():
    assert submission_name("triggers", "o4b-2", "short-0", "wdflow",
                           "26-08-12") == \
        "triggers_o4b-2_short-0_wdflow_26-08-12.csv"
    # The underscore separates the fields of the name, so a field may not hold
    # one; upper case is forbidden outright.
    for bad in ("wdf_flow", "WDFlow"):
        with pytest.raises(ValueError, match="lower-case"):
            submission_name("triggers", "o4b-2", "short-0", bad, "26-08-12")
    with pytest.raises(ValueError, match="YY-MM-DD"):
        submission_name("triggers", "o4b-2", "short-0", "wdflow", "2026-08-12")


def test_a_trigger_outside_every_segment_is_refused(tmp_path):
    """The two files must describe the same data."""
    triggers = pd.DataFrame({c: [1.0] for c in SUBMISSION_COLUMNS})
    triggers["gps_start"] = 1400000000.0
    triggers["gps_end"] = 1400000001.0
    with pytest.raises(ValueError, match="outside every analysed segment"):
        write_submission(str(tmp_path), triggers,
                         [(1400001000.0, 1400002000.0)],
                         mdc="o4b-2", dataset="short-0", pipeline="wdflow",
                         date="26-08-12")


def test_both_files_are_written_headerless_and_paired(tmp_path):
    candidates = _candidates(20)
    background = np.random.default_rng(2).exponential(2.0, 5000) + 5.0
    table = submission_triggers(
        candidates, background, livetime_s=1e6, statistic="statistic",
        far_threshold_hz=1e-2,
        columns=dict(gps_start="gpsStart", gps_peak="gps_candidate"))
    table["gps_end"] = table.gps_start + 1.0

    segments = [(1399999999.0, 1400010000.0)]
    paths = write_submission(str(tmp_path), table, segments, mdc="o4b-2",
                             dataset="short-0", pipeline="wdflow",
                             date="26-08-12")

    for kind in ("triggers", "segments"):
        assert os.path.exists(paths[kind])
    # Paired by everything but the first field, and read back as bare numbers.
    a = os.path.basename(paths["triggers"]).split("_", 1)[1]
    b = os.path.basename(paths["segments"]).split("_", 1)[1]
    assert a == b
    written = pd.read_csv(paths["triggers"], header=None)
    assert written.shape == (len(table), len(SUBMISSION_COLUMNS))
    assert pd.read_csv(paths["segments"], header=None).shape == (1, 2)
