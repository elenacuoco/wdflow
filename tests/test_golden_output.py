"""Golden-output regression test: the legacy `wdf` package has no test that
pins exact trigger-generation numerics (checked its tests/ directory before
starting this package), so a silent change to AR estimation, whitening, or
the wavelet trigger finder could go unnoticed. This pins wdfUnitDSWorker's
output on a small, fast, fully-synthetic (fixed-seed Gaussian noise) fixture.
"""
import os

import pandas as pd
import pytest

from conftest import FIXTURES_DIR, run_segment_process

GOLDEN_CSV = os.path.join(FIXTURES_DIR, "golden_triggers.csv")
COMPARE_COLUMNS = ["gps", "gpsPeak", "duration", "EnWDF", "snrMean", "snrPeak",
                   "freqMin", "freqMean", "freqMax", "freqPeak", "wave"]


def test_segment_process_matches_golden_output(tmp_outdir):
    golden = pd.read_csv(GOLDEN_CSV)
    result = run_segment_process(tmp_outdir)

    assert len(result) == len(golden), (
        f"trigger count changed: {len(result)} vs golden {len(golden)}. "
        "If this is an intentional algorithm change, regenerate "
        "tests/fixtures/golden_triggers.csv."
    )

    numeric_cols = [c for c in COMPARE_COLUMNS if c != "wave"]
    pd.testing.assert_frame_equal(
        result[numeric_cols].reset_index(drop=True),
        golden[numeric_cols].reset_index(drop=True),
        check_exact=False, rtol=1e-6, atol=1e-30,
    )
    assert (result["wave"].reset_index(drop=True) == golden["wave"].reset_index(drop=True)).all()
