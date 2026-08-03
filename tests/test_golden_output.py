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

# WDF2Classify's candidate basis set: the orthonormal GSL family only (Haar +
# every Daubechies/Daubechies-centered order it supports). The biorthogonal
# B-spline family (Bspline*/BsplineC*) is excluded -- see WDF2Classify.cpp's
# GetDataVector for why (it isn't L2-energy-preserving, so a pooled-median
# sigma estimate lets it win basis selection spuriously even on pure noise).
ORTHONORMAL_WAVES = {
    "Haar",
    "Daub4", "Daub6", "Daub8", "Daub10", "Daub12", "Daub14", "Daub16", "Daub18", "Daub20",
    "DaubC4", "DaubC6", "DaubC8", "DaubC10", "DaubC12", "DaubC14", "DaubC16", "DaubC18", "DaubC20",
}


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


def test_no_biorthogonal_wave_wins_on_pure_noise(tmp_outdir):
    """Regression guard for the basis-selection fix: on pure Gaussian noise,
    only orthonormal bases should ever be selected. If this starts failing,
    either a biorthogonal basis was reintroduced to WDF2Classify's candidate
    list, or an unexpected basis name is coming through -- both worth
    catching explicitly rather than only via the golden-value diff above.
    """
    result = run_segment_process(tmp_outdir)
    seen = set(result["wave"].unique())
    assert seen <= ORTHONORMAL_WAVES, f"unexpected/non-orthonormal wave(s): {seen - ORTHONORMAL_WAVES}"
