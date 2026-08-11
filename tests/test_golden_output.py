"""Golden-output regression test: the legacy `wdf` package has no test that
pins exact trigger-generation numerics (checked its tests/ directory before
starting this package), so a silent change to AR estimation, whitening, or
the wavelet trigger finder could go unnoticed. This pins wdfUnitDSWorker's
output on a small, fast, fully-synthetic (fixed-seed Gaussian noise) fixture.
"""
import os

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES_DIR, run_segment_process

GOLDEN = os.path.join(FIXTURES_DIR, "golden_triggers.parquet")

# What the search itself produces, straight from p4TSA. A change here is a
# change to AR estimation, whitening or the trigger finder.
SEARCH_COLUMNS = ["gps", "EnWDF", "sigma"]

# What wdflow derives from the coefficients. A change here is a change to the
# metaparameter estimators.
DERIVED_COLUMNS = ["gpsStart", "gpsCentroid", "tSpread", "gpsPeak", "duration",
                   "snrPeak", "freqMin", "freqMean", "freqMax"]

# WDF2Classify's candidate basis set (2026-08-03, trimmed to 10 -- see
# WDF2Classify.cpp's kCandidateBases for the full rationale): Haar,
# Daubechies-centered at every other order (4/8/12/16/20 -- plain Daub and
# the skipped centered orders are near-duplicates of these, dropped),
# Symlet 4/8, Coiflet 1/2. All orthonormal. The biorthogonal B-spline family
# (Bspline*/BsplineC*) and a plain (non-wavelet-packet) DCT both remain
# excluded -- see WDF2Classify.cpp for why.
ORTHONORMAL_WAVES = {
    "Haar",
    "DaubC4", "DaubC8", "DaubC12", "DaubC16", "DaubC20",
    "Sym4", "Sym8",
    "Coif1", "Coif2",
}


def test_segment_process_matches_golden_output(tmp_outdir):
    golden = pd.read_parquet(GOLDEN)
    result = run_segment_process(tmp_outdir)

    assert len(result) == len(golden), (
        f"trigger count changed: {len(result)} vs golden {len(golden)}. "
        "If this is an intentional algorithm change, regenerate "
        "tests/fixtures/golden_triggers.parquet."
    )

    numeric = SEARCH_COLUMNS + DERIVED_COLUMNS
    pd.testing.assert_frame_equal(
        result[numeric].reset_index(drop=True),
        golden[numeric].reset_index(drop=True),
        check_exact=False, rtol=1e-6, atol=1e-30,
    )
    assert (result["wave"].reset_index(drop=True) == golden["wave"].reset_index(drop=True)).all()


def test_the_golden_coefficients_are_reproduced(tmp_outdir):
    """The coefficients are the record, to the precision they are stored in.

    Which coefficients survive is exact and is checked as such: an index is an
    integer, and a different one means a different tile, which is a change of
    behaviour however small the amplitude that caused it.

    Their values are checked to one unit in the last place of `float32`, the
    type they are stored as. The transform runs in double precision and is
    rounded once on the way to disk, so two builds that contract a multiply-add
    differently, or sum a dot product in a different order, disagree in the
    final bit of that rounding while computing the same quantity. Demanding bit
    equality there would make this a test of the compiler rather than of the
    pipeline; a real change moves many coefficients by far more than a bit.
    """
    golden = pd.read_parquet(GOLDEN)
    result = run_segment_process(tmp_outdir)

    for row, (index, value) in enumerate(zip(golden.wt_index, golden.wt_value)):
        assert np.array_equal(np.asarray(result.wt_index.iloc[row]), np.asarray(index))
        np.testing.assert_allclose(
            np.asarray(result.wt_value.iloc[row], dtype=np.float64),
            np.asarray(value, dtype=np.float64),
            rtol=float(np.finfo(np.float32).eps), atol=0.0,
            err_msg=f"coefficients of trigger {row} differ by more than a "
                    f"float32 rounding")


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


def test_no_end_of_segment_zero_padding_artifact(tmp_outdir):
    """Regression guard for the end-of-segment margin bug (found 2026-08-03 on
    real GW170817 data): with only one par.len of safety margin, the main
    detection loop in wdfUnitDSWorker.segmentProcess could issue a final read
    reaching a full par.len past the segment's true end. FrameIChannel didn't
    raise for that -- it silently returned the read with its tail
    zero-padded, which BandPassDownSampling/DWhitening's forward-backward,
    lookahead-dependent filtering turned into a large, near-constant
    whitened artifact, which WDF2Classify then flagged as a spurious
    astronomically-high-EnWDF trigger (this pure-noise fixture's own old
    golden output had one, up to this fix). On pure Gaussian noise, EnWDF
    should never be more than a handful of noise-sigma above threshold --
    catching this directly (rather than only via the exact-value golden
    diff above) so a future regression doesn't get silently re-pinned into
    a new golden fixture.
    """
    result = run_segment_process(tmp_outdir)
    assert result["EnWDF"].max() < 50.0, (
        f"EnWDF max={result['EnWDF'].max():.3g} on pure noise -- suspiciously high, "
        "likely the end-of-segment zero-padding artifact (see wdfUnitDSWorker's "
        "self.par.gpsEnd margin)"
    )
