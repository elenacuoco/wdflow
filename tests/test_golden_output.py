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
COMPARE_COLUMNS = ["gps", "gpsPeak", "duration", "EnWDF", "sigma", "snrMean",
                   "snrPeak", "freqMin", "freqMean", "freqMax", "freqPeak", "wave"]

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
