import os

import numpy as np
import pandas as pd

from wdf.analysis.report import build_run_report
from _synth import synth_raw_triggers


class _FakePar:
    def __init__(self, resampling, sigma):
        self.resampling = resampling
        self.sigma = sigma


def _with_clustered_columns(df):
    # build_run_report's default ranking/table columns expect a
    # clustered-events-like shape; the raw-trigger synth fixture already
    # carries gpsPeak/EnWDF/freqMean, so only n_triggers is missing.
    out = df.copy()
    out["n_triggers"] = 1
    return out


def test_build_run_report_writes_self_contained_html(tmp_path):
    cleaned = {
        "H1": synth_raw_triggers("H1", n_background=20, gps0=1000.0, span_s=100.0,
                                  seed=1, burst_gps=1050.0),
        "L1": synth_raw_triggers("L1", n_background=15, gps0=1000.0, span_s=100.0, seed=2),
    }
    clustered = {ifo: _with_clustered_columns(df) for ifo, df in cleaned.items()}

    path = build_run_report(
        str(tmp_path), cleaned, clustered=clustered,
        gps_reference=1050.0, event_name="TEST_EVENT", top_n=3,
    )

    assert path == str(tmp_path / "report.html")
    assert os.path.exists(path)
    html = open(path).read()
    assert "TEST_EVENT" in html
    assert "H1" in html and "L1" in html
    # glitchgram + winning-basis-distribution plot per detector, at minimum
    assert html.count("data:image/png;base64,") >= 4
    assert "<table>" in html
    assert "Winning wavelet basis distribution" in html


def test_build_run_report_skips_tf_plot_without_wt_columns(tmp_path):
    # raw_triggers/par omitted entirely -- must not raise, TF section just
    # doesn't appear. No `clustered` dict given either, so rank_col=None
    # must auto-fall-back to `cleaned`'s own raw-trigger schema (snrPeak,
    # since it has no EnWDF).
    cleaned = {"H1": synth_raw_triggers("H1", n_background=10, gps0=0.0, span_s=50.0, seed=3)}
    path = build_run_report(str(tmp_path), cleaned, event_name="NO_WT")
    html = open(path).read()
    assert "time-frequency" not in html.lower()


def test_build_review_report_writes_html_and_markdown(tmp_path):
    """The review report is the last cell of a review run, so a broken keyword
    in it only surfaces after the whole analysis has been recomputed."""
    from wdf.analysis.review_report import build_review_report

    rng = np.random.default_rng(4)
    n = 60
    recovery = pd.DataFrame({
        "category": np.where(np.arange(n) < 30, "cbc", "glitch"),
        "subclass": np.where(np.arange(n) < 30, "bbh", "blip"),
        "detector": np.where(np.arange(n) % 2 == 0, "H1", "L1"),
        "gps": 1000.0 + np.arange(n),
        "injected_snr": rng.uniform(8, 100, n),
        "found": rng.random(n) < 0.8,
    })
    recovery["recovered_snr"] = recovery["injected_snr"] * rng.uniform(0.3, 1.1, n)
    recovery["dt_s"] = rng.uniform(-0.02, 0.02, n)
    injections = recovery[["category", "subclass", "gps", "injected_snr"]].copy()
    injections["network_snr"] = injections["injected_snr"]

    candidates = pd.DataFrame({
        "gps_candidate": 1000.0 + np.arange(20),
        "network_enwdf": rng.uniform(5, 40, 20),
        "dt_s": rng.uniform(-0.01, 0.01, 20),
    })

    paths = build_review_report(
        str(tmp_path) + os.sep,
        recovery=recovery,
        coincidence_matched=recovery[recovery.category == "cbc"].assign(
            dt_s=rng.uniform(-0.01, 0.01, 30)),
        candidates=candidates,
        background_candidates=candidates,
        injections=injections,
        livetime_days=1.0,
    )

    for kind, path in paths.items():
        assert os.path.exists(path), f"{kind} report not written"
        assert os.path.getsize(path) > 0
