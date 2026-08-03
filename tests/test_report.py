import os

from wdf.analysis.report import build_run_report
from tests._synth import synth_raw_triggers


class _FakePar:
    def __init__(self, resampling, sigma):
        self.resampling = resampling
        self.sigma = sigma


def _with_clustered_columns(df):
    # build_run_report's default ranking/table columns expect a
    # clustered-events-like shape (gpsMax/snrMax/n_triggers); reuse the
    # raw-trigger synth fixture with renamed columns rather than adding a
    # second synthetic generator. Drop the fixture's own freqMax first --
    # renaming freqPeak->freqMax on top of it would otherwise produce two
    # same-named columns.
    out = df.drop(columns=["freqMax"]).rename(columns={"gpsPeak": "gpsMax", "snrPeak": "snrMax",
                                                         "freqPeak": "freqMax"})
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
    # since it has no snrMax).
    cleaned = {"H1": synth_raw_triggers("H1", n_background=10, gps0=0.0, span_s=50.0, seed=3)}
    path = build_run_report(str(tmp_path), cleaned, event_name="NO_WT")
    html = open(path).read()
    assert "time-frequency" not in html.lower()
