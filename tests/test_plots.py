"""Drawing a run, on axes the caller owns."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _synth import synth_raw_triggers
from wdf.analysis.coefficients import trigger_statistics
from wdf.analysis.plots import (
    plot_cluster_membership,
    plot_event_span,
    plot_glitchgram,
    plot_rate,
    plot_trigger_statistics,
    plot_trigger_tiles,
    plot_windows_per_event,
)

FS = 2048.0


@pytest.fixture
def triggers():
    return synth_raw_triggers("H1", n_background=60, gps0=1000.0, span_s=300.0,
                              seed=0, burst_gps=1150.0)


def test_the_glitchgram_draws_a_point_per_trigger(triggers):
    fig, ax = plt.subplots()
    plot_glitchgram(ax, triggers, colorbar=False)
    assert sum(len(c.get_offsets()) for c in ax.collections) == len(triggers)
    plt.close(fig)


def test_the_frequency_and_the_statistic_are_choices_not_fixtures(triggers):
    """freqPeak and freqMean answer different questions and get compared."""
    fig, ax = plt.subplots()
    plot_glitchgram(ax, triggers, frequency="freqMean", statistic="snrPeak",
                    colorbar=False)
    drawn = np.asarray(ax.collections[0].get_offsets())[:, 1]
    assert drawn == pytest.approx(triggers["freqMean"].to_numpy())
    plt.close(fig)


def test_the_rate_integrates_back_to_the_trigger_count(triggers):
    fig, ax = plt.subplots()
    plot_rate(ax, triggers, bin_s=30.0)
    heights = ax.lines[0].get_ydata()
    assert heights.sum() * 30.0 == pytest.approx(len(triggers))
    plt.close(fig)


def test_the_tiles_drawn_are_the_coefficients_kept(triggers):
    fig, ax = plt.subplots()
    subset = triggers.head(10)
    plot_trigger_tiles(ax, subset, FS, t0=float(subset["gps"].iloc[0]))
    expected = sum(len(v) for v in subset["wt_index"])
    assert len(ax.patches) == expected
    plt.close(fig)


def test_an_empty_trigger_set_draws_nothing_and_does_not_raise():
    fig, ax = plt.subplots()
    empty = pd.DataFrame(columns=["gps", "wt_index", "wt_value", "n_coeff", "wave", "sigma"])
    plot_trigger_tiles(ax, empty, FS)
    assert not ax.patches
    plt.close(fig)


def test_the_statistics_panel_fills_every_axis(triggers):
    fig = plt.figure(figsize=(11, 7))
    plot_trigger_statistics(fig, triggers)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_the_statistics_report_the_sparsity_that_justifies_the_format(triggers):
    summary = trigger_statistics(triggers)
    n_nonzero = np.array([len(v) for v in triggers["wt_index"]])

    assert summary["n_triggers"] == len(triggers)
    assert summary["n_nonzero_mean"] == pytest.approx(n_nonzero.mean())
    assert summary["density"] == pytest.approx(n_nonzero.mean() / triggers["n_coeff"].iloc[0])
    assert summary["n_coeff"] == [int(triggers["n_coeff"].iloc[0])]
    assert sum(summary["wave_counts"].values()) == len(triggers)


def test_a_run_with_no_triggers_still_reports():
    assert trigger_statistics(pd.DataFrame()) == {"n_triggers": 0}


def _chained_events():
    """A run of consecutive windows, clustered, plus the matcher's output."""
    from types import SimpleNamespace

    from _synth import triggers_from_signal
    from wdf.analysis.injections import match_injections
    from wdf.analysis.robust_events import ClusterConfig, cluster_detector_triggers

    fs, window, overlap = 2048.0, 512, 128
    step = window - overlap
    rng = np.random.default_rng(0)
    signal = np.hanning(window + 6 * step) * np.sin(
        2 * np.pi * 200.0 * np.arange(window + 6 * step) / fs)
    signal += 1e-3 * rng.standard_normal(signal.size)

    triggers = triggers_from_signal(signal, fs, window, overlap, gps0=1000.0)
    parameters = SimpleNamespace(window=window, overlap=overlap, resampling=fs)
    labeled, events = cluster_detector_triggers(
        triggers, parameters, config=ClusterConfig(max_missing_windows=0))
    injections = pd.DataFrame([{"gps": 1000.3, "duration": 0.8}])
    matched = match_injections(events, injections, window_s=0.5,
                               candidate_time="gpsCentroid")
    return labeled, events, matched, fs


def test_cluster_membership_fills_the_member_tiles_and_outlines_the_rest():
    labeled, events, _, fs = _chained_events()
    cluster_id = int(events.iloc[0].cluster_id)
    members = labeled[labeled.cluster_id == cluster_id]

    fig, ax = plt.subplots()
    plot_cluster_membership(ax, labeled, events, cluster_id, fs)
    filled = [p for p in ax.patches if p.get_facecolor()[3] > 0]
    assert len(filled) == sum(len(v) for v in members.wt_index)
    plt.close(fig)


def test_windows_per_event_draws_one_point_per_event():
    _, events, _, _ = _chained_events()
    fig, ax = plt.subplots()
    plot_windows_per_event(ax, events)
    drawn = np.asarray(ax.collections[0].get_offsets())
    assert len(drawn) == len(events)
    assert (drawn[:, 1] == events.n_triggers.to_numpy()).all()
    plt.close(fig)


def test_the_event_span_bar_covers_the_injection_it_matched():
    """Which is the property that makes the span the right thing to match on."""
    _, events, matched, _ = _chained_events()
    assert bool(matched.found.iloc[0])

    fig, ax = plt.subplots()
    plot_event_span(ax, events, matched)
    bar = [line for line in ax.lines if len(line.get_xdata()) == 2][0]
    x0, x1 = bar.get_xdata()
    assert x0 <= 0.0 <= x1
    plt.close(fig)
