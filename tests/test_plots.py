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
    plot_glitchgram,
    plot_rate,
    plot_trigger_statistics,
    plot_trigger_tiles,
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
