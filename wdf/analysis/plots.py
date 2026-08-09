"""Drawing what a run produced, on axes the caller owns.

Every function here takes an Axes and returns it, so the same call serves a
notebook cell and the HTML report. Which column carries the frequency, or the
statistic, is an argument with a default rather than a choice built into the
plot: `freqPeak` and `freqMean` answer different questions and are meant to be
compared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def plot_trigger_tiles(ax, triggers, fs, t0=0.0, sigma=None, cmap="viridis"):
    """Draw the time-frequency tiles of the coefficients a trigger set kept.

    :type ax: matplotlib.axes.Axes
    :param ax: axes to draw on.
    :type triggers: pandas.DataFrame
    :param triggers: triggers carrying `gps` and the coefficient columns.
    :type fs: float
    :param fs: sampling frequency the coefficients were computed at, Hz.
    :type t0: float
    :param t0: GPS time the horizontal axis is measured from.
    :type sigma: float or None
    :param sigma: noise scale the colour is expressed in; the triggers' own
        median if None.
    :type cmap: str
    :param cmap: colormap for the tile amplitude.
    :return: matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    from wdf.analysis.clustering import collect_significant_pixels

    pixels = collect_significant_pixels(triggers, fs)
    if pixels.empty:
        return ax

    if sigma is None:
        scale = pd.to_numeric(pixels["sigma"], errors="coerce")
        scale = scale[np.isfinite(scale) & (scale > 0)]
        sigma = float(scale.median()) if len(scale) else 1.0

    amplitude = np.sqrt(pixels["energy"].to_numpy()) / sigma
    colours = plt.get_cmap(cmap)(np.clip(amplitude / max(amplitude.max(), 1e-30), 0, 1))
    for tile, colour in zip(pixels.itertuples(), colours):
        ax.add_patch(plt.Rectangle(
            (tile.t_lo - t0, tile.f_lo),
            tile.t_hi - tile.t_lo, tile.f_hi - tile.f_lo,
            color=colour, alpha=0.85, linewidth=0))

    ax.set_xlim(pixels["t_lo"].min() - t0, pixels["t_hi"].max() - t0)
    ax.set_ylim(max(pixels["f_lo"].min(), fs / 2 ** 20), fs / 2)
    ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz]")
    return ax


def plot_glitchgram(ax, triggers, statistic="EnWDF", frequency="freqMean",
                    t0=None, cmap="viridis", colorbar=True):
    """Time against frequency, coloured by the statistic.

    The frequency defaults to `freqMean`, the spectral moment over every
    surviving tile. `freqPeak` is time-localised, and for the median trigger
    only one coefficient overlaps its peak tile in time, so it can only return
    that tile's own frequency: plotted, it draws the octave ladder as horizontal
    lines rather than the distribution of the data. `freqPeak` remains the
    sharper answer on a transient that sweeps, and stays available here.

    :type ax: matplotlib.axes.Axes
    :param ax: axes to draw on.
    :type triggers: pandas.DataFrame
    :param triggers: triggers or events to draw.
    :type statistic: str
    :param statistic: column the colour is taken from.
    :type frequency: str
    :param frequency: column the vertical axis is taken from.
    :type t0: float or None
    :param t0: GPS time the horizontal axis is measured from; the earliest
        trigger if None.
    :type cmap: str
    :param cmap: colormap for the statistic.
    :type colorbar: bool
    :param colorbar: draw a colorbar beside the axes.
    :return: matplotlib.axes.Axes
    """
    from matplotlib.colors import LogNorm

    time = pd.to_numeric(triggers["gpsCentroid" if "gpsCentroid" in triggers
                                  else "gpsPeak"], errors="coerce").to_numpy()
    t0 = float(np.nanmin(time)) if t0 is None else float(t0)
    value = pd.to_numeric(triggers[statistic], errors="coerce").to_numpy()
    positive = value[np.isfinite(value) & (value > 0)]

    drawn = ax.scatter(
        time - t0, pd.to_numeric(triggers[frequency], errors="coerce").to_numpy(),
        c=value, cmap=cmap, s=10, linewidths=0, alpha=0.7,
        norm=LogNorm(vmin=positive.min(), vmax=positive.max()) if positive.size else None)
    ax.set_yscale("log")
    ax.set_xlabel(f"time - {t0:.1f} [s]")
    ax.set_ylabel(f"{frequency} [Hz]")
    if colorbar:
        ax.figure.colorbar(drawn, ax=ax, label=statistic)
    return ax


def plot_rate(ax, triggers, bin_s=60.0, t0=None):
    """Trigger rate against time, which is the first thing that says whether a
    stretch of data is usable.

    :type ax: matplotlib.axes.Axes
    :param ax: axes to draw on.
    :type triggers: pandas.DataFrame
    :param triggers: triggers to count.
    :type bin_s: float
    :param bin_s: width of a time bin, seconds.
    :type t0: float or None
    :param t0: GPS time the horizontal axis is measured from.
    :return: matplotlib.axes.Axes
    """
    time = pd.to_numeric(triggers["gps"], errors="coerce").to_numpy()
    time = time[np.isfinite(time)]
    if not time.size:
        return ax
    t0 = float(time.min()) if t0 is None else float(t0)

    edges = np.arange(time.min(), time.max() + bin_s, bin_s)
    counts, _ = np.histogram(time, bins=edges)
    ax.step(edges[:-1] - t0, counts / bin_s, where="post")
    ax.set_xlabel(f"time - {t0:.1f} [s]")
    ax.set_ylabel("trigger rate [Hz]")
    return ax


def plot_trigger_statistics(fig, triggers, fs=None):
    """A panel per question the run summary answers.

    :type fig: matplotlib.figure.Figure
    :param fig: figure to fill; any existing axes are left alone.
    :type triggers: pandas.DataFrame
    :param triggers: one detector's triggers.
    :type fs: float or None
    :param fs: sampling frequency; the triggers' own if None.
    :return: matplotlib.figure.Figure
    """
    from wdf.analysis.coefficients import window_length
    from wdf.analysis.wavelets import coeff_levels

    axes = fig.subplots(2, 2)
    n_nonzero = np.array([len(v) for v in triggers["wt_index"]])

    plot_rate(axes[0, 0], triggers)
    axes[0, 0].set_title("trigger rate")

    axes[0, 1].hist(n_nonzero, bins=np.arange(0, n_nonzero.max() + 2) - 0.5)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("surviving coefficients per trigger")
    axes[0, 1].set_ylabel("triggers")
    axes[0, 1].set_title("how sparse a trigger is")

    counts = triggers["wave"].value_counts().sort_values()
    axes[1, 0].barh(counts.index.astype(str), counts.to_numpy())
    axes[1, 0].set_xlabel("triggers")
    axes[1, 0].set_title("winning basis")

    level, _ = coeff_levels(window_length(triggers))
    index = np.concatenate([np.asarray(v, dtype=np.int64)
                            for v in triggers["wt_index"]]) if len(triggers) else np.empty(0, int)
    occupancy = np.bincount(level[index] + 1, minlength=int(level.max()) + 2)
    axes[1, 1].bar(np.arange(len(occupancy)) - 1, occupancy)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("octave level (-1 is the scaling coefficient)")
    axes[1, 1].set_ylabel("coefficients")
    axes[1, 1].set_title("where the energy sits on the dyadic ladder")

    fig.tight_layout()
    return fig
