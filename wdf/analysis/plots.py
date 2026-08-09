"""Drawing what a run produced, on axes the caller owns.

Every function here takes an Axes and returns it, so the same call serves a
notebook cell and the HTML report. Which column carries the frequency, or the
statistic, is an argument with a default rather than a choice built into the
plot: a caller comparing two statistics should not have to edit a plot.
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
                    t0=None, cmap="viridis", colorbar=True, band=False):
    """Time against frequency, coloured by the statistic.

    The frequency is `freqMean`, the energy-weighted moment over every surviving
    tile. It is continuous only where a trigger's coefficients span more than one
    octave: within an octave every tile has the same band, so the moment collapses
    onto that band's centre and a fifth of triggers land on a handful of values.
    That is the transform's resolution and not an artefact of the estimator --
    a sub-octave number would be resolution the data does not carry.

    `band` draws each trigger's own `[freqMin, freqMax]` behind the point, which
    is what the trigger actually determines, so the octave-limited ones read as
    bands rather than as lines.

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
    :type band: bool
    :param band: draw each trigger's `[freqMin, freqMax]` extent behind its point.
    :return: matplotlib.axes.Axes
    """
    from matplotlib.colors import LogNorm

    time = pd.to_numeric(triggers["gpsCentroid" if "gpsCentroid" in triggers
                                  else "gpsPeak"], errors="coerce").to_numpy()
    t0 = float(np.nanmin(time)) if t0 is None else float(t0)
    value = pd.to_numeric(triggers[statistic], errors="coerce").to_numpy()
    positive = value[np.isfinite(value) & (value > 0)]

    if band and {"freqMin", "freqMax"} <= set(triggers.columns):
        low = pd.to_numeric(triggers["freqMin"], errors="coerce").to_numpy()
        high = pd.to_numeric(triggers["freqMax"], errors="coerce").to_numpy()
        ax.vlines(time - t0, low, high, color="0.6", lw=0.4, alpha=0.35, zorder=1)

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


def plot_cluster_membership(ax, labeled, events, cluster_id, fs, cluster_column="cluster_id"):
    """One event's windows on the wavegram, with everything around them.

    Member tiles are drawn filled, the tiles of the windows on either side in
    outline. A chain broken through the middle of a signal shows as a
    continuous track that stops being filled part way along, which is what
    reading counts off a table cannot show.

    :type ax: matplotlib.axes.Axes
    :param ax: axes to draw on.
    :type labeled: pandas.DataFrame
    :param labeled: triggers carrying a cluster label and the coefficients.
    :type events: pandas.DataFrame
    :param events: the event catalogue.
    :type cluster_id: int
    :param cluster_id: which event to draw.
    :type fs: float
    :param fs: sampling frequency the coefficients were computed at, Hz.
    :type cluster_column: str
    :param cluster_column: the label column in `labeled`.
    :return: matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    from wdf.analysis.clustering import collect_significant_pixels

    event = events.loc[events[cluster_column] == cluster_id].iloc[0]
    members = labeled[labeled[cluster_column] == cluster_id]
    span = float(event["duration"])
    lo = float(event["gpsStart"]) - max(span, 1.0)
    hi = float(event["gpsStart"]) + span + max(span, 1.0)

    around = labeled[(labeled["gps"] >= lo) & (labeled["gps"] <= hi)]
    t0 = float(event["gpsStart"])

    for tile in collect_significant_pixels(around, fs).itertuples():
        ax.add_patch(plt.Rectangle(
            (tile.t_lo - t0, tile.f_lo), tile.t_hi - tile.t_lo, tile.f_hi - tile.f_lo,
            facecolor="none", edgecolor="0.6", lw=0.4))
    for tile in collect_significant_pixels(members, fs).itertuples():
        ax.add_patch(plt.Rectangle(
            (tile.t_lo - t0, tile.f_lo), tile.t_hi - tile.t_lo, tile.f_hi - tile.f_lo,
            facecolor="tab:blue", alpha=0.7, lw=0))

    ax.set_xlim(lo - t0, hi - t0)
    ax.set_ylim(max(fs / 2 ** 20, 1.0), fs / 2)
    ax.set_yscale("log")
    ax.set_xlabel(f"time - {t0:.3f} [s]")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title(f"event {int(cluster_id)}: {int(event['n_triggers'])} windows, "
                 f"{span * 1e3:.0f} ms")
    return ax


def plot_windows_per_event(ax, events, injections=None, duration_column="duration"):
    """How many windows an event spans, against how long the signal was.

    A chaining rule that works gives a rising trend. One that breaks gives a
    ceiling: events stop growing at whatever length the rule can still follow,
    however long the signal is.

    :type ax: matplotlib.axes.Axes
    :param ax: axes to draw on.
    :type events: pandas.DataFrame
    :param events: the event catalogue, carrying `n_triggers`.
    :type injections: pandas.DataFrame or None
    :param injections: matched injections, to take the true duration from; the
        events' own measured duration is used when None.
    :type duration_column: str
    :param duration_column: column holding the signal duration, seconds.
    :return: matplotlib.axes.Axes
    """
    source = events if injections is None else injections
    duration = pd.to_numeric(source[duration_column], errors="coerce").to_numpy(float)
    windows = pd.to_numeric(events["n_triggers"], errors="coerce").to_numpy(float)
    keep = np.isfinite(duration) & np.isfinite(windows) & (duration > 0)

    ax.scatter(duration[keep], windows[keep], s=8, alpha=0.4, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("signal duration [s]")
    ax.set_ylabel("analysis windows in the event")
    return ax


def plot_event_span(ax, events, injections, injection_time="gps",
                    candidate_column="candidate_index"):
    """Where each recovered event's extent sits against the injection it matched.

    The bar is the event's `[gpsStart, gpsStart+duration]`, the marker the
    injection. An event that covers its injection is doing its job even when its
    own reported time is seconds away, which is the ordinary case for a chirp.

    :type ax: matplotlib.axes.Axes
    :param ax: axes to draw on.
    :type events: pandas.DataFrame
    :param events: the event catalogue, indexed as the matcher recorded it.
    :type injections: pandas.DataFrame
    :param injections: output of `wdf.analysis.injections.match_injections`.
    :type injection_time: str
    :param injection_time: column of `injections` holding the injection time.
    :type candidate_column: str
    :param candidate_column: column of `injections` holding the matched event.
    :return: matplotlib.axes.Axes
    """
    found = injections[injections["found"]]
    for row, (_, injection) in enumerate(found.iterrows()):
        event = events.loc[injection[candidate_column]]
        t_inj = float(injection[injection_time])
        start = float(event["gpsStart"]) - t_inj
        ax.plot([start, start + float(event["duration"])], [row, row],
                lw=1.5, color="tab:blue", solid_capstyle="butt")
    ax.axvline(0.0, color="crimson", ls="--", lw=1, label="injection")
    ax.set_xlabel("time from the injection [s]")
    ax.set_ylabel("recovered injection")
    ax.legend(fontsize=8)
    return ax
