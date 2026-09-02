"""Draw the detector-stage graph the site shows: one detector, one window length.

Run from a checkout with the compiled core available:

    python site/figures/detector_stage_graph.py site/figures/detector_stage_graph.png

The triggers are produced by the pipeline's own transform and metaparameters,
and the edges are the ones `build_detector_graph` admits, so the picture is the
search's output rather than a drawing of it.
"""
import sys

import numpy as np
import pandas as pd

from wdf.analysis.coefficients import from_dense
from wdf.analysis.detector_graph import DetectorGraphConfig, build_detector_graph
from wdf.analysis.metaparameters import meta_features

FS = 4096.0
WINDOW = 256
STEP = 64
SIGMA = 1.0
WAVE = "DaubC12"

TEAL = "#2f6d80"
RUST = "#a8580c"
GREY = "#b9c6cc"


def forward(samples):
    """One window's wavelet coefficients, before thresholding.

    :param samples: the window's samples.
    :return: numpy.ndarray -- the coefficients, in the packet ordering.
    """
    from pytsa.tsa import WaveletTransform

    from wdf.structures.array2SeqView import array2SeqView

    view = array2SeqView(0.0, 1.0, len(samples))
    view = view.Fill(0.0, np.asarray(samples, dtype=float).copy())
    WaveletTransform(len(samples), getattr(WaveletTransform, WAVE)).Forward(view)
    return np.array([view.GetY(0, i) for i in range(len(samples))])


def triggers(signal, gps0=0.0):
    """One trigger per analysis window covering the signal.

    :param signal: the whitened samples to cut into windows.
    :param gps0: time of the first sample.
    :return: pandas.DataFrame -- the trigger schema, one row per window.
    """
    rows = []
    for first in range(0, len(signal) - WINDOW + 1, STEP):
        coefficients = forward(signal[first:first + WINDOW])
        index, value = from_dense(coefficients)
        gps = gps0 + first / FS
        rows.append(dict(
            meta_features(index, value, WINDOW, FS, SIGMA, gps=gps),
            gps=gps, EnWDF=float(np.linalg.norm(coefficients) / SIGMA),
            sigma=SIGMA, wave=WAVE, n_coeff=WINDOW, fs=FS,
            wt_index=index, wt_value=value, ifo="H1"))
    return pd.DataFrame(rows)


def sweep(times, start_hz, end_hz, centre, width, amplitude):
    """A transient whose frequency sweeps across its own duration.

    :param times: the time of each sample.
    :param start_hz: frequency at the start of the sweep.
    :param end_hz: frequency at its end.
    :param centre: time the envelope peaks at.
    :param width: standard deviation of the envelope, seconds.
    :param amplitude: peak amplitude, on the noise scale.
    :return: numpy.ndarray -- the samples.
    """
    span = times - centre
    rate = (end_hz - start_hz) / (6.0 * width)
    phase = 2.0 * np.pi * (start_hz * span + 0.5 * rate * span ** 2)
    return amplitude * np.exp(-0.5 * (span / width) ** 2) * np.sin(phase)


def data():
    """Whitened noise carrying two transients of different band and length.

    :return: tuple -- the sample times and the samples.
    """
    rng = np.random.default_rng(11)
    duration = 1.0
    times = np.arange(int(duration * FS)) / FS
    samples = rng.standard_normal(times.size)
    samples += sweep(times, 90.0, 420.0, centre=0.30, width=0.045, amplitude=9.0)
    samples += sweep(times, 1100.0, 1100.0, centre=0.70, width=0.006, amplitude=11.0)
    return times, samples


def panel(axes, graph, labels, title):
    """Draw one detector graph on `axes`.

    :param axes: the matplotlib axes to draw on.
    :param graph: the DetectorGraph to draw.
    :param labels: connected-component label per node.
    :param title: the panel's title.
    :return: None
    """
    nodes = graph.nodes
    time = nodes["gpsPeak"].to_numpy(dtype=float)
    freq = nodes["freqMean"].to_numpy(dtype=float)
    grouped = np.bincount(labels)[labels] > 1

    for i, j in graph.edges:
        axes.plot([time[i], time[j]], [freq[i], freq[j]],
                  color=RUST, lw=1.0, alpha=0.5, zorder=1)
    axes.scatter(time[~grouped], freq[~grouped], s=28, color=GREY,
                 edgecolor="white", linewidth=0.6, zorder=2)
    axes.scatter(time[grouped], freq[grouped], s=46, color=TEAL,
                 edgecolor="white", linewidth=0.6, zorder=3)

    axes.set_yscale("log", base=2)
    axes.set_xlabel("time [s]")
    axes.set_title(title, loc="left", fontsize=10.5)
    axes.grid(True, which="major", color="#e6ecef", lw=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)


def draw(path):
    """Write the figure to `path`.

    :param path: file to write the PNG to.
    :return: None
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, samples = data()
    table = triggers(samples)

    admissible = build_detector_graph(table)
    significance = np.log1p(table["EnWDF"].to_numpy(dtype=float))
    floor = float(np.quantile(significance, 0.75))
    selected = build_detector_graph(
        table, config=DetectorGraphConfig(minimum_significance=floor))

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=170,
                                         sharey=True, sharex=True)
    panel(left, admissible, admissible.components(),
          "Every admissible edge: the noise percolates")
    panel(right, selected, selected.components(),
          "Once the detector-stage selection is applied: two events")
    left.set_ylabel("frequency [Hz]")
    figure.suptitle("The detector stage: one detector, one window length. "
                    "The connected components are its events.",
                    x=0.008, ha="left", fontsize=12)

    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=TEAL,
                   markeredgecolor="white", markersize=8,
                   label="trigger joined into an event"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=GREY,
                   markeredgecolor="white", markersize=7,
                   label="trigger left on its own"),
        plt.Line2D([], [], color=RUST, lw=1.4,
                   label="edge: continuous in time and in band"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                  fontsize=9, bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout()
    figure.savefig(path, facecolor="white", bbox_inches="tight")
    print(f"{path}: {len(admissible.nodes)} triggers, "
          f"{len(admissible.edges)} admissible edges -> "
          f"{len(set(admissible.components()))} components; "
          f"after selection {len(selected.nodes)} nodes, "
          f"{len(selected.edges)} edges -> "
          f"{len(set(selected.components()))} components")


if __name__ == "__main__":
    draw(sys.argv[1] if len(sys.argv) > 1 else "detector_stage_graph.png")
