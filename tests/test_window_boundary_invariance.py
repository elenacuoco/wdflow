"""The window grid is a computational detail, not a property of the transient.

An analysis window starts where the search happened to begin, so a transient
falls at a different place inside the grid depending only on where the run
started. The quantities describing the transient must not follow that choice:
grouping exists so that the event, and not the window, is the physical object.

The transform is not shift invariant, so the coefficients themselves do change
with the grid, and the test asks for stability rather than equality. The
tolerances below are the size of the variation the grouping is allowed to leave,
expressed relative to the quantity itself or to the window it was measured in.
"""
import numpy as np
import pytest

from wdf.analysis.detector_graph import (
    DetectorGraphConfig, build_detector_graph, detector_events,
)

from _synth import triggers_from_signal

FS = 2048.0
WINDOW = 512
OVERLAP = 256
STEP = WINDOW - OVERLAP
GPS0 = 1000.0

# How much of itself each quantity may vary across the grid offsets; a time is
# read against one window instead, being absolute. The tolerances differ by
# quantity because the quantities do: the statistic is measured on the stitched
# reconstruction, which covers the same samples however they were cut, while the
# extent and the band are read off the tiles that survived, and a tile is
# quantised --- one octave in frequency, one tile width in time --- so a grid
# offset that moves a marginal coefficient across the threshold moves them by a
# whole step. That is a property of the tiling, not a failure of the grouping.
TOLERANCE = dict(gpsCentroid=0.10, duration=0.50, freqMean=0.25, EnWDF=0.05)


def sine_gaussian(fs, duration_s, f0, q, amplitude=20.0):
    """A narrowband burst, long enough to cross a window boundary.

    :param fs: sampling frequency, Hz.
    :param duration_s: length of the returned series, seconds.
    :param f0: central frequency, Hz.
    :param q: quality factor, setting the envelope width as `q / (2 pi f0)`.
    :param amplitude: peak amplitude, on the noise scale of the triggers.
    :return: numpy.ndarray -- the burst, centred in the series.
    """
    n = int(round(duration_s * fs))
    t = (np.arange(n) - 0.5 * n) / fs
    tau = q / (2.0 * np.pi * f0)
    return amplitude * np.exp(-0.5 * (t / tau) ** 2) * np.sin(2.0 * np.pi * f0 * t)


def loudest_event(signal, offset, config):
    """The event a grid shifted by `offset` samples builds from `signal`.

    The grid moves and the data do not: dropping the first `offset` samples and
    advancing the GPS time of the first sample by as much leaves every remaining
    sample at the time it already had, so the transient sits at one absolute
    time throughout and only the window edges move around it.

    :param signal: the samples the windows are cut from.
    :param offset: samples to shift the start of the grid by.
    :param config: the level-one rule.
    :return: pandas.Series -- the loudest event, or None when nothing triggered.
    """
    triggers = triggers_from_signal(signal[offset:], FS, WINDOW, OVERLAP,
                                    gps0=GPS0 + offset / FS)
    triggers = triggers[triggers.EnWDF > 1.0].reset_index(drop=True)
    if triggers.empty:
        return None
    graph = build_detector_graph(triggers, config=config)
    events = detector_events(graph, labels=graph.components())
    return events.loc[events.EnWDF.idxmax()] if len(events) else None


@pytest.fixture(scope="module")
def events_across_the_grid():
    """The same burst, read by grids that start at different samples."""
    signal = np.zeros(int(8 * WINDOW))
    burst = sine_gaussian(FS, 3.0 * WINDOW / FS, f0=180.0, q=12.0)
    start = (len(signal) - len(burst)) // 2
    signal[start:start + len(burst)] = burst

    config = DetectorGraphConfig()
    found = {}
    for offset in (0, STEP // 4, STEP // 2, 3 * STEP // 4):
        event = loudest_event(signal, offset, config)
        assert event is not None, f"the burst triggered nothing at offset {offset}"
        found[offset] = event
    return found


@pytest.mark.parametrize("quantity", sorted(TOLERANCE))
def test_the_event_survives_a_shift_of_the_window_grid(events_across_the_grid,
                                                       quantity):
    """No event quantity follows where the grid happened to start."""
    values = np.array([float(event[quantity])
                       for event in events_across_the_grid.values()])
    spread = float(values.max() - values.min())
    # A time is absolute, so its variation is read against the window it was
    # measured in; everything else against its own size.
    reference = (WINDOW / FS if quantity.startswith("gps")
                 else float(np.mean(np.abs(values))))
    assert spread <= TOLERANCE[quantity] * reference, (
        f"{quantity} moved by {spread:.4g} across grid offsets, more than "
        f"{TOLERANCE[quantity]:.0%} of {reference:.4g}: {values}")


def test_the_burst_is_one_event_wherever_the_boundary_falls(events_across_the_grid):
    """The grouping does not split the transient at a window edge."""
    spanned = np.array([int(event.n_triggers)
                        for event in events_across_the_grid.values()])
    assert (spanned > 1).all(), (
        f"a burst longer than one window was not assembled: {spanned}")
