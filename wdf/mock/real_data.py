"""The mock waveform set injected into real strain.

The simulated set is Gaussian and stationary by construction, so it cannot show
how the search behaves against the noise a detector actually produces. This
draws the same population of signals and injects it into recorded strain,
keeping what makes the simulated set usable: the foreground and the background
are written from one reading of the data, differing only by the injections, so
their difference is exactly the injected waveform and the background is the same
noise realisation without it.

Everything about the signals is reused unchanged --- `draw_injections` for the
population, `_inject_one` for the projection onto each detector with its antenna
response and light travel time, and `_write_frames` for the output. What differs
is where the noise comes from and how its spectrum is obtained: measured from
the data rather than taken from a design curve, since a recorded spectrum is
neither the design one nor stationary across a run.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from wdf.mock.dataset import (
    GROUND_TRUTH_COLUMNS,
    _inject_one,
    _write_frames,
    draw_injections,
)


def measured_psd(strain, sample_rate, segment_s=8.0):
    """The one-sided power spectral density of a stretch of strain.

    :type strain: numpy.ndarray
    :param strain: the samples.
    :type sample_rate: float
    :param sample_rate: samples per second.
    :type segment_s: float
    :param segment_s: length of the averaging segment, seconds.
    :return: pycbc.types.FrequencySeries -- the density, on the frequency
        spacing the averaging segment implies. A series and not a pair of
        arrays: what consumes it interpolates onto the spacing a waveform
        needs, and an interpolation needs to know its own frequency step.
    """
    from pycbc.types import FrequencySeries
    from scipy.signal import welch

    nperseg = int(round(segment_s * float(sample_rate)))
    frequency, density = welch(np.asarray(strain, dtype=float),
                               fs=float(sample_rate), nperseg=nperseg)
    return FrequencySeries(density, delta_f=float(frequency[1] - frequency[0]))


def read_strain(frames, channels, start_gps, duration):
    """Read one span of strain per detector from GWF.

    :type frames: dict[str, str]
    :param frames: ``{ifo: path}`` of the GWF files.
    :type channels: dict[str, str]
    :param channels: ``{ifo: channel}`` to read from each.
    :type start_gps: float
    :param start_gps: GPS time of the first sample.
    :type duration: float
    :param duration: span to read, seconds.
    :return: tuple -- ({ifo: samples}, sample_rate).
    """
    from gwpy.timeseries import TimeSeries

    strain, rate = {}, None
    for ifo, path in frames.items():
        series = TimeSeries.read(path, channels[ifo], start=start_gps,
                                 end=start_gps + duration)
        strain[ifo] = np.asarray(series.value, dtype=float)
        rate = float(series.sample_rate.value) if rate is None else rate
    return strain, rate


def inject_into_strain(
    outdir,
    frames,
    channels,
    start_gps,
    duration,
    n_cbc=60,
    n_glitch=60,
    snr_range=(6.0, 60.0),
    seed=0,
    minimum_gap=400.0,
    edge_pad=400.0,
    low_frequency_cutoff=20.0,
    high_frequency_cutoff=None,
    channel_suffix="REAL-STRAIN",
    frame_length=1024.0,
    psd_segment_s=8.0,
):
    """Write a foreground and a background frame set from recorded strain.

    The two differ only by the injections, so their difference is the injected
    waveform and the background is the same noise without it.

    `minimum_gap` defaults to well beyond the longest signal drawn: on the
    simulated set a binary-neutron-star injection lasts of order two hundred
    seconds, and injections packed closer than they are long leave no stretch of
    noise between them to measure a background against.

    :type outdir: str
    :param outdir: directory to write the frames, FFL indices and truth table to.
    :type frames: dict[str, str]
    :param frames: ``{ifo: path}`` of the recorded GWF files.
    :type channels: dict[str, str]
    :param channels: ``{ifo: channel}`` to read from each.
    :type start_gps: float
    :param start_gps: GPS time of the first sample to use.
    :type duration: float
    :param duration: span to use, seconds.
    :type n_cbc: int
    :param n_cbc: compact-binary signals to draw.
    :type n_glitch: int
    :param n_glitch: single-detector transients to draw.
    :type snr_range: tuple[float, float]
    :param snr_range: range of optimal network signal-to-noise ratio to draw in.
    :type seed: int
    :param seed: seed for the draw.
    :type minimum_gap: float
    :param minimum_gap: least separation between injections, seconds.
    :type edge_pad: float
    :param edge_pad: span kept free at each end, seconds.
    :type low_frequency_cutoff: float
    :param low_frequency_cutoff: lower bound of the band signals are drawn in.
    :type high_frequency_cutoff: float | None
    :param high_frequency_cutoff: upper bound; Nyquist if None.
    :type channel_suffix: str
    :param channel_suffix: written channel name after the detector prefix.
    :type frame_length: float
    :param frame_length: seconds per output frame.
    :type psd_segment_s: float
    :param psd_segment_s: averaging segment for the measured spectrum, seconds.
    :return: pandas.DataFrame -- the truth table, also written as
        `injections.parquet`.
    """
    outdir = os.fspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    detectors = tuple(frames)
    strain, sample_rate = read_strain(frames, channels, start_gps, duration)
    nyquist = 0.5 * sample_rate
    f_high = nyquist if high_frequency_cutoff is None else float(high_frequency_cutoff)

    # The spectrum the signals are scaled against is the one the data has, not a
    # design curve: a recorded spectrum is neither, and the amplitude of an
    # injection is only meaningful against the noise it has to be seen through.
    psd = {ifo: measured_psd(values, sample_rate, psd_segment_s)
           for ifo, values in strain.items()}

    specs = draw_injections(
        n_cbc=n_cbc, n_glitch=n_glitch, duration=duration, start_gps=start_gps,
        edge_pad=edge_pad, snr_range=snr_range, seed=seed,
        sample_rate=int(sample_rate), detectors=detectors,
        minimum_gap=minimum_gap)

    background = {ifo: values.copy() for ifo, values in strain.items()}
    foreground = {ifo: values.copy() for ifo, values in strain.items()}

    rows = []
    for spec in specs:
        rows.append(_inject_one(
            spec, foreground, start_gps, sample_rate, detectors,
            low_frequency_cutoff, f_high, psd_name=None, psd=psd))
    truth = pd.DataFrame(rows, columns=GROUND_TRUTH_COLUMNS)

    from gwpy.timeseries import TimeSeries

    for ifo in detectors:
        for tag, data in (("foreground", foreground[ifo]),
                          ("background", background[ifo])):
            _write_frames(TimeSeries, data, start_gps, sample_rate, ifo,
                          channel_suffix, outdir, tag, frame_length=frame_length)

    truth.to_parquet(os.path.join(outdir, "injections.parquet"), index=False)
    return truth
