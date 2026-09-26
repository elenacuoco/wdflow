"""Several detectors searched over a stretch of released frames.

The front half of the chain from frames, in the order the paper states it.
Each detector is searched on its own --- its own science time, its own lines
and noise model, fitted where its own noise is stationary --- and only the
triggers are combined downstream:

    frames --> science time --> fit stretch --> check --> search --> triggers

Science time is read from the data-quality mask the frames carry beside the
strain, on the bits the caller names, so a stretch the detector was not
observing is never searched and never counted as livetime. The released frames
carry the strain whether or not the detector was observing, and a noise model
fitted on a stretch that is flat or not finite returns a scale that silently
removes, or admits, everything.

The noise model of each searched segment is fitted on the stretch whose
periodograms have their mean closest to their median, octave by octave. A
least-squares fit such as Burg puts a transient inside the fit stretch into the
model, and a model standing above the noise whitens the stream below it; the
mean of the periodograms is lifted by a transient while their median is not,
so the ratio is one where the stretch is stationary and large where it holds
one. The choice looks at nothing but the noise, so it is the same whether or
not a signal lies elsewhere in the segment.

Every segment of every detector is checked before any of them is searched
(`wdf.processes.validation`): the whitened stream the search would read must be
white, Gaussian and stationary in every octave it reads, and clean. A failure
anywhere stops the stretch before its first search, naming the detector, the
band and the criterion. The check of a segment is made once, in a directory of
its own, and every search of that segment --- one per rule for the
coefficients of a window --- reads its lines, its model, its gates and its
verdict from there, so the searches are handed one and the same stream.

Every segment is one process holding its own C++ state. The pool is capped:
the machine is shared, and a segment is a long, memory-bound job.
"""
from __future__ import annotations

import glob
import json
import logging
import multiprocessing
import os
import re
import shutil
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

#: The released frames name their stretch in the file name, as
#: `<site>-<frametype>-<gps>-<seconds>.gwf`.
FRAME_NAME = re.compile(r"-(\d{9,10})-(\d+)\.gwf$")

#: The run name the checks are filed under. Every search of a segment reads
#: its lines, its model and its gates from there.
CHECK_RUN = "conditioning"

#: Seconds beyond which the conditioning refuses a filter that still rings
#: (`BandPassDownSampling.settling_length`): the longest settling of real data
#: a fit stretch can need on each side.
SETTLING_LIMIT_S = 120.0

#: The worker's parameters a segment's conditioning depends on. A check or a
#: search filed under the same directory with any of them different describes
#: a different stream and is not read back.
CONDITIONING_KEYS = ("channel", "sampling", "ResamplingFactor", "LowFrequencyCut",
                     "FilterOrder", "ARorder", "learn", "AREstimationOffset",
                     "WhiteningModel", "SqrtWhiteningOrder", "WhiteningExtraSize",
                     "LineThreshold", "GateThreshold", "GateTaper",
                     "SearchLowFrequency", "segments")

#: What a search adds to them: the window, the first threshold and the rule.
SEARCH_KEYS = ("window", "overlap", "threshold", "waveletThreshold")


def frame_index(directory: str) -> list:
    """Every frame of one detector's directory, read from the file names.

    A released run holds thousands of frames, and opening each to ask what it
    covers costs a read of each; the name already says it.

    :type directory: str
    :param directory: where one detector's frames are.
    :return: list -- `(path, gps, seconds)` per frame, in time order.
    """
    found = []
    for path in glob.glob(os.path.join(directory, "*.gwf")):
        match = FRAME_NAME.search(os.path.basename(path))
        if match:
            found.append((os.path.abspath(path), int(match.group(1)),
                          int(match.group(2))))
    return sorted(found, key=lambda row: row[1])


def write_frame_list(index: list, start: float, stop: float, path: str) -> str:
    """The frame file list covering a stretch, in the form the reader takes.

    :type index: list
    :param index: what `frame_index` returns.
    :type start: float
    :param start: first GPS second wanted.
    :type stop: float
    :param stop: GPS second the stretch ends at.
    :type path: str
    :param path: where to write the list.
    :return: str -- `path`.
    :raises ValueError: if no frame covers any of the stretch.
    """
    lines = [f"{frame} {gps} {seconds} 0 0" for frame, gps, seconds in index
             if gps < stop and gps + seconds > start]
    if not lines:
        raise ValueError(f"no frame covers {start:.0f}-{stop:.0f}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def mask_segments(mask, start: float, step: float, bits: int) -> list:
    """The stretches of a data-quality mask on which every named bit is set.

    :param mask: the mask, one integer per sample.
    :type start: float
    :param start: GPS time of the first sample.
    :type step: float
    :param step: seconds per sample.
    :type bits: int
    :param bits: the bits that must all be set.
    :return: list -- `(start, stop)` per stretch, GPS seconds.
    """
    mask = np.asarray(mask, dtype=np.int64).reshape(-1)
    good = ((mask & int(bits)) == int(bits)).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], good, [0]))))
    return [(float(start + a * step), float(start + b * step))
            for a, b in zip(edges[::2], edges[1::2])]


def merge_segments(segments) -> list:
    """Stretches that touch or overlap, as one.

    :param segments: `(start, stop)` pairs.
    :return: list -- merged, in time order.
    """
    merged = []
    for a, b in sorted((float(a), float(b)) for a, b in segments):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def science_segments(index: list, channel: str, start: float, stop: float,
                     bits: int, scratch: str) -> list:
    """The stretches in which the data-quality mask names the detector observing.

    Read one frame at a time, so a frame missing from the directory is time
    the detector cannot be searched on rather than a failed read.

    :type index: list
    :param index: the detector's frames, as `frame_index` returns.
    :type channel: str
    :param channel: the data-quality mask channel.
    :type start: float
    :param start: GPS start of the stretch asked about.
    :type stop: float
    :param stop: GPS end of it.
    :type bits: int
    :param bits: the bits of the mask that must all be set.
    :type scratch: str
    :param scratch: a file the one-frame list is written to.
    :return: list -- `(start, stop)` per stretch, merged and clipped to the
        stretch asked about.
    """
    from py4tsa.tsa import FrameIChannel, SeqView_double_t as SeqView

    found = []
    for frame, gps, seconds in index:
        if gps + seconds <= start or gps >= stop:
            continue
        write_frame_list([(frame, gps, seconds)], gps, gps + seconds, scratch)
        view = SeqView()
        FrameIChannel(scratch, channel, float(seconds), float(gps)).GetData(view)
        mask = np.array([view.GetY(0, i) for i in range(view.GetSize())])
        found += mask_segments(mask, view.GetStart(), view.GetSampling(), bits)
    return [(max(a, float(start)), min(b, float(stop)))
            for a, b in merge_segments(found) if b > start and a < stop]


def contamination(samples, rate: float, bands, nperseg: int = 8192) -> np.ndarray:
    """Mean over median of a stretch's periodograms, octave by octave.

    One where the stretch is stationary. A transient lifts the mean of the
    periodograms of the segments that hold it and leaves their median alone,
    so the ratio grows with how much of the stretch's power it carries.

    :param samples: the conditioned stretch.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :param bands: the octaves, as `wdf.processes.gating.octave_bands` returns.
    :type nperseg: int
    :param nperseg: samples per periodogram.
    :return: numpy.ndarray -- the median ratio in each octave.
    """
    from scipy.signal import welch

    samples = np.asarray(samples, dtype=float)
    frequency, mean = welch(samples, fs=float(rate), nperseg=int(nperseg),
                            average="mean")
    _, median = welch(samples, fs=float(rate), nperseg=int(nperseg),
                      average="median")
    ratio = mean / np.maximum(median, np.finfo(float).tiny)
    return np.array([float(np.median(ratio[(frequency >= lo) & (frequency < hi)]))
                     for lo, hi in bands])


@dataclass
class SearchConfig:
    """What each detector's search is told.

    :param frames: `{ifo: directory}` of each detector's frames.
    :param channels: `{ifo: channel}`, the strain each is searched on.
    :param quality: `{ifo: channel}`, the data-quality mask beside it.
    :param science_bits: the bits of the mask that must all be set for the
        detector to count as observing.
    :param resampling_factor: decimation from the frames' rate to the rate the
        search analyses at.
    :param low_frequency_cut: stop-band edge of the conditioning band-pass, Hz;
        the band-pass every detector shares.
    :param filter_order: order of the conditioning filter.
    :param search_low_frequency: `{ifo: Hz}`, the frequency from which a
        detector is searched and checked, for a detector whose noise is not to
        be searched as low as the shared band-pass reaches
        (`BandPassDownSampling.search_low_frequency`); a detector not named is
        searched from the band-pass's own first whole octave.
    :param line_threshold: height above the local floor, as a ratio of
        amplitude spectral densities, from which a line is notched; zero
        notches nothing.
    :param gate_threshold: height of a transient of the whitened stream, in
        robust standard deviations, from which it is gated; zero gates nothing
        the census finds.
    :param gate_taper_s: seconds over which a gate takes the stream to zero on
        each side.
    :param ar_order: order of the autoregressive noise model.
    :param learn_s: seconds the noise model is fitted on.
    :param sqrt_order: order of the square-root model the zero-phase
        whitening runs forward and backward; its own latency.
    :param extra_size: the whitening's backward look-ahead, samples of the
        analysed stream.
    :param whitening_model: `burg` or `spectrum`, as the worker reads it.
    :param window: analysis window, samples of the analysed stream.
    :param overlap: overlap between consecutive windows, samples.
    :param threshold: the search statistic `EnWDF` a window must reach to be
        written.
    :param wavelet_rule: name of the `WaveletThreshold` rule for the
        coefficients of a window.
    :param block_s: seconds read and conditioned per step; an I/O quantity,
        which changes no trigger. The worker stops a few blocks before the end
        of a segment, since it reads ahead of what it emits, so a shorter
        block leaves less of each segment unsearched.
    :param pre_white: seconds fed to the whitening before the search starts;
        a floor, which the worker raises to the filters' own settling.
    :param fit_step_s: step of the scan for the fit stretch, seconds.
    :param contamination_limit: largest mean-over-median ratio a fit stretch
        may have in any octave; a segment with none below it is not searched.
    :param minimum_segment_s: shortest science segment searched, seconds.
    :param trim_step_s: when a segment fails its check, the step, seconds, of
        the later starts it is checked again from; the earliest from which it
        passes is kept, and the stretch before it is neither searched nor
        counted. None checks each segment from its start only.
    :param processes: most processes run at once.
    """

    frames: dict
    channels: dict
    quality: dict
    science_bits: int = 1
    resampling_factor: int = 2
    low_frequency_cut: float = 6.0
    filter_order: int = 10
    search_low_frequency: dict = field(default_factory=dict)
    line_threshold: float = 5.0
    gate_threshold: float = 50.0
    gate_taper_s: float = 0.25
    ar_order: int = 3000
    learn_s: float = 300.0
    sqrt_order: int = 3000
    extra_size: int = 3000
    whitening_model: str = "burg"
    window: int = 512
    overlap: int = 32
    threshold: float = 5.0
    wavelet_rule: str = "block"
    block_s: float = 30.0
    pre_white: int = 4
    fit_step_s: float = 300.0
    contamination_limit: float = 3.0
    minimum_segment_s: float = 900.0
    trim_step_s: float | None = None
    processes: int = 32

    @property
    def ifos(self) -> list:
        """The detectors, in the order the frames are given."""
        return list(self.frames)


@dataclass
class Job:
    """One detector's work over one science segment.

    :param ifo: the detector.
    :param segment: `(start, stop)`, GPS seconds.
    :param frame_list: the frame file list covering it.
    :param outdir: where the worker writes, with a trailing separator.
    :param run: the run name the worker files its output under.
    """

    ifo: str
    segment: tuple
    frame_list: str
    outdir: str
    run: str

    def directory(self, channel: str) -> str:
        """Where the worker writes this segment's output."""
        return os.path.join(self.outdir, self.run, self.ifo,
                            f"{channel}_{int(self.segment[0])}")


class StretchRejected(RuntimeError):
    """A stretch whose conditioning failed the check somewhere, and was not searched.

    :param reports: every `ValidationReport` that failed, in the order the
        segments were planned.
    :param table: the whole check, one row per segment, as `check` would have
        returned it, passing segments included; None when not given.
    """

    def __init__(self, reports, table=None):
        super().__init__("\n".join(report.message() for report in reports))
        self.reports = list(reports)
        self.table = table


def worker_parameters(config: SearchConfig, job: Job, sampling: float,
                      fit_offset: float):
    """The configuration `wdfUnitDSWorker` is given for one job.

    :type config: SearchConfig
    :param config: the search's configuration.
    :type job: Job
    :param job: the job.
    :type sampling: float
    :param sampling: the frames' rate, Hz, as read from them.
    :type fit_offset: float
    :param fit_offset: seconds into the segment the noise model is fitted at.
    :return: wdf.config.Parameters.Parameters
    """
    from wdf.config.Parameters import Parameters

    par = Parameters()
    par.__dict__.update(
        file=job.frame_list, channel=config.channels[job.ifo], itf=job.ifo,
        run=job.run, outdir=job.outdir, dir=job.outdir,
        sampling=float(sampling), ResamplingFactor=int(config.resampling_factor),
        LowFrequencyCut=float(config.low_frequency_cut),
        FilterOrder=int(config.filter_order), ARorder=int(config.ar_order),
        learn=int(config.learn_s), preWhite=int(config.pre_white),
        AREstimationOffset=float(fit_offset),
        WhiteningModel=str(config.whitening_model),
        SqrtWhiteningOrder=int(config.sqrt_order),
        WhiteningExtraSize=int(config.extra_size),
        LineThreshold=float(config.line_threshold),
        GateThreshold=float(config.gate_threshold),
        GateTaper=float(config.gate_taper_s),
        len=float(config.block_s), window=int(config.window),
        overlap=int(config.overlap), threshold=float(config.threshold),
        gps=float(job.segment[0]), segments=[list(job.segment)], nproc=1)
    if job.ifo in config.search_low_frequency:
        par.SearchLowFrequency = float(config.search_low_frequency[job.ifo])
    return par


def _signature(par, keys) -> dict:
    """The named entries of a worker's parameters, as a JSON record holds them."""
    return json.loads(json.dumps({key: getattr(par, key, None) for key in keys}))


def frame_rate(frame_list: str, channel: str, gps: float) -> float:
    """The rate a channel is recorded at, read from the frames.

    :type frame_list: str
    :param frame_list: a frame file list.
    :type channel: str
    :param channel: the channel.
    :type gps: float
    :param gps: a second the frames hold.
    :return: float -- samples per second.
    """
    from py4tsa.tsa import FrameIChannel, SeqView_double_t as SeqView

    view = SeqView()
    FrameIChannel(frame_list, channel, 1.0, float(gps)).GetData(view)
    return float(round(1.0 / view.GetSampling()))


def _front_end(config: SearchConfig, sampling: float):
    """The conditioning the noise model is fitted under, before any line is known.

    :return: tuple -- `(front end, context)`: a `BandPassDownSampling` in
        estimation mode, and the whole seconds of real data it settles over on
        each side of a stretch.
    """
    from wdf.config.Parameters import Parameters
    from wdf.processes.BandPassDownSampling import BandPassDownSampling

    par = Parameters()
    par.__dict__.update(sampling=float(sampling),
                        ResamplingFactor=int(config.resampling_factor),
                        LowFrequencyCut=float(config.low_frequency_cut),
                        FilterOrder=int(config.filter_order), len=4.0)
    par.resampling = float(sampling) / int(config.resampling_factor)
    ds = BandPassDownSampling(par, estimation=True)
    return ds, float(np.ceil(ds.padlen / float(sampling)))


def conditioned(config: SearchConfig, job: Job, sampling: float, gps: float,
                seconds: float) -> np.ndarray:
    """A stretch of the strain conditioned as the noise model's stretch is.

    Band-passed and decimated with the shared band-pass in estimation mode,
    read with its settling of real data on each side, so the stretch's edges
    are filtered as the stream is filtered there
    (`BandPassDownSampling.condition_stretch`). The lines are not notched: they
    are found on the fit stretch itself, and a line is stationary, so it
    changes neither the mean nor the median of the periodograms this is scored
    on.

    :type config: SearchConfig
    :param config: the search's configuration.
    :type job: Job
    :param job: the job whose frames and channel are read.
    :type sampling: float
    :param sampling: the frames' rate, Hz.
    :type gps: float
    :param gps: GPS start of the stretch.
    :type seconds: float
    :param seconds: its length.
    :return: numpy.ndarray -- the band-passed, decimated samples.
    """
    from py4tsa.tsa import FrameIChannel, SeqView_double_t as SeqView

    ds, context = _front_end(config, sampling)
    view = SeqView()
    FrameIChannel(job.frame_list, config.channels[job.ifo], float(seconds) + 2.0 * context,
                  float(gps) - context).GetData(view)
    out = ds.condition_stretch(view, int(round(context * float(sampling))))
    return np.array([out.GetY(0, i) for i in range(out.GetSize())])


def quietest_offset(config: SearchConfig, job: Job, sampling: float) -> tuple:
    """Where in the segment the noise model is fitted.

    Every stretch of `learn_s` seconds, in steps of `fit_step_s`, is
    conditioned as the model's stretch is, and the one whose worst octave has
    the smallest mean-over-median ratio is kept, over the octaves the
    detector's search reads. The scan starts and ends a
    settling of the band-pass inside the segment, since each stretch is read
    with that much real data on both sides; the worker moves the stretch
    inward by the longer settling the lines' notches add, which the scan's
    step dwarfs.

    :type config: SearchConfig
    :param config: the search's configuration.
    :type job: Job
    :param job: the job.
    :type sampling: float
    :param sampling: the frames' rate, Hz.
    :return: tuple -- `(offset, contamination)`: seconds into the segment,
        and the worst octave's ratio there; `(nan, inf)` when no stretch could
        be read.
    """
    from wdf.processes.gating import octave_bands

    rate = float(sampling) / int(config.resampling_factor)
    ds, context = _front_end(config, sampling)
    # The octaves the detector's search reads, from its own low frequency or
    # the band-pass's first whole octave: a fit stretch is judged where the
    # stream it whitens is searched.
    low = float(config.search_low_frequency.get(job.ifo, ds.search_low_frequency))
    bands = octave_bands(rate, low)
    start, stop = job.segment
    best = (float("nan"), float("inf"))
    for offset in np.arange(context, stop - start - config.learn_s - context + 1e-9,
                            config.fit_step_s):
        try:
            samples = conditioned(config, job, sampling, start + offset,
                                  config.learn_s)
        except Exception as problem:            # a frame the reader refuses
            logging.warning("%s %.0f: stretch at +%.0f s not read (%s)",
                            job.ifo, start, offset, problem)
            continue
        worst = float(np.max(contamination(samples, rate, bands)))
        if worst < best[1]:
            best = (float(offset), worst)
    return best


def _limit_threads():
    """One thread per process: the pool is the parallelism, and it is capped."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:                         # pragma: no cover
        pass


def check_job(arguments) -> dict:
    """One detector's segment: fit stretch, lines, model, whitened stream, check.

    Filed under `CHECK_RUN`. A check already made there under the same
    conditioning is read back rather than made again.

    :param arguments: `(config, job)`, the job's run being `CHECK_RUN`, or
        `(config, job, fit_offset)` to fit the model at that offset into the
        segment rather than where the scan would choose, which is how a
        segment checked from a later start keeps its own fit stretch.
    :return: dict -- the job's `ifo`, `segment` and `frame_list`, the fit
        stretch chosen and its contamination, the lines notched, the
        `ValidationReport` (None when no fit stretch was clean enough to fit a
        model on), and the wall time.
    """
    from wdf.processes.validation import ValidationReport
    from wdf.processes.wdfUnitDSWorker import wdfUnitDSWorker

    config, job = arguments[:2]
    fixed = arguments[2] if len(arguments) > 2 else None
    began = time.time()
    directory = job.directory(config.channels[job.ifo])
    record = os.path.join(directory, "check.json")
    sampling = frame_rate(job.frame_list, config.channels[job.ifo], job.segment[0])
    stored = None
    if os.path.isfile(record) and os.path.isfile(
            os.path.join(directory, "conditioning-check.json")):
        with open(record, encoding="utf-8") as handle:
            stored = json.load(handle)
    if stored is not None:
        par = worker_parameters(config, job, sampling, stored["fit_offset"])
        if stored["conditioning"] == _signature(par, CONDITIONING_KEYS):
            with open(os.path.join(directory, "conditioning-check.json"),
                      encoding="utf-8") as handle:
                report = ValidationReport.from_dict(json.load(handle))
            return dict(ifo=job.ifo, segment=tuple(job.segment),
                        frame_list=job.frame_list,
                        fit_offset=stored["fit_offset"],
                        fit_contamination=stored["fit_contamination"],
                        lines=stored["lines"], report=report,
                        seconds=time.time() - began)

    offset, worst = ((float(fixed), float("nan")) if fixed is not None
                     else quietest_offset(config, job, sampling))
    if fixed is None and not (np.isfinite(offset) and worst <= config.contamination_limit):
        logging.warning("%s %.0f-%.0f not checked: the cleanest fit stretch has "
                        "mean over median %.2f, above %.2f", job.ifo,
                        *job.segment, worst, config.contamination_limit)
        return dict(ifo=job.ifo, segment=tuple(job.segment),
                    frame_list=job.frame_list, fit_offset=offset,
                    fit_contamination=worst, lines=0, report=None,
                    seconds=time.time() - began)
    par = worker_parameters(config, job, sampling, offset)
    worker = wdfUnitDSWorker(par)
    report = worker.validate(tuple(job.segment))
    lines = len(worker.par.LineNotches or [])
    with open(record, "w", encoding="utf-8") as handle:
        json.dump(dict(fit_offset=offset, fit_contamination=worst, lines=lines,
                       fit_start=float(worker.par.AREstimationStart),
                       conditioning=_signature(par, CONDITIONING_KEYS)),
                  handle, indent=1)
    return dict(ifo=job.ifo, segment=tuple(job.segment),
                frame_list=job.frame_list, fit_offset=offset,
                fit_contamination=worst, lines=lines, report=report,
                seconds=time.time() - began)


def search_job(arguments) -> dict:
    """One detector's segment searched, on the stream its check passed.

    The worker is handed the segment's model, lines and fit stretch as the
    check found them, and its gates as declared stretches with no census of
    its own, so the stream it searches is the stream the check read; the check
    is not made a second time. A search already filed under the same run with
    the same configuration is read back.

    :param arguments: `(config, job, checked)`: the search's configuration, the
        job, and what `check_job` returned for the same segment.
    :return: dict -- the job's `ifo` and `segment`, the fit stretch, the
        trigger files written, and the wall time.
    :raises RuntimeError: if the run's directory holds a search made with a
        different configuration, which would be read back as this one.
    """
    from py4tsa.tsa import WaveletThreshold

    from wdf.processes.wdfUnitDSWorker import wdfUnitDSWorker

    config, job, checked = arguments
    began = time.time()
    channel = config.channels[job.ifo]
    directory = job.directory(channel)
    check = Job(ifo=job.ifo, segment=job.segment, frame_list=job.frame_list,
                outdir=job.outdir, run=CHECK_RUN).directory(channel)
    sampling = frame_rate(job.frame_list, channel, job.segment[0])
    par = worker_parameters(config, job, sampling, checked["fit_offset"])
    par.waveletThreshold = str(config.wavelet_rule)
    wanted = _signature(par, CONDITIONING_KEYS + SEARCH_KEYS)

    written = sorted(glob.glob(os.path.join(directory, "WDFTriggers-*.parquet")))
    if os.path.isfile(os.path.join(directory, "ProcessEnded.check")) and written:
        used = glob.glob(os.path.join(directory, "parametersUsed-Win*.json"))
        with open(used[0], encoding="utf-8") as handle:
            recorded = json.load(handle)
        recorded = {key: recorded.get(key) for key in wanted}
        # The search records the gates it applied, and declares none of its
        # own, so the census threshold it ran under is the check's.
        recorded["GateThreshold"] = wanted["GateThreshold"]
        if recorded != wanted:
            differing = sorted(k for k in wanted if recorded[k] != wanted[k])
            raise RuntimeError(
                f"{directory} holds a search made with another configuration "
                f"({', '.join(differing)}); move it aside or file this one under "
                "another run name")
        return dict(ifo=job.ifo, segment=tuple(job.segment),
                    fit_offset=checked["fit_offset"], files=written,
                    seconds=time.time() - began)

    os.makedirs(directory, exist_ok=True)
    # The model the check fitted, under the name that digests its
    # conditioning: the worker finds it and fits nothing.
    for model in glob.glob(os.path.join(check, "*coeff-AR*.h5")):
        shutil.copy2(model, directory)
    shutil.copy2(os.path.join(check, "conditioning-check.json"), directory)
    par.Gates = np.asarray(checked["report"].gates, dtype=float).tolist()
    par.GateThreshold = 0.0
    par.ValidateConditioning = False
    wdfUnitDSWorker(par).segmentProcess(
        tuple(job.segment), wavThresh=getattr(WaveletThreshold, config.wavelet_rule))
    files = sorted(glob.glob(os.path.join(directory, "WDFTriggers-*.parquet")))
    return dict(ifo=job.ifo, segment=tuple(job.segment),
                fit_offset=checked["fit_offset"], files=files,
                seconds=time.time() - began)


def later_starts(config: SearchConfig, segment, fit_offset: float) -> np.ndarray:
    """The later starts a failing segment is checked from.

    Every `trim_step_s` after the segment's start, as long as what is left
    holds `minimum_segment_s` and the fit stretch with its settling of real
    data, which a later start keeps. The grid is fixed by the segment and the
    configuration alone, so the start kept depends on the data's check and on
    nothing else.

    :type config: SearchConfig
    :param config: the search's configuration.
    :param segment: `(start, stop)` of the segment, GPS seconds.
    :type fit_offset: float
    :param fit_offset: seconds into the segment its fit stretch starts at.
    :return: numpy.ndarray -- GPS starts, ascending; empty when none fits.
    """
    start, stop = float(segment[0]), float(segment[1])
    # The settling a notched filter needs is bounded by the limit
    # `settling_length` refuses beyond, so this margin always holds it.
    latest = min(stop - float(config.minimum_segment_s),
                 start + float(fit_offset) - 2.0 * SETTLING_LIMIT_S)
    step = float(config.trim_step_s)
    return start + step * np.arange(1, int(np.floor((latest - start) / step)) + 1)


def trim_job(arguments) -> dict:
    """The earliest later start from which a failing segment passes its check.

    The segment's stream is whitened once, from its own model, and the later
    starts of `later_starts` are checked in order on its tail
    (`wdfUnitDSWorker.validate_starts`) until one passes. The result is filed
    beside the segment's check under the configuration it was found with, and
    read back while that holds.

    :param arguments: `(config, job, checked)`: the search's configuration, the
        job (its run `CHECK_RUN`), and what `check_job` returned for it.
    :return: dict -- the job's `ifo` and `segment`, `start`, the earliest start
        that passes (None when none does), and `tried`, one `(start, passed,
        failures)` per start checked.
    """
    from wdf.processes.wdfUnitDSWorker import wdfUnitDSWorker

    config, job, checked = arguments
    directory = job.directory(config.channels[job.ifo])
    record = os.path.join(directory, "trim.json")
    sampling = frame_rate(job.frame_list, config.channels[job.ifo], job.segment[0])
    par = worker_parameters(config, job, sampling, checked["fit_offset"])
    starts = later_starts(config, job.segment, checked["fit_offset"])
    wanted = dict(conditioning=_signature(par, CONDITIONING_KEYS),
                  starts=[float(v) for v in starts])
    if os.path.isfile(record):
        with open(record, encoding="utf-8") as handle:
            stored = json.load(handle)
        if {key: stored.get(key) for key in wanted} == wanted:
            return dict(ifo=job.ifo, segment=tuple(job.segment), start=stored["start"],
                        tried=[tuple(t) for t in stored["tried"]])
    tried, found = [], None
    if len(starts):
        for later, report in wdfUnitDSWorker(par).validate_starts(tuple(job.segment),
                                                                   starts):
            passed = report is not None and report.passed
            tried.append((float(later), bool(passed),
                          [] if report is None else list(report.failures)))
            if passed:
                found = float(later)
    with open(record, "w", encoding="utf-8") as handle:
        json.dump(dict(wanted, start=found, tried=tried), handle, indent=1)
    return dict(ifo=job.ifo, segment=tuple(job.segment), start=found, tried=tried)


def plan(config: SearchConfig, start: float, stop: float, outdir: str,
         run: str = CHECK_RUN) -> list:
    """Every job the stretch holds: one per detector and science segment.

    :type config: SearchConfig
    :param config: the search's configuration.
    :type start: float
    :param start: GPS start of the stretch.
    :type stop: float
    :param stop: GPS end of it.
    :type outdir: str
    :param outdir: where everything is written.
    :type run: str
    :param run: the run name the output is filed under.
    :return: list of Job, the segments shorter than `minimum_segment_s` left
        out.
    """
    outdir = os.path.join(os.path.abspath(outdir), "")
    jobs = []
    for ifo in config.ifos:
        index = frame_index(config.frames[ifo])
        # Named by the stretch, so that two stretches filed under one
        # directory do not read each other's frames.
        name = f"{ifo}-{int(np.floor(start))}-{int(np.ceil(stop - start))}"
        frame_list = write_frame_list(index, start, stop,
                                      os.path.join(outdir, f"{name}.ffl"))
        scratch = os.path.join(outdir, f"{name}-quality.ffl")
        for segment in science_segments(index, config.quality[ifo], start,
                                        stop, config.science_bits, scratch):
            if segment[1] - segment[0] >= config.minimum_segment_s:
                jobs.append(Job(ifo=ifo, segment=segment,
                                frame_list=frame_list, outdir=outdir, run=run))
    return jobs


def _pool(config: SearchConfig, n_jobs: int):
    """The capped pool every job of a stretch runs in."""
    return multiprocessing.get_context("fork").Pool(
        max(1, min(int(n_jobs), int(config.processes))), initializer=_limit_threads)


def check(config: SearchConfig, start: float, stop: float, outdir: str) -> pd.DataFrame:
    """Every segment of every detector in the stretch, checked before any search.

    With `trim_step_s` set, a segment that fails is checked again from later
    starts (`later_starts`, `trim_job`) and kept from the earliest that passes,
    its own fit stretch and model unchanged; the check of the shorter segment
    is then made and filed like any other. The tolerances are the check's own:
    what is left out is the stretch before the start kept, which is neither
    searched nor counted.

    :type config: SearchConfig
    :param config: the search's configuration.
    :type start: float
    :param start: GPS start of the stretch.
    :type stop: float
    :param stop: GPS end of it.
    :type outdir: str
    :param outdir: where everything is written; a segment already checked
        there under the same conditioning is read back.
    :return: pandas.DataFrame -- one row per segment, in the order planned:
        `ifo`, `segment` (from the start kept, when a later one was),
        `frame_list`, `fit_offset`, `fit_contamination`, `lines`, the `report`
        (a `ValidationReport`, or None where no fit stretch was clean enough),
        `seconds`, `checked_from` (the science segment's own start),
        `excluded_s` (the seconds before the start kept), `tried` (one
        `(start, passed, failures)` per later start checked), `whole_report`
        (the check of the whole science segment, where a later start was kept;
        None otherwise) and `passed`.
    :raises StretchRejected: if any segment fails its check, after every
        segment has been checked; its message names each detector, band and
        criterion that failed, and it carries the whole table as `table`.
    :raises ValueError: if the stretch holds no segment long enough to search.
    """
    jobs = plan(config, start, stop, outdir, CHECK_RUN)
    if not jobs:
        raise ValueError(f"no detector holds a science segment of "
                         f"{config.minimum_segment_s:g} s in {start:.0f}-{stop:.0f}")
    with _pool(config, len(jobs)) as pool:
        results = pool.map(check_job, [(config, job) for job in jobs])
    for result in results:
        result.update(checked_from=result["segment"][0], excluded_s=0.0, tried=[],
                      whole_report=None)
    failing = [k for k, r in enumerate(results)
               if r["report"] is not None and not r["report"].passed]
    if config.trim_step_s and failing:
        # Every failing segment of every detector goes through the same rule.
        with _pool(config, len(failing)) as pool:
            trims = pool.map(trim_job, [(config, jobs[k], results[k]) for k in failing])
        later = [(k, trim) for k, trim in zip(failing, trims) if trim["start"] is not None]
        shorter = [replace(jobs[k], segment=(trim["start"], jobs[k].segment[1]))
                   for k, trim in later]
        offsets = [results[k]["fit_offset"] + jobs[k].segment[0] - trim["start"]
                   for k, trim in later]
        with _pool(config, max(len(shorter), 1)) as pool:
            rechecked = pool.map(check_job, [(config, job, offset) for job, offset
                                             in zip(shorter, offsets)])
        for (k, trim), result in zip(later, rechecked):
            result.update(checked_from=jobs[k].segment[0],
                          excluded_s=trim["start"] - jobs[k].segment[0],
                          whole_report=results[k]["report"])
            results[k] = result
        for k, trim in zip(failing, trims):
            results[k]["tried"] = trim["tried"]
    table = pd.DataFrame(results)
    table["passed"] = [r is not None and r.passed for r in table["report"]]
    failed = [r for r in table["report"] if r is not None and not r.passed]
    if failed:
        raise StretchRejected(failed, table)
    return table


def search(config: SearchConfig, start: float, stop: float, outdir: str,
           run: str | None = None):
    """Every detector searched over its science time in the stretch.

    The segments are checked first (`check`); nothing is searched unless every
    one of them passes.

    :type config: SearchConfig
    :param config: the search's configuration.
    :type start: float
    :param start: GPS start of the stretch.
    :type stop: float
    :param stop: GPS end of it.
    :type outdir: str
    :param outdir: where everything is written; a segment already checked or
        searched there under the same configuration is read back rather than
        done again.
    :type run: str | None
    :param run: the run name the triggers are filed under; the rule's name
        when None, so that searches under two rules sit side by side.
    :return: tuple -- `(triggers, spans, jobs)`: `{ifo: triggers}` as
        `wdf.analysis.io.triggers_from_files` reads them, segments combined;
        `{ifo: [(start, end), ...]}`, the stretch each segment's search
        actually covered, from its first window to the end of its last; and
        one row per segment saying where its model was fitted, how clean that
        stretch was, what its check measured, how long its search took and
        which trigger files it wrote (`trigger_files`), beside each of which
        the worker recorded the parameters it ran with.
    :raises StretchRejected: if any segment fails its check.
    """
    from wdf.analysis.io import triggers_from_files

    run = str(config.wavelet_rule) if run is None else str(run)
    if run == CHECK_RUN:
        raise ValueError(f"{CHECK_RUN!r} is where the checks are filed")
    checked = check(config, start, stop, outdir)
    arguments = []
    for row in checked.itertuples(index=False):
        if row.report is None:
            continue
        job = Job(ifo=row.ifo, segment=tuple(row.segment), frame_list=row.frame_list,
                  outdir=os.path.join(os.path.abspath(outdir), ""), run=run)
        arguments.append((config, job, row._asdict()))
    results = []
    with _pool(config, len(arguments)) as pool:
        for result in pool.imap_unordered(search_job, arguments):
            results.append(result)
            logging.info("%s %.0f-%.0f: %d file(s) in %.0f s", result["ifo"],
                         *result["segment"], len(result["files"]),
                         result["seconds"])

    triggers, spans = {}, {}
    for result in sorted(results, key=lambda r: (r["ifo"], r["segment"])):
        if not result["files"]:
            continue
        found = triggers_from_files(result["files"], result["ifo"])
        if found.empty:
            continue
        end = found["gps"].to_numpy(dtype=float) + (
            found["n_coeff"].to_numpy(dtype=float) / found["fs"].to_numpy(dtype=float))
        spans.setdefault(result["ifo"], []).append(
            (float(found["gps"].min()), float(end.max())))
        triggers.setdefault(result["ifo"], []).append(found)
    triggers = {ifo: pd.concat(parts, ignore_index=True)
                for ifo, parts in triggers.items()}
    seconds = {(r["ifo"], tuple(r["segment"])): r["seconds"] for r in results}
    files = {(r["ifo"], tuple(r["segment"])): list(r["files"]) for r in results}
    table = checked.assign(
        search_seconds=[seconds.get((i, tuple(s)), np.nan)
                        for i, s in zip(checked.ifo, checked.segment)],
        trigger_files=[files.get((i, tuple(s)), [])
                       for i, s in zip(checked.ifo, checked.segment)])
    return triggers, spans, table


def checked_segment(config: SearchConfig, start: float, stop: float, outdir: str,
                    ifo: str, first: float, last: float):
    """The checked segment of a detector holding two instants, as its search reads it.

    Found among the segments `check` filed: from the later start it was kept
    from, when it was and the instants follow that start, since the search
    reads that segment's stream, gated by its own check; the stretch before
    the start kept is read on the whole segment's, the stream its search would
    have read.

    :type config: SearchConfig
    :param config: the configuration the stretch was checked with.
    :type start: float
    :param start: GPS start of the stretch.
    :type stop: float
    :param stop: GPS end of it.
    :type outdir: str
    :param outdir: where the stretch was checked.
    :type ifo: str
    :param ifo: the detector.
    :type first: float
    :param first: GPS time of the first instant.
    :type last: float
    :param last: GPS time of the last instant.
    :return: tuple -- `(segment, parameters, gates)`: the segment's `(start,
        stop)`, the worker's parameters as its check was made with them, and
        its gated stretches.
    :raises ValueError: if no checked segment of the detector holds both
        instants, or if the stretch was not checked under this configuration.
    """
    from wdf.processes.validation import ValidationReport

    jobs = [job for job in plan(replace(config, frames={ifo: config.frames[ifo]}),
                                start, stop, outdir, CHECK_RUN)
            if job.segment[0] <= first and last <= job.segment[1]]
    if not jobs:
        raise ValueError(f"no science segment of {ifo} holds {first}-{last}")
    job = jobs[0]
    trimmed = os.path.join(job.directory(config.channels[ifo]), "trim.json")
    if os.path.isfile(trimmed):
        with open(trimmed, encoding="utf-8") as handle:
            kept = json.load(handle).get("start")
        if kept is not None and first >= kept:
            job = replace(job, segment=(float(kept), job.segment[1]))
    directory = job.directory(config.channels[ifo])
    record = os.path.join(directory, "check.json")
    if not os.path.isfile(record):
        raise ValueError(f"{ifo} {job.segment[0]:.0f} was not checked in {outdir}")
    with open(record, encoding="utf-8") as handle:
        stored = json.load(handle)
    sampling = frame_rate(job.frame_list, config.channels[ifo], job.segment[0])
    par = worker_parameters(config, job, sampling, stored["fit_offset"])
    if stored["conditioning"] != _signature(par, CONDITIONING_KEYS):
        raise ValueError(f"{ifo} {job.segment[0]:.0f} was checked under another "
                         "conditioning")
    with open(os.path.join(directory, "conditioning-check.json"),
              encoding="utf-8") as handle:
        gates = ValidationReport.from_dict(json.load(handle)).gates
    return tuple(job.segment), par, gates


def whitened_around(config: SearchConfig, start: float, stop: float, outdir: str,
                    ifo: str, first: float, last: float):
    """A detector's whitened stream between two instants, as its search read it.

    From the segment `checked_segment` finds, rebuilt from the model, the
    lines, the fit stretch and the gates its check recorded
    (`wdfUnitDSWorker.whitened_stretch`), on the search's noise scale.

    :type config: SearchConfig
    :param config: the configuration the stretch was checked with.
    :type start: float
    :param start: GPS start of the stretch.
    :type stop: float
    :param stop: GPS end of it.
    :type outdir: str
    :param outdir: where the stretch was checked.
    :type ifo: str
    :param ifo: the detector.
    :type first: float
    :param first: GPS time of the first sample wanted.
    :type last: float
    :param last: GPS time of the last sample wanted.
    :return: tuple -- `(t0, samples)` at the analysed rate.
    :raises ValueError: as `checked_segment`.
    """
    from wdf.processes.wdfUnitDSWorker import wdfUnitDSWorker

    segment, par, gates = checked_segment(config, start, stop, outdir, ifo, first, last)
    return wdfUnitDSWorker(par).whitened_stretch(segment, first, last, gates)
