"""Several detectors searched over a stretch of released frames, then released.

The whole chain from frames, in the order the paper states it. Each detector
is searched on its own --- its own science time, its own noise model, fitted
where its own noise is stationary --- and only the triggers are combined:

    frames --> science time --> fit stretch --> search --> triggers
           --> wdf.analysis.release: events, event cut, network, rate

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

Every searched segment is one process holding its own C++ state. The pool is
capped: the machine is shared, and a segment is a long, memory-bound job.
"""
from __future__ import annotations

import glob
import logging
import multiprocessing
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: The released frames name their stretch in the file name, as
#: `<site>-<frametype>-<gps>-<seconds>.gwf`.
FRAME_NAME = re.compile(r"-(\d{9,10})-(\d+)\.gwf$")


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


def octaves(rate: float, low: float) -> tuple:
    """The octave bands from half the rate down to a low edge.

    :type rate: float
    :param rate: sampling rate of the analysed stream, Hz.
    :type low: float
    :param low: the lowest frequency of interest, Hz.
    :return: tuple -- `(low, high)` per octave, ascending.
    """
    bands, top = [], 0.5 * float(rate)
    while top > float(low):
        bands.append((max(0.5 * top, float(low)), top))
        top *= 0.5
    return tuple(reversed(bands))


def contamination(samples, rate: float, bands, nperseg: int = 8192) -> np.ndarray:
    """Mean over median of a stretch's periodograms, octave by octave.

    One where the stretch is stationary. A transient lifts the mean of the
    periodograms of the segments that hold it and leaves their median alone,
    so the ratio grows with how much of the stretch's power it carries.

    :param samples: the conditioned stretch.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :param bands: the octaves, as `octaves` returns.
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
    :param low_frequency_cut: stop-band edge of the conditioning filter, Hz.
    :param filter_order: order of the conditioning filter.
    :param ar_order: order of the autoregressive noise model.
    :param learn_s: seconds the noise model is fitted on.
    :param sqrt_order: order of the square-root model the zero-phase
        whitening runs forward and backward; its own latency.
    :param extra_size: the whitening's backward look-ahead, samples of the
        analysed stream.
    :param whitening_model: `burg` or `spectrum`, as the worker reads it.
    :param window: analysis window, samples of the analysed stream.
    :param overlap: overlap between consecutive windows, samples.
    :param threshold: the first threshold: the search statistic a window must
        reach to be written. At the block rule's floor every window that keeps
        a coefficient is written, since a surviving block carries `EnWDF` of at
        least `sqrt(4.505)`.
    :param wavelet_rule: name of the `WaveletThreshold` rule for the
        coefficients of a window.
    :param block_s: seconds read and conditioned per step; an I/O quantity,
        which changes no trigger. The worker stops a few blocks before the end
        of a segment, since it reads ahead of what it emits, so a shorter
        block leaves less of each segment unsearched.
    :param pre_white: seconds fed to the whitening before the search starts.
    :param fit_step_s: step of the scan for the fit stretch, seconds.
    :param contamination_limit: largest mean-over-median ratio a fit stretch
        may have in any octave; a segment with none below it is not searched.
    :param minimum_segment_s: shortest science segment searched, seconds.
    :param processes: most processes run at once.
    """

    frames: dict
    channels: dict
    quality: dict
    science_bits: int = 1
    resampling_factor: int = 2
    low_frequency_cut: float = 6.0
    filter_order: int = 10
    ar_order: int = 3000
    learn_s: float = 300.0
    sqrt_order: int = 3000
    extra_size: int = 3000
    whitening_model: str = "burg"
    window: int = 512
    overlap: int = 32
    # The block rule's floor: a window keeping any block has EnWDF >= sqrt(4.505).
    threshold: float = 2.12
    wavelet_rule: str = "block"
    block_s: float = 30.0
    pre_white: int = 4
    fit_step_s: float = 300.0
    contamination_limit: float = 3.0
    minimum_segment_s: float = 900.0
    processes: int = 32

    @property
    def ifos(self) -> list:
        """The detectors, in the order the frames are given."""
        return list(self.frames)


@dataclass
class Job:
    """One detector's search over one science segment.

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
        """Where the worker writes this segment's triggers."""
        return os.path.join(self.outdir, self.run, self.ifo,
                            f"{channel}_{int(self.segment[0])}")


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
        len=float(config.block_s), window=int(config.window),
        overlap=int(config.overlap), threshold=float(config.threshold),
        gps=float(job.segment[0]), segments=[list(job.segment)], nproc=1)
    return par


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


def conditioned(config: SearchConfig, job: Job, sampling: float, gps: float,
                seconds: float) -> np.ndarray:
    """A stretch of the strain conditioned as the search conditions it.

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

    from wdf.config.Parameters import Parameters
    from wdf.processes.BandPassDownSampling import BandPassDownSampling

    par = Parameters()
    par.__dict__.update(sampling=float(sampling),
                        ResamplingFactor=int(config.resampling_factor),
                        LowFrequencyCut=float(config.low_frequency_cut),
                        FilterOrder=int(config.filter_order), len=4.0)
    par.resampling = float(sampling) / int(config.resampling_factor)
    view = SeqView()
    FrameIChannel(job.frame_list, config.channels[job.ifo], float(seconds),
                  float(gps)).GetData(view)
    out = BandPassDownSampling(par, estimation=True).Process(view)
    return np.array([out.GetY(0, i) for i in range(out.GetSize())])


def quietest_offset(config: SearchConfig, job: Job, sampling: float) -> tuple:
    """Where in the segment the noise model is fitted.

    Every stretch of `learn_s` seconds, in steps of `fit_step_s`, is
    conditioned as the search conditions it, and the one whose worst octave
    has the smallest mean-over-median ratio is kept.

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
    rate = float(sampling) / int(config.resampling_factor)
    bands = octaves(rate, config.low_frequency_cut)
    start, stop = job.segment
    best = (float("nan"), float("inf"))
    for offset in np.arange(0.0, stop - start - config.learn_s + 1e-9,
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


def run_job(arguments) -> dict:
    """One detector over one science segment: fit stretch, then search.

    :param arguments: `(config, job)`.
    :return: dict -- the job's `ifo` and `segment`, the fit stretch chosen and
        its contamination, the trigger files written, and the wall time.
    """
    from py4tsa.tsa import WaveletThreshold

    from wdf.processes.wdfUnitDSWorker import wdfUnitDSWorker

    from wdf.analysis.io import run_parameters

    config, job = arguments
    began = time.time()
    directory = job.directory(config.channels[job.ifo])
    written = sorted(glob.glob(os.path.join(directory, "WDFTriggers-*.parquet")))
    if os.path.isfile(os.path.join(directory, "ProcessEnded.check")) and written:
        # Searched before: the fit stretch is what the worker recorded.
        used = run_parameters(written[0])
        return dict(ifo=job.ifo, segment=tuple(job.segment),
                    fit_offset=float(getattr(used, "AREstimationOffset", np.nan)),
                    fit_contamination=np.nan, files=written,
                    seconds=time.time() - began)

    sampling = frame_rate(job.frame_list, config.channels[job.ifo], job.segment[0])
    offset, worst = quietest_offset(config, job, sampling)
    files = []
    if np.isfinite(offset) and worst <= config.contamination_limit:
        par = worker_parameters(config, job, sampling, offset)
        wdfUnitDSWorker(par).segmentProcess(
            tuple(job.segment), wavThresh=getattr(WaveletThreshold,
                                                  config.wavelet_rule))
        files = sorted(glob.glob(os.path.join(directory, "WDFTriggers-*.parquet")))
    else:
        logging.warning("%s %.0f-%.0f not searched: the cleanest fit stretch has "
                        "mean over median %.2f, above %.2f", job.ifo,
                        *job.segment, worst, config.contamination_limit)
    return dict(ifo=job.ifo, segment=tuple(job.segment), fit_offset=offset,
                fit_contamination=worst, files=files,
                seconds=time.time() - began)


def plan(config: SearchConfig, start: float, stop: float, outdir: str,
         run: str = "search") -> list:
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
        frame_list = write_frame_list(index, start, stop,
                                      os.path.join(outdir, f"{ifo}.ffl"))
        scratch = os.path.join(outdir, f"{ifo}-quality.ffl")
        for segment in science_segments(index, config.quality[ifo], start,
                                        stop, config.science_bits, scratch):
            if segment[1] - segment[0] >= config.minimum_segment_s:
                jobs.append(Job(ifo=ifo, segment=segment,
                                frame_list=frame_list, outdir=outdir, run=run))
    return jobs


def search(config: SearchConfig, start: float, stop: float, outdir: str,
           run: str = "search"):
    """Every detector searched over its science time in the stretch.

    :type config: SearchConfig
    :param config: the search's configuration.
    :type start: float
    :param start: GPS start of the stretch.
    :type stop: float
    :param stop: GPS end of it.
    :type outdir: str
    :param outdir: where everything is written; a segment already searched
        there is read back rather than searched again.
    :type run: str
    :param run: the run name the output is filed under.
    :return: tuple -- `(triggers, spans, jobs)`: `{ifo: triggers}` as
        `wdf.analysis.io.triggers_from_files` reads them, segments combined;
        `{ifo: [(start, end), ...]}`, the stretch each segment's search
        actually covered, from its first window to the end of its last; and
        one row per job saying where its model was fitted, how clean that
        stretch was and how long it took.
    """
    from wdf.analysis.io import triggers_from_files

    jobs = plan(config, start, stop, outdir, run)
    results = []
    with multiprocessing.get_context("fork").Pool(
            max(1, min(len(jobs), int(config.processes)))) as pool:
        for result in pool.imap_unordered(run_job, [(config, job) for job in jobs]):
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
    table = pd.DataFrame([{key: value for key, value in result.items()
                           if key != "files"} | dict(files=len(result["files"]))
                          for result in results])
    return triggers, spans, table


def search_and_release(config: SearchConfig, release_config, start: float,
                       stop: float, outdir: str, run: str = "search"):
    """The network's released candidates over a stretch of frames.

    The one call from frames to the released list: every detector searched at
    the first threshold over its own science time, and the triggers handed to
    `wdf.analysis.release.release`.

    :type config: SearchConfig
    :param config: what each detector's search is told.
    :type release_config: wdf.analysis.release.ReleaseConfig
    :param release_config: what the detector and network stages are told.
    :type start: float
    :param start: GPS start of the stretch.
    :type stop: float
    :param stop: GPS end of it.
    :type outdir: str
    :param outdir: where everything is written.
    :type run: str
    :param run: the run name the output is filed under.
    :return: tuple -- `(release, jobs)`: the `Release`, and the table of the
        jobs the search ran.
    :raises ValueError: if the first threshold is above the reference the
        second is translated from, or the window does not match.
    """
    from wdf.analysis.release import release

    if config.threshold > release_config.reference_threshold:
        raise ValueError(
            f"the first threshold {config.threshold:g} is above the reference "
            f"{release_config.reference_threshold:g}: the reference population "
            "would not be contained in what was searched")
    if (config.window, config.overlap) != (release_config.window,
                                           release_config.overlap):
        raise ValueError("the search and the release are told different windows")
    triggers, spans, jobs = search(config, start, stop, outdir, run)
    return release(triggers, spans, release_config), jobs
