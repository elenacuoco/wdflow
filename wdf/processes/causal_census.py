"""The census of transients of a whitened stream, causal, block by block.

`CausalCensus` is the online counterpart of the offline census
(`wdf.processes.census.transients`, `glitch_subtraction.glitch_seeds`): the
same levels, read from the past only. Each block of the whitened stream is
split into the broadband stream and its octaves from `GlitchCensusLow`
(`octave_bands`) by causal Butterworth filters of order `GlitchOctaveOrder`
(`octave_filter` run forward, its state carried). Each stream is divided by
its robust standard deviation, ``median |s| / MAD_GAUSSIAN``, measured on the
`GlitchCensusMemory` seconds of that stream strictly before the block; until
`GlitchCensusMinimum` seconds are held there is no census. A sample's level
is the largest of these ratios; samples above `GlitchFlag` closer than
`GlitchMergeGap` seconds form one transient, kept when its peak reaches
`GlitchTrigger`. Its extent is where the broadband stream or an octave from
`GlitchExtentLow` exceeds `GlitchFlag`: the lowest octaves carry the
whitening's own memory and would size it by the filter.

`overlaps` says whether a stretch of time meets a transient known so far,
open or closed, widened by `GlitchSeedPad`: the online worker labels a
trigger "glitch" when its window does (`GlitchLabel` "census"). The census
reads the whitened stream up to the end of the block the trigger is emitted
with, so the label uses no data beyond the search's own latency.
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from wdf.config.defaults import defaults
from wdf.processes.census import (DEFAULT_OCTAVE_FILTER_ORDER, MAD_GAUSSIAN, octave_bands,
                                  octave_filter)

#: The defaults of the online census: table ``[census.causal]`` of
#: wdf/config/defaults.toml; the others are those of the offline census
#: (``[glitch_subtraction]``, ``[census]``).
_CAUSAL = defaults("census.causal")
#: The offline census's table, for the keys the two share.
_GS = defaults("glitch_subtraction")

#: Default seconds of past each stream's robust sigma is measured on
#: (`GlitchCensusMemory`).
DEFAULT_CENSUS_MEMORY_S = _CAUSAL["GlitchCensusMemory"]
#: Default seconds of past needed before the census starts (`GlitchCensusMinimum`).
DEFAULT_CENSUS_MINIMUM_S = _CAUSAL["GlitchCensusMinimum"]
#: The glitch labels the online worker can apply (`GlitchLabel`).
GLITCH_LABELS = ("census", "bases")


@dataclass
class CensusSettings:
    """The census's choices, read from the run parameters (the names of the
    offline census, `M4Settings`).

    :param trigger: smallest peak of a transient, robust sigma
        (`GlitchTrigger`).
    :param flag: level of a transient's samples (`GlitchFlag`).
    :param merge_s: largest gap inside one transient, s (`GlitchMergeGap`).
    :param low: lowest octave, Hz (`GlitchCensusLow`, default `MinFrequency`,
        else `LowFrequencyCut`).
    :param extent_low: lowest octave of the extent, Hz (`GlitchExtentLow`,
        default `[glitch_subtraction] extent_low_factor` times `low`).
    :param pad: added on each side of a transient's extent, s
        (`GlitchSeedPad`).
    :param order: octave filter order (`GlitchOctaveOrder`).
    :param memory_s: past of the robust sigma, s (`GlitchCensusMemory`).
    :param minimum_s: past needed before the census starts, s
        (`GlitchCensusMinimum`).
    """

    trigger: float
    flag: float
    merge_s: float
    low: float
    extent_low: float
    pad: float
    order: int
    memory_s: float
    minimum_s: float

    KEYS = dict(trigger="GlitchTrigger", flag="GlitchFlag", merge_s="GlitchMergeGap",
                low="GlitchCensusLow", extent_low="GlitchExtentLow", pad="GlitchSeedPad",
                order="GlitchOctaveOrder", memory_s="GlitchCensusMemory",
                minimum_s="GlitchCensusMinimum")

    @classmethod
    def from_parameters(cls, par, write=True):
        """The settings a run asks for, defaults filled in and (`write`)
        written back."""
        from wdf.config.Parameters import low_frequency_cut
        from wdf.processes import glitch_subtraction as gs

        def get(key, default):
            value = getattr(par, key, None)
            return default if value is None else value

        low = get("GlitchCensusLow", _GS["GlitchCensusLow"])
        low = float(getattr(par, "MinFrequency", None) or low_frequency_cut(par) if low is None
                    else low)
        extent_low = get("GlitchExtentLow", _GS["GlitchExtentLow"])
        out = cls(trigger=float(get("GlitchTrigger", gs.DEFAULT_TRIGGER)),
                  flag=float(get("GlitchFlag", gs.DEFAULT_FLAG)),
                  merge_s=float(get("GlitchMergeGap", gs.DEFAULT_MERGE_S)), low=low,
                  extent_low=float(gs.DEFAULT_EXTENT_LOW_FACTOR * low if extent_low is None
                                   else extent_low),
                  pad=float(get("GlitchSeedPad", gs.DEFAULT_SEED_PAD_S)),
                  order=int(get("GlitchOctaveOrder", DEFAULT_OCTAVE_FILTER_ORDER)),
                  memory_s=float(get("GlitchCensusMemory", DEFAULT_CENSUS_MEMORY_S)),
                  minimum_s=float(get("GlitchCensusMinimum", DEFAULT_CENSUS_MINIMUM_S)))
        if write:
            for name, key in cls.KEYS.items():
                setattr(par, key, getattr(out, name))
        return out

    def seeds(self):
        """The keyword arguments of `glitch_seeds` (the offline census)."""
        return dict(trigger=self.trigger, low=self.low, extent_low=self.extent_low,
                    flag=self.flag, merge_s=self.merge_s, pad=self.pad, order=self.order)


class CausalCensus:
    """The census of a whitened stream, one block at a time, from the past.

    :type settings: CensusSettings
    :param settings: the census's settings.
    :type rate: float
    :param rate: sampling rate of the whitened stream, Hz.
    """

    def __init__(self, settings, rate):
        self.settings, self.rate = settings, float(rate)
        self.bands = octave_bands(self.rate, settings.low)
        self.sos = [octave_filter(self.rate, b, settings.order) for b in self.bands]
        self.state = [None] * len(self.sos)
        self.extent_rows = [0] + [i + 1 for i, b in enumerate(self.bands)
                                  if b[0] >= settings.extent_low]
        held = int(round(settings.memory_s * self.rate))
        self.past = [deque(maxlen=held) for _ in range(len(self.sos) + 1)]
        self.minimum = int(round(settings.minimum_s * self.rate))
        self.closed = []
        self.open = None

    def _streams(self, w):
        out = [w]
        for i, sos in enumerate(self.sos):
            if self.state[i] is None:
                self.state[i] = sosfilt_zi(sos) * 0.0
            y, self.state[i] = sosfilt(sos, w, zi=self.state[i])
            out.append(y)
        return out

    def push(self, w, start):
        """Read the next whitened block.

        :type w: numpy.ndarray
        :param w: the block.
        :type start: float
        :param start: GPS time of its first sample.
        :return: list[dict] -- the transients closed by it.
        """
        s = self.settings
        w = np.asarray(w, dtype=float)
        streams = self._streams(w)
        ready = len(self.past[0]) >= self.minimum
        sigmas = ([max(float(np.median(np.abs(np.fromiter(p, float)))) / MAD_GAUSSIAN, 1e-300)
                   for p in self.past] if ready else None)
        for p, y in zip(self.past, streams):
            p.extend(y.tolist())
        if not ready:
            return []
        ratios = [np.abs(y) / sg for y, sg in zip(streams, sigmas)]
        level = np.maximum.reduce(ratios)
        extent = np.maximum.reduce([ratios[i] for i in self.extent_rows])
        t = start + np.arange(w.size) / self.rate
        done = []
        for k in np.flatnonzero(level > s.flag):
            ext = bool(extent[k] > s.flag)
            op = self.open
            if op is not None and t[k] - op["last"] > s.merge_s:
                done.append(self._close())
                op = None
            if op is None:
                self.open = op = dict(first=t[k], last=t[k], peak=0.0, peak_time=t[k],
                                      extent_first=None, extent_last=None)
            op["last"] = t[k]
            if level[k] > op["peak"]:
                op["peak"], op["peak_time"] = float(level[k]), float(t[k])
            if ext:
                op["extent_first"] = t[k] if op["extent_first"] is None else op["extent_first"]
                op["extent_last"] = t[k]
        if self.open is not None and t[-1] - self.open["last"] > s.merge_s:
            done.append(self._close())
        return [d for d in done if d is not None]

    def _close(self):
        op, self.open = self.open, None
        if op is None or op["peak"] < self.settings.trigger:
            return None
        row = self._row(op)
        self.closed.append(row)
        return row

    def _row(self, op):
        first = op["extent_first"] if op["extent_first"] is not None else op["peak_time"]
        last = op["extent_last"] if op["extent_last"] is not None else op["peak_time"]
        return dict(first=float(op["first"]), last=float(op["last"]), peak=float(op["peak"]),
                    peak_time=float(op["peak_time"]), extent_first=float(first),
                    extent_last=float(last + 1.0 / self.rate))

    def known(self):
        """The transients known so far: closed, and the open one when its peak
        already reaches the trigger."""
        out = list(self.closed)
        if self.open is not None and self.open["peak"] >= self.settings.trigger:
            out.append(self._row(self.open))
        return out

    def overlaps(self, t0, t1):
        """Whether ``[t0, t1)`` meets a known transient's extent widened by
        `GlitchSeedPad`."""
        pad = self.settings.pad
        return any(r["extent_first"] - pad < t1 and t0 < r["extent_last"] + pad
                   for r in self.known())
