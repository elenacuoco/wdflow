"""The noise model of a detector over a stretch, chosen before the search and
given to the zero-phase worker as `ARModelsFrom`.

The worker fits its own models by default (`NoiseModel` "worker"). With
`NoiseModel` set to a rule, the models are chosen here, written by
`save_noise_models` in the format the worker reuses, and the worker fits
nothing. The rules are tried in the order `NoiseModel`, then
`NoiseModelFallback`; the first whose whitening passes the check is kept:

- **"quiet"** (`quiet_model`): one autoregressive model (Burg, `ARorder`) for
  the whole stretch, fitted on the latest `learn` s that end before the
  stretch's first whitened second, lie within `NoiseLookback` s before it
  and, with `ARFitContext` s on each side, have every second good in the
  data-quality and injection masks (`good_seconds`: every bit of
  `NoiseDQBits` set in `NoiseDQChannel`, every bit of `NoiseInjectionBits`
  in `NoiseInjectionChannel`), finite strain and no transient second
  (`transient_seconds`). Candidates are scanned from the latest back: one
  with a bad second is replaced by the latest that ends before it, one with
  a transient by the latest that ends before its first transient second,
  less `ARFitExcludePad` (`quiet_stretch`). The model switches at the start
  of every run.
- **"piecewise"** (`piecewise_models`): per run, one model per piece of
  `learn` s of the searched stretch, fitted on the longest stretch without
  transient seconds inside the `learn` s centred on the piece, at least
  `ARFitMinimumLearn` s long, else on the `learn` s as they are.
- **"spectral"** (`spectral_model`): per run, one model from the power
  spectrum of the run (Welch, Hann windows of `NoiseSpectralSegment` s, half
  overlap, combined by `NoiseSpectralAverage`), held flat outside
  `NoiseSpectralHold` (`zero_phase_whitening.held_outside`), its
  autocorrelation turned into an autoregressive model of order `ARorder` by
  Levinson-Durbin (`zero_phase_whitening.levinson`); the rule fails when a
  reflection coefficient has modulus 1 or more.

A transient second is one whose power in any band of `NoiseBands` (Butterworth
of order `NoiseTransientOrder`, zero phase) exceeds `NoiseTransientRatio`
times the band's median over the stretch, padded by `ARFitExcludePad`.

**The whitening check** (`check_models`): the conditioned stream whitened by
the truncated zero-phase filter of each model (`TruncatedWhitening`,
`ZeroPhaseDuration`) and divided by its sigma, as the worker whitens it
(`waveform_completion.whitened_stretch`); per piece where one model applies,
the amplitude spectral density over white (Welch, Hann windows of
`NoiseCheckSegment` s, half overlap, `NoiseCheckAverage`) averaged in power
over each band of `NoiseBands`. A rule passes when every piece lies within
`NoiseCheckRange` in every band.

`choose_noise_model` runs the rules in order on the stream the worker will
whiten (with M4, the conditioned stream with the glitch atoms subtracted:
the rules fitted on the stretch itself then fit on the cleaned stream).
When no rule passes the detector's method is "failed": it is searched with
the models of the last stable rule tried and is left out of the network
events. A "quiet" model is fitted outside the stretch and does not depend on
the glitch subtraction; the order of the offline flavour is: the first
models (`initial_models`), the M4 census and fit through them, then
`choose_noise_model` on the cleaned stream.

Every default is in the table ``[noise_model]`` of wdf/config/defaults.toml.
`gwosc_frame_reader` reads GWOSC frame files (strain and masks) through
gwpy.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np

from wdf.config.defaults import defaults, parameter

_NM = defaults("noise_model")
_ZP = defaults("whitening.zero_phase")
SECTION = "noise_model"

#: The value of `NoiseModel` that leaves the models to the worker.
WORKER = "worker"
#: The rules, and those fitted on the stretch itself.
RULES = ("quiet", "piecewise", "spectral")
STRETCH_RULES = ("piecewise", "spectral")
#: `NoiseModel` values.
CHOICES = (WORKER,) + RULES
#: The method of a detector no rule whitened.
FAILED = "failed"


@dataclass
class NoiseModelSettings:
    """Every choice of the noise-model rules (table ``[noise_model]`` and the
    worker's `learn`, `ARorder`, `ARFitContext`, `ARFitExcludePad`,
    `ARFitMinimumLearn`).

    :param model: `NoiseModel`.
    :param fallback: `NoiseModelFallback`.
    :param lookback_s: `NoiseLookback`, s.
    :param dq_channel: `NoiseDQChannel`; :param injection_channel:
        `NoiseInjectionChannel` (``{ifo}`` filled in).
    :param dq_bits: `NoiseDQBits`; :param injection_bits: `NoiseInjectionBits`.
    :param transient_ratio: `NoiseTransientRatio`.
    :param transient_order: `NoiseTransientOrder`.
    :param transient_top_fraction: `NoiseTransientTopFraction`.
    :param bands: `NoiseBands`, ``((low, high), ...)`` Hz, or None.
    :param band_top_fraction: `NoiseBandTopFraction`.
    :param low_frequency: `LowFrequencyCut`, Hz.
    :param check_range: `NoiseCheckRange`.
    :param check_segment_s: `NoiseCheckSegment`, s.
    :param check_average: `NoiseCheckAverage`.
    :param spectral_segment_s: `NoiseSpectralSegment`, s.
    :param spectral_average: `NoiseSpectralAverage`.
    :param spectral_hold: `NoiseSpectralHold`, ``(low, high)`` Hz, or None.
    :param learn_s: `learn`, s; :param order: `ARorder`;
        :param context_s: `ARFitContext`, s.
    :param exclude_pad_s: `ARFitExcludePad`, s (default `GlitchSeedPad`).
    :param minimum_learn_s: `ARFitMinimumLearn`, s (default
        `ARFitMinimumLearnFraction` of `learn`).
    """

    model: str = _NM["NoiseModel"]
    fallback: tuple = _NM["NoiseModelFallback"]
    lookback_s: float = _NM["NoiseLookback"]
    dq_channel: str = _NM["NoiseDQChannel"]
    injection_channel: str = _NM["NoiseInjectionChannel"]
    dq_bits: int = _NM["NoiseDQBits"]
    injection_bits: int = _NM["NoiseInjectionBits"]
    transient_ratio: float = _NM["NoiseTransientRatio"]
    transient_order: int = _NM["NoiseTransientOrder"]
    transient_top_fraction: float = _NM["NoiseTransientTopFraction"]
    bands: tuple = _NM["NoiseBands"]
    band_top_fraction: float = _NM["NoiseBandTopFraction"]
    low_frequency: float = None
    check_range: tuple = _NM["NoiseCheckRange"]
    check_segment_s: float = _NM["NoiseCheckSegment"]
    check_average: str = _NM["NoiseCheckAverage"]
    spectral_segment_s: float = _NM["NoiseSpectralSegment"]
    spectral_average: str = _NM["NoiseSpectralAverage"]
    spectral_hold: tuple = _NM["NoiseSpectralHold"]
    learn_s: float = None
    order: int = None
    context_s: float = None
    exclude_pad_s: float = None
    minimum_learn_s: float = None

    #: The run parameter each field of the table is read from.
    KEYS = dict(model="NoiseModel", fallback="NoiseModelFallback", lookback_s="NoiseLookback",
                dq_channel="NoiseDQChannel", injection_channel="NoiseInjectionChannel",
                dq_bits="NoiseDQBits", injection_bits="NoiseInjectionBits",
                transient_ratio="NoiseTransientRatio", transient_order="NoiseTransientOrder",
                transient_top_fraction="NoiseTransientTopFraction", bands="NoiseBands",
                band_top_fraction="NoiseBandTopFraction", check_range="NoiseCheckRange",
                check_segment_s="NoiseCheckSegment", check_average="NoiseCheckAverage",
                spectral_segment_s="NoiseSpectralSegment", spectral_average="NoiseSpectralAverage",
                spectral_hold="NoiseSpectralHold")

    def __post_init__(self):
        self.model = str(self.model)
        self.fallback = tuple(str(r) for r in (self.fallback or ()))
        if self.model not in CHOICES:
            raise ValueError(f"NoiseModel {self.model!r}: use one of {CHOICES}")
        bad = [r for r in self.fallback if r not in STRETCH_RULES]
        if bad:
            raise ValueError(f"NoiseModelFallback {bad}: the fallback rules are {STRETCH_RULES}")
        if self.check_average not in ("mean", "median"):
            raise ValueError(f"NoiseCheckAverage {self.check_average!r}: 'mean' or 'median'")
        if self.spectral_segment_s is None:
            self.spectral_segment_s = _ZP["spectrum_segment_s"]
        if self.spectral_average is None:
            self.spectral_average = _ZP["spectrum_average"]
        if self.bands is not None:
            self.bands = tuple((float(lo), float(hi)) for lo, hi in self.bands)
        self.check_range = tuple(float(v) for v in self.check_range)

    @classmethod
    def from_parameters(cls, par):
        """The settings a run asks for, defaults filled in and written back.

        :param par: run configuration; it receives every value of the table
            used, and `ARFitExcludePad`, `ARFitMinimumLearn` when it resolves
            them.
        :return: NoiseModelSettings
        """
        from wdf.config.Parameters import low_frequency_cut

        values = {f: parameter(par, key, SECTION) for f, key in cls.KEYS.items()}
        learn = getattr(par, "learn", None)
        pad = parameter(par, "ARFitExcludePad", "worker.online")
        if pad is None:
            pad = parameter(par, "GlitchSeedPad", "glitch_subtraction")
        shortest = parameter(par, "ARFitMinimumLearn", "worker.online")
        if shortest is None and learn is not None:
            shortest = parameter(par, "ARFitMinimumLearnFraction", "worker.online") * float(learn)
        out = cls(**values, low_frequency=low_frequency_cut(par),
                  learn_s=None if learn is None else float(learn),
                  order=None if getattr(par, "ARorder", None) is None else int(par.ARorder),
                  context_s=float(parameter(par, "ARFitContext", "worker.online")),
                  exclude_pad_s=float(pad), minimum_learn_s=None if shortest is None else float(shortest))
        for f, key in cls.KEYS.items():
            value = getattr(out, f)
            setattr(par, key, [list(v) if isinstance(v, tuple) else v for v in value]
                    if isinstance(value, tuple) else value)
        par.ARFitExcludePad = out.exclude_pad_s
        if out.minimum_learn_s is not None:
            par.ARFitMinimumLearn = out.minimum_learn_s
        return out

    def rules(self):
        """The rules in the order they are tried; empty for "worker"."""
        if self.model == WORKER:
            return ()
        return (self.model,) + tuple(r for r in self.fallback if r != self.model)

    def bands_at(self, rate):
        """`NoiseBands`, or the octaves from `LowFrequencyCut` with the last
        ending at `NoiseBandTopFraction` of the Nyquist frequency of `rate`."""
        if self.bands is not None:
            return self.bands
        top = float(self.band_top_fraction) * 0.5 * float(rate)
        low = float(self.low_frequency)
        if not 0.0 < low < top:
            raise ValueError(f"no band between LowFrequencyCut {low} Hz and {top} Hz")
        out = []
        while low < top:
            out.append((low, min(2 * low, top)))
            low = 2 * low
        return tuple(out)


@dataclass
class NoiseRun:
    """One run of valid data of a detector, as the rules read it.

    :param conditioned: ``(samples, GPS of the first, rate)``: the stream the
        worker whitens (`waveform_completion.conditioned_stretch`; with M4,
        the glitch atoms subtracted).
    :param start: GPS the first model whitens from (the worker's first
        whitened second, its warm-up included).
    :param end: GPS of the end of the run's stretch.
    :param bounds: ``(start, end)`` GPS the check reads (the searched span).
    """

    conditioned: tuple
    start: float
    end: float
    bounds: tuple


@dataclass
class NoiseChoice:
    """What `choose_noise_model` decided for one detector.

    :param method: the rule kept, `FAILED`, `WORKER`, or "unavailable" (no
        rule gave a model: the detector cannot be searched).
    :param applied: the rule whose models are given to the worker.
    :param models: per run, the list of model dicts (`save_noise_models`).
    :param attempts: one dict per rule tried: ``rule``, ``status``,
        ``passes`` and the check's levels per band.
    :param check: one dict per rule, run, piece and band.
    :param fits: one dict per model fitted.
    :param spectra: per rule, the pieces' ``f`` and ``asd`` over white.
    """

    method: str
    applied: str = None
    models: list = None
    attempts: list = field(default_factory=list)
    check: list = field(default_factory=list)
    fits: list = field(default_factory=list)
    spectra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ masks and transients
def good_seconds(dq, injection, dq_bits, injection_bits):
    """Per second, whether every bit of `dq_bits` is set in the data-quality
    mask and every bit of `injection_bits` in the injection mask; a missing
    (non-finite) value is not good."""
    dq = np.asarray(dq, dtype=float)
    injection = np.asarray(injection, dtype=float)
    good = np.zeros(dq.size, bool)
    ok = np.isfinite(dq) & np.isfinite(injection)
    good[ok] = (((dq[ok].astype(np.int64) & int(dq_bits)) == int(dq_bits))
                & ((injection[ok].astype(np.int64) & int(injection_bits)) == int(injection_bits)))
    return good


def transient_seconds(samples, start, rate, settings):
    """The transient seconds of a conditioned stretch.

    :return: tuple -- (GPS of the loud seconds, per second whether it is left
        out with its pad, GPS of every second).
    """
    from scipy.signal import butter, sosfiltfilt

    y = np.asarray(samples, dtype=float)
    n_sec = int(y.size // rate)
    top = float(settings.transient_top_fraction) * 0.5 * rate
    power = np.stack([(sosfiltfilt(butter(int(settings.transient_order), [lo, min(hi, top)], btype="band",
                                          fs=rate, output="sos"), y)[:int(n_sec * rate)] ** 2)
                      .reshape(n_sec, int(rate)).mean(axis=1) for lo, hi in settings.bands_at(rate)])
    loud = (power / np.median(power, axis=1, keepdims=True) > float(settings.transient_ratio)).any(axis=0)
    secs = float(start) + np.arange(n_sec)
    pad = float(settings.exclude_pad_s)
    bad = np.zeros(n_sec, bool)
    for k in np.flatnonzero(loud):
        bad |= (secs + 1 >= secs[k] - pad) & (secs <= secs[k] + 1 + pad)
    return secs[loud], bad, secs


def quiet_stretch(good, origin, before, settings, inspect=None):
    """The latest fit stretch whose masks are good, scanned from `before`
    back.

    :param good: per second from `origin`, whether the masks are good
        (`good_seconds`).
    :param origin: GPS of the first second of `good`.
    :param before: the stretch must end at or before this GPS.
    :type settings: NoiseModelSettings
    :param settings: `learn_s`, `context_s` and `lookback_s` are read.
    :param inspect: ``inspect(a, end)`` of a stretch with good masks: None to
        accept it, else ``(end, note)``, the latest end to try next and why.
    :return: tuple -- ``(a, end)`` or None, and a record (``candidates``,
        ``rejected``: the notes).
    """
    learn, ctx, look = float(settings.learn_s), int(settings.context_s), float(settings.lookback_s)
    good = np.asarray(good, bool)
    cum = np.concatenate([[0], np.cumsum(good)])
    end = int(np.floor(before))
    record = dict(candidates=0, rejected=[])
    while end - learn >= before - look:
        a = int(end - learn)
        i0, i1 = a - ctx - int(origin), end + ctx - int(origin)
        if i0 < 0:
            break
        if i1 > good.size:                  # the masks end earlier: end where they end
            end = int(origin) + good.size - ctx
            continue
        if cum[i1] - cum[i0] < i1 - i0:
            bad_last = i0 + int(np.flatnonzero(~good[i0:i1])[-1])
            end = min(end - 1, bad_last + int(origin) - ctx)
            continue
        record["candidates"] += 1
        verdict = inspect(a, end) if inspect is not None else None
        if verdict is None:
            return (a, end), record
        nxt, note = verdict
        record["rejected"].append(note)
        end = min(end - 1, int(nxt))
    return None, record


def quiet_model(par, read, ifo, before, settings=None):
    """The "quiet" rule: an autoregressive model fitted on the latest clean
    stretch before `before`.

    :param par: the run configuration (`ARorder`, `learn`, `len`, the
        band-pass keys).
    :param read: ``read(channel, a, b) -> (samples, rate)`` over ``[a, b)``,
        NaN where no data, ``(None, None)`` when nothing is held
        (`gwosc_frame_reader`).
    :param ifo: the detector; :param before: GPS the stretch must end by.
    :return: tuple -- (the model, scale first, or None; the record).
    """
    from wdf.analysis.waveform_completion import conditioned_stretch
    from wdf.processes.BandPassDownSampling import BandPassDownSampling
    from wdf.processes.wdfUnitOnline import fit_autoregressive

    settings = settings or NoiseModelSettings.from_parameters(par)
    learn, ctx, blk = float(settings.learn_s), int(settings.context_s), float(par.len)
    look = float(settings.lookback_s)
    a_lb, b_lb = int(np.floor(before - look - ctx)), int(np.ceil(before + ctx))
    record = dict(detector=ifo, rule="quiet", lookback_s=look, dq_bits=int(settings.dq_bits),
                  injection_bits=int(settings.injection_bits), candidates=0, rejected="")
    dq, _ = read(settings.dq_channel.format(ifo=ifo), a_lb, b_lb)
    injection, _ = read(settings.injection_channel.format(ifo=ifo), a_lb, b_lb)
    if dq is None or injection is None:
        return None, dict(record, status="no mask in the lookback")
    good = good_seconds(dq, injection, settings.dq_bits, settings.injection_bits)
    record["good_s"] = int(good[:int(before - a_lb)].sum())
    fitted = {}

    def inspect(a, end):
        x0 = a - ctx - int(2 * blk)
        x, fs = read(par.channel, x0, end + ctx + int(2 * blk))
        if x is None:
            return end - 1, f"{a}-{end}: no strain"
        if not np.all(np.isfinite(x)):
            last = x0 + np.flatnonzero(~np.isfinite(x))[-1] / fs
            return int(np.floor(last)) - ctx - int(2 * blk), f"{a}-{end}: missing strain"
        y, y0, r = conditioned_stretch(x, fs, x0, BandPassDownSampling(par), blk)
        loud, _, _ = transient_seconds(y, y0, r, settings)
        loud = loud[(loud >= a) & (loud < end)]
        if loud.size:
            return (int(np.floor(loud.min() - float(settings.exclude_pad_s))),
                    f"{a}-{end}:" + ",".join(f"{t:.0f}" for t in loud))
        j0, j1 = int(round((a - y0) * r)), int(round((end - y0) * r))
        fitted["ar"] = np.asarray(fit_autoregressive(y[j0:j1], float(a), r, int(settings.order)), dtype=float)
        return None

    found, rec = quiet_stretch(good, a_lb, before, settings, inspect)
    record.update(candidates=rec["candidates"], rejected="; ".join(rec["rejected"]))
    if found is None:
        return None, dict(record, status=f"no {learn:g} s stretch with good masks and no transient "
                                         f"in the last {look:g} s")
    a, end = found
    record.update(status="fitted", fit_start=float(a), fit_end=float(end), fit_s=float(end - a),
                  seconds_before=float(before - end), ar_sigma=float(fitted["ar"][0]))
    return fitted["ar"], record


# ------------------------------------------------------------------ the rules on the stretch
def piecewise_models(run, settings):
    """The "piecewise" rule on one run.

    :type run: NoiseRun
    :return: tuple -- (models, rows), one model per piece of `learn` s.
    """
    from wdf.processes.wdfUnitOnline import fit_autoregressive

    y, y0, r = run.conditioned
    y = np.asarray(y, dtype=float)
    learn = float(settings.learn_s)
    s0, s1 = float(run.start), float(run.end)
    y_end = y0 + y.size / r
    _, bad, secs = transient_seconds(y, y0, r, settings)
    models, rows = [], []
    for sw in np.arange(s0, s1, learn):
        c = sw + min(learn, s1 - sw) / 2
        a = float(np.clip(c - learn / 2, max(y0, s0 - learn / 2), min(y_end, s1) - learn))
        win = (secs >= a) & (secs + 1 <= a + learn)
        clean = win & ~bad
        edges = np.flatnonzero(np.diff(np.concatenate(([0], clean.view(np.int8), [0]))))
        runs = [(secs[e0], secs[e1 - 1] + 1) for e0, e1 in zip(edges[::2], edges[1::2])]
        runs = [q for q in runs if q[1] - q[0] >= float(settings.minimum_learn_s)]
        f0, f1 = (max(runs, key=lambda q: (q[1] - q[0], -abs((q[0] + q[1]) / 2 - c)))
                  if runs and (~clean & win).any() else (a, a + learn))
        j0 = int(round((f0 - y0) * r))
        ar = np.asarray(fit_autoregressive(y[j0:j0 + int(round((f1 - f0) * r))], f0, r, int(settings.order)),
                        dtype=float)
        models.append(dict(switch_gps=float(sw), fit_start=float(f0), fit_end=float(f1), sigma=np.nan, ar=ar))
        rows.append(dict(rule="piecewise", run_start=s0, switch=float(sw), fit_start=float(f0),
                         fit_end=float(f1), fit_s=float(f1 - f0), transient_s=int((win & bad).sum()),
                         ar_sigma=float(ar[0])))
    return models, rows


def spectral_model(run, settings):
    """The "spectral" rule on one run.

    :type run: NoiseRun
    :return: tuple -- (models, stable, row): one model switching at the run's
        start; stable when every reflection coefficient has modulus below 1.
    """
    from scipy.signal import welch

    from wdf.processes.zero_phase_whitening import held_outside, levinson

    y, y0, r = run.conditioned
    y = np.asarray(y, dtype=float)
    a, b = max(float(run.start), y0), min(float(run.end), y0 + y.size / r)
    n = int(round(float(settings.spectral_segment_s) * r))
    f, psd = welch(y[int(round((a - y0) * r)):int(round((b - y0) * r))], fs=r, window="hann", nperseg=n,
                   noverlap=n // 2, average=str(settings.spectral_average))
    hold = (tuple(float(v) for v in settings.spectral_hold) if settings.spectral_hold
            else (float(settings.low_frequency), 0.5 * r))
    acf = np.fft.irfft(held_outside(f, psd, hold) * r / 2, 2 * (f.size - 1))
    poly, err, refl = levinson(acf, int(settings.order))
    k_max = float(np.max(np.abs(refl)))
    stable = bool(np.isfinite(k_max) and k_max < 1 and err > 0)
    ar = np.concatenate([[np.sqrt(err) if err > 0 else np.nan], -poly[1:]])
    row = dict(rule="spectral", run_start=float(run.start), switch=float(run.start), fit_start=float(a),
               fit_end=float(b), fit_s=float(b - a), ar_sigma=float(ar[0]),
               spectral_segment_s=float(settings.spectral_segment_s),
               spectral_average=str(settings.spectral_average), hold_hz=f"{hold[0]:g}-{hold[1]:g}",
               max_reflection=k_max, stable=stable)
    return [dict(switch_gps=float(run.start), fit_start=float(a), fit_end=float(b), sigma=np.nan, ar=ar)], \
        stable, row


# ------------------------------------------------------------------ the check
def model_sigmas(models, par):
    """`models` with the sigma of their truncated zero-phase whitening
    (`TruncatedWhitening` of `ZeroPhaseDuration` at `resampling`)."""
    from wdf.processes.zero_phase_whitening import TruncatedWhitening

    return [dict(m, sigma=float(TruncatedWhitening(np.asarray(m["ar"], dtype=float), 1, 0,
                                                   duration=float(par.ZeroPhaseDuration),
                                                   sampling=float(par.resampling)).sigma)) for m in models]


def check_pieces(samples, first, rate, switch, bounds, settings):
    """The check of a whitened stream over sigma, one piece per model in
    force within `bounds`.

    :return: tuple -- (spectra, rows): per piece ``start``, ``end``, ``f``,
        ``asd`` (over white); per piece and band ``asd_over_white`` and
        ``passes``.
    """
    from scipy.signal import welch

    y = np.asarray(samples, dtype=float)
    seg = float(settings.check_segment_s)
    lo_range, hi_range = settings.check_range
    lo_b, hi_b = bounds
    spectra, rows = [], []
    switch = np.asarray(switch, dtype=float)
    for k, sw in enumerate(switch):
        a = max(sw, lo_b, first)
        b = min(switch[k + 1] if k + 1 < switch.size else hi_b, hi_b, first + y.size / rate)
        i0, i1 = int(round((a - first) * rate)), int(round((b - first) * rate))
        if i1 - i0 < 2 * seg * rate:
            continue
        n = int(seg * rate)
        f, pxx = welch(y[i0:i1], fs=rate, window="hann", nperseg=n, noverlap=n // 2,
                       average=str(settings.check_average))
        asd = np.sqrt(pxx * rate / 2)
        spectra.append(dict(start=float(a), end=float(b), f=f, asd=asd))
        for lo, hi in settings.bands_at(rate):
            m = (f >= lo) & (f < min(hi, rate / 2))
            level = float(np.sqrt(np.mean(asd[m] ** 2)))
            rows.append(dict(piece_start=float(a), piece_end=float(b), band_hz=f"{lo:g}-{hi:g}",
                             asd_over_white=level, passes=bool(lo_range <= level <= hi_range)))
    return spectra, rows


def check_models(run, models, par, settings):
    """The whitening check of one run's models (they must carry their sigma,
    `model_sigmas`).

    :return: tuple -- (spectra, rows) of `check_pieces`.
    """
    from wdf.analysis.hrss import ModelHistory
    from wdf.analysis.waveform_completion import whitened_stretch

    history = ModelHistory(models, dict(WhiteningMode="zerophase", resampling=float(par.resampling),
                                        ZeroPhaseDuration=float(par.ZeroPhaseDuration)))
    y, y0, r = run.conditioned
    first, w, rate = whitened_stretch(np.asarray(y, dtype=float), r, y0, history, normalise=True)
    return check_pieces(np.asarray(w, dtype=float), float(first), float(rate), history.switch, run.bounds,
                        settings)


def initial_models(runs, par, settings, quiet=None):
    """The models the M4 census whitens through before the choice: the
    "quiet" model when `NoiseModel` is "quiet" and it was found, else those
    of the first rule fitted on the stretch.

    :return: tuple -- (rule, per run the models), or (None, None) with no
        rule.
    """
    for rule in settings.rules():
        if rule == "quiet":
            if quiet is None:
                continue
            return rule, [model_sigmas(_quiet_models(quiet, run), par) for run in runs]
        fitted = [_fit(rule, run, settings) for run in runs]
        if all(ok for _, ok, _ in fitted):
            return rule, [model_sigmas(m, par) for m, _, _ in fitted]
    return None, None


def _quiet_models(ar, run):
    return [dict(switch_gps=float(run.start), fit_start=np.nan, fit_end=np.nan, sigma=np.nan,
                 ar=np.asarray(ar, dtype=float))]


def _fit(rule, run, settings):
    if rule == "piecewise":
        models, rows = piecewise_models(run, settings)
        return models, True, rows
    models, stable, row = spectral_model(run, settings)
    return models, stable, [row]


def choose_noise_model(runs, par, settings=None, quiet=None):
    """The rules in order, the first whose whitening passes the check kept.

    :param runs: the detector's runs (`NoiseRun`), each with the stream the
        worker will whiten.
    :param par: run configuration (`ZeroPhaseDuration`, `resampling`).
    :param quiet: ``(model, record)`` of `quiet_model`, or None when it was
        not run; read when "quiet" is among the rules.
    :return: NoiseChoice
    """
    settings = settings or NoiseModelSettings.from_parameters(par)
    rules = settings.rules()
    if not rules:
        return NoiseChoice(method=WORKER)
    choice = NoiseChoice(method=FAILED)
    for rule in rules:
        if rule == "quiet":
            ar, record = quiet if quiet is not None else (None, dict(status="not run"))
            if ar is None:
                choice.attempts.append(dict(rule=rule, status=str(record.get("status")), passes=False))
                continue
            models = []
            for run in runs:
                m = _quiet_models(ar, run)
                m[0].update(fit_start=float(record.get("fit_start", np.nan)),
                            fit_end=float(record.get("fit_end", np.nan)))
                models.append(m)
            choice.fits.append(dict(record))
        else:
            fitted = [_fit(rule, run, settings) for run in runs]
            if not all(ok for _, ok, _ in fitted):
                choice.attempts.append(dict(rule=rule, status="unstable: a reflection coefficient of "
                                                              "modulus 1 or more", passes=False))
                continue
            models = [m for m, _, _ in fitted]
            choice.fits += [row for _, _, rows in fitted for row in rows]
        models = [model_sigmas(m, par) for m in models]
        choice.models, choice.applied = models, rule
        spectra, rows = [], []
        for run, m in zip(runs, models):
            s, r = check_models(run, m, par, settings)
            spectra += s
            rows += r
        passes = bool(rows) and all(r["passes"] for r in rows)
        levels = {}
        for r in rows:
            levels.setdefault(r["band_hz"], []).append(r["asd_over_white"])
        choice.attempts.append(dict(rule=rule, status="checked", passes=passes, pieces=len(spectra),
                                    failing_pieces=len({(r["piece_start"], r["piece_end"])
                                                        for r in rows if not r["passes"]}),
                                    asd_over_white={b: dict(min=min(v), max=max(v)) for b, v in levels.items()}))
        choice.check += [dict(r, rule=rule) for r in rows]
        choice.spectra[rule] = spectra
        if passes:
            choice.method = rule
            return choice
    if choice.models is None:
        choice.method = "unavailable"
    return choice


def save_noise_models(path, models, par):
    """Write models in the format `ARModelsFrom` reads (`save_ar_models`),
    for the zero-phase worker of `par`.

    :param models: the model dicts (``switch_gps``, ``fit_start``,
        ``fit_end``, ``sigma``, ``ar``), of one or several runs.
    :return: str -- `path`.
    :raises ValueError: when the run is not in the "zerophase" whitening the
        check reads.
    """
    from wdf.processes.wdfUnitOnline import save_ar_models

    mode = str(parameter(par, "WhiteningMode", "worker.online"))
    if mode != "zerophase":
        raise ValueError(f"the noise-model rules are checked on the zero-phase whitening; WhiteningMode is {mode!r}")
    save_ar_models(path, sorted(models, key=lambda m: m["switch_gps"]),
                   dict(ARorder=int(par.ARorder), WhiteningMode=mode, resampling=float(par.resampling),
                        ZeroPhaseDuration=float(par.ZeroPhaseDuration), itf=str(getattr(par, "itf", ""))))
    return path


# ------------------------------------------------------------------ GWOSC frames
def gwosc_frame_reader(pattern):
    """A reader of GWOSC frame files through gwpy.

    :param pattern: glob of the files with ``{ifo}`` for the detector, e.g.
        ``"/data/O4b/{ifo}/*.gwf"``; the files are named
        ``<...>-<gps>-<seconds>.gwf``.
    :return: callable -- ``read(channel, a, b) -> (samples, rate)`` over
        ``[a, b)``, NaN where no file holds them, ``(None, None)`` when none
        does; the detector is the channel's prefix.
    """
    def files(ifo):
        out = {}
        for p in glob.glob(pattern.format(ifo=ifo)):
            m = re.search(r"-(\d+)-(\d+)\.gwf$", os.path.basename(p))
            if m:
                out[int(m.group(1))] = (p, int(m.group(2)))
        return out

    def read(channel, a, b):
        from gwpy.timeseries import TimeSeries

        parts, rate = [], None
        for g, (path, n) in sorted(files(channel.split(":")[0]).items()):
            lo, hi = max(g, a), min(g + n, b)
            if lo < hi:
                ts = TimeSeries.read(path, channel, start=lo, end=hi)
                rate = float(ts.sample_rate.value)
                parts.append((lo, np.asarray(ts.value, dtype=float)))
        if rate is None:
            return None, None
        out = np.full(int(round((b - a) * rate)), np.nan)
        for lo, v in parts:
            i = int(round((lo - a) * rate))
            out[i:i + v.size] = v[:max(out.size - i, 0)]
        return out, rate

    return read
