"""Root-sum-square strain amplitude (hrss) of a detector event.

An event's reconstruction (`detector_graph.event_waveform`) is the stitched
inverse transform of its surviving wavelet coefficients, in the whitened
stream's domain. In the online zero-phase search that stream is the conditioned
strain convolved with the whitening taps of the model in force
(`wdfUnitOnline.LookaheadWhitening`, `zero_phase_whitening.truncated_taps`),
not divided by the model's scale, so it is in strain units but coloured by the
filter's response ``R(f)``. The conditioning before it is the zero-phase
band-pass (`BandPassDownSampling`), whose power gain ``|H(f)|**2`` the data
receive.

The strain is brought back by dividing the reconstruction's spectrum by those
two responses, inside a band ``[f_lo, f_hi]``, and

    hrss = sqrt( integral h(t)**2 dt ) = sqrt( sum h_k**2 / fs )

is read on the result. It is the detector-frame hrss of the strain each
detector records (the antenna-pattern projection of the source), restricted to
the band, not the source-frame ``sqrt(hrss_+**2 + hrss_x**2)``.

Uncertainty and bias. Each surviving coefficient carries noise of variance
``sigma**2`` (the event's `sigma`, the median of its windows' robust noise
level, in the same units as the coefficients). With ``n`` independent surviving
coefficients (the members' survivors of the reconstructed window length, times
the windows' stride over their length, since overlapped windows describe the
same samples) and ``e`` the whitened energy ``sum w_k**2``:

- the noise floor ``hrss_noise = sqrt(n sigma**2 g / fs)``, with ``g`` the
  reconstruction's own strain-over-whitened energy ratio: the hrss pure noise
  would give on the same coefficients;
- the one-sigma error ``hrss_err = hrss * sqrt(4 max(e - n sigma**2, 0)
  sigma**2 + 2 n sigma**4) / (2 e)``: the spread of ``e`` for a signal plus
  Gaussian noise on ``n`` coefficients, carried to ``hrss``.

The estimate is biased in two opposite directions: noise adds about
``hrss_noise**2`` to ``hrss**2`` (upward, dominant near threshold), and the
signal energy left in the coefficients the threshold discarded, or outside the
band, is lost (downward, dominant for long, weak-per-coefficient signals).

Configuration (`HrssSettings.from_parameters`):

- ``HrssEstimate``: true to add the columns (default false: no column added);
- ``HrssLowFrequency``: "MinFrequency" (default; the run's `MinFrequency`, or
  the band-pass's `LowFrequencyCut` when that is zero or unset) or Hz;
- ``HrssHighFrequency``: "lowpass" (default; the band-pass's low-pass corner,
  `LowPassCorner` or `LowPassCornerFraction` of the decimated Nyquist) or Hz;
- ``HrssBandPassCorrection``: true (default) to divide by ``|H(f)|**2``.
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wdf.config.constants import (DB_AMPLITUDE, DECIBEL_BASE, NONCENTRAL_ENERGY_NOISE_POWER,
                                  NONCENTRAL_ENERGY_SIGNAL_FACTOR)
from wdf.config.defaults import defaults

_D = defaults("hrss")
#: The online zero-phase worker's defaults, whose `HoldOutsideBand` `_held` mirrors.
_ZEROPHASE = defaults("worker.online.zerophase")

#: Per-detector event columns `add_hrss` writes.
HRSS_COLUMNS = ("hrss", "hrss_err", "hrss_noise")
#: Per-detector event columns `add_hrss` writes beside them: the GPS span of
#: the reconstruction hrss is integrated over (its first window's start to its
#: last window's end).
HRSS_SPAN_COLUMNS = ("hrss_start", "hrss_end")
#: Network candidate column: root-sum-square of the detectors' hrss.
NETWORK_HRSS_COLUMN = "hrss_network_det"


class _Rates:
    """`par` seen with the stream's rates, for `BandPassDownSampling`."""

    def __init__(self, par, **rates):
        self._par, self._rates = par, rates

    def __getattr__(self, name):
        if name in self._rates:
            return self._rates[name]
        return getattr(self._par, name, None)


def _flag(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


@dataclass
class HrssSettings:
    """The hrss estimate's choices.

    :param active: `HrssEstimate`.
    :param low_frequency: `HrssLowFrequency` as given ("MinFrequency" or Hz).
    :param high_frequency: `HrssHighFrequency` as given ("lowpass" or Hz).
    :param bandpass_correction: `HrssBandPassCorrection`.
    :param par: the run configuration, read for the band-pass and the band.
    """

    active: bool = _D["HrssEstimate"]
    low_frequency: object = _D["HrssLowFrequency"]
    high_frequency: object = _D["HrssHighFrequency"]
    bandpass_correction: bool = _D["HrssBandPassCorrection"]
    par: object = field(default=None, repr=False)

    @classmethod
    def from_parameters(cls, par):
        """The settings a run asks for."""
        low = getattr(par, "HrssLowFrequency", None)
        high = getattr(par, "HrssHighFrequency", None)
        return cls(active=_flag(getattr(par, "HrssEstimate", None), _D["HrssEstimate"]),
                   low_frequency=_D["HrssLowFrequency"] if low is None else low,
                   high_frequency=_D["HrssHighFrequency"] if high is None else high,
                   bandpass_correction=_flag(getattr(par, "HrssBandPassCorrection", None),
                                             _D["HrssBandPassCorrection"]),
                   par=par)

    def bandpass(self, resampling):
        """The run's `BandPassDownSampling` at the stream rate `resampling`
        (the input rate is `sampling`, or `resampling` x `ResamplingFactor`)."""
        from wdf.processes.BandPassDownSampling import BandPassDownSampling

        factor = int(getattr(self.par, "ResamplingFactor", None) or _D["ResamplingFactor"])
        sampling = getattr(self.par, "sampling", None)
        sampling = int(round(resampling * factor)) if sampling is None else int(sampling)
        return BandPassDownSampling(_Rates(self.par, sampling=sampling,
                                           resampling=int(round(resampling)),
                                           ResamplingFactor=factor))

    def band(self, resampling):
        """``(f_lo, f_hi)``, Hz, at the stream rate `resampling`."""
        from wdf.config.Parameters import low_frequency_cut

        low = self.low_frequency
        if isinstance(low, str):
            if low.strip().lower() != "minfrequency":
                raise ValueError(f"HrssLowFrequency {low!r}: use 'MinFrequency' or Hz")
            value = getattr(self.par, "MinFrequency", None)
            low = float(value) if value else low_frequency_cut(self.par)
        high = self.high_frequency
        if isinstance(high, str):
            if high.strip().lower() != "lowpass":
                raise ValueError(f"HrssHighFrequency {high!r}: use 'lowpass' or Hz")
            high = float(self.bandpass(resampling).cutoff_frequency)
        low, high = float(low), float(high)
        if not 0.0 <= low < high <= 0.5 * resampling:
            raise ValueError(f"hrss band {low}-{high} Hz is not inside 0-{0.5 * resampling} Hz")
        return low, high


class ModelHistory:
    """The noise models a run applied, in switch order (`save_ar_models`),
    possibly from several files (one per chunk).

    :param models: list of model dicts (``switch_gps``, ``ar``, ...).
    :param attributes: the files' attributes (``WhiteningMode``,
        ``resampling``, ``ZeroPhaseDuration``).
    """

    def __init__(self, models, attributes):
        mode = str(attributes.get("WhiteningMode", "zerophase"))
        if mode != "zerophase":
            raise ValueError(f"hrss reads the zero-phase whitening; the models were applied in {mode!r}")
        self.models = sorted(models, key=lambda m: m["switch_gps"])
        if not self.models:
            raise ValueError("no noise model")
        self.switch = np.array([m["switch_gps"] for m in self.models], dtype=float)
        self.resampling = float(attributes["resampling"])
        self.duration = float(attributes["ZeroPhaseDuration"])
        self._taps = {}

    @classmethod
    def from_files(cls, paths):
        """Read and merge `ARModels-*.h5` files; a model present in more than
        one (same switch time and fit stretch) is kept once."""
        from wdf.processes.wdfUnitOnline import load_ar_models

        models, attributes, seen = [], None, set()
        for path in ([paths] if isinstance(paths, str) else paths):
            got, attrs = load_ar_models(path)
            if attributes is None:
                attributes = attrs
            for key in ("WhiteningMode", "resampling", "ZeroPhaseDuration"):
                if str(attrs.get(key)) != str(attributes.get(key)):
                    raise ValueError(f"{path}: {key} {attrs.get(key)!r} differs from {attributes.get(key)!r}")
            for m in got:
                key = (m["switch_gps"], m["fit_start"], m["fit_end"])
                if key not in seen:
                    seen.add(key)
                    models.append(m)
        if attributes is None:
            raise ValueError("no model file")
        return cls(models, attributes)

    def index_at(self, gps):
        """Index of the model in force at `gps` (the last switched on at or
        before it; the first one before every switch)."""
        return max(int(np.searchsorted(self.switch, float(gps), side="right")) - 1, 0)

    def taps(self, index, held=None):
        """The whitening taps of model `index` (`truncated_taps`)."""
        from wdf.processes.zero_phase_whitening import truncated_taps

        key = (int(index), None if not held else (tuple(held["band"]), float(held["blend"])))
        if key not in self._taps:
            extra = {} if not held else dict(band=tuple(held["band"]), blend=float(held["blend"]))
            self._taps[key] = truncated_taps(np.asarray(self.models[index]["ar"], dtype=float),
                                             self.duration, self.resampling, **extra)
        return self._taps[key]


def _held(settings, resampling):
    """``band`` and ``blend`` of `HoldOutsideBand`, as the online worker builds them."""
    par = settings.par
    if not _flag(getattr(par, "HoldOutsideBand", None), _ZEROPHASE["HoldOutsideBand"]):
        return None
    from wdf.processes.wdfUnitDSWorker import DEFAULT_PASSBAND_LEVEL_DB
    from wdf.processes.zero_phase_whitening import DEFAULT_BAND_BLEND_HZ

    level = getattr(par, "PassbandLevel", None)
    blend = getattr(par, "BandEdgeBlend", None)
    band = settings.bandpass(resampling).passband(DEFAULT_PASSBAND_LEVEL_DB if level is None else float(level))
    return dict(band=band, blend=DEFAULT_BAND_BLEND_HZ if blend is None else float(blend))


def grid_length(samples, taps):
    """FFT length `strain_from_whitened` works on: a power of two holding the
    series and `[hrss] grid_filter_lengths` filter lengths, so the inverse filter does not wrap onto
    the event."""
    return 1 << int(np.ceil(np.log2(int(samples) + _D["grid_filter_lengths"] * int(taps))))


def response_on_grid(taps, n, fs, bandpass=None):
    """The amplitude response the strain received on the `n`-point grid: the
    whitening taps' (symmetric, lag zero in the middle) times, with
    `bandpass`, its power gain ``|H|**2``.

    :return: numpy.ndarray -- ``n // 2 + 1`` values.
    """
    taps = np.asarray(taps, dtype=float)
    k = taps.size // 2
    shifted = np.zeros(int(n))
    shifted[:k + 1] = taps[k:]
    shifted[int(n) - k:] = taps[:k]
    response = np.abs(np.fft.rfft(shifted))
    if bandpass is not None:
        freq = np.fft.rfftfreq(int(n), 1.0 / fs)
        response = response * DECIBEL_BASE ** (bandpass.response_db(freq) / DB_AMPLITUDE)
    return response


def strain_from_whitened(samples, fs, taps, band, bandpass=None, response=None):
    """The strain whose whitening by `taps` gives `samples`, inside `band`.

    :param samples: whitened series (zero outside the event).
    :param fs: its sampling rate, Hz.
    :param taps: the symmetric whitening taps, lag zero in the middle.
    :param band: ``(f_lo, f_hi)``, Hz; the spectrum outside is set to zero.
    :param bandpass: `BandPassDownSampling` whose power gain is divided out,
        or None.
    :param response: `response_on_grid` on the `grid_length` grid, when
        already computed (it then replaces `taps` and `bandpass`).
    :return: tuple -- (strain series, whitened series in the same band), both
        on the zero-padded `grid_length` grid.
    """
    samples = np.asarray(samples, dtype=float)
    size = np.asarray(taps).size
    n = grid_length(samples.size, size)
    if response is None:
        response = response_on_grid(taps, n, fs, bandpass)
    padded = np.zeros(n)
    padded[2 * size:2 * size + samples.size] = samples
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    inside = (freq >= band[0]) & (freq <= band[1]) & (response > 0.0)
    spectrum = np.fft.rfft(padded)
    white = np.where(inside, spectrum, 0.0)
    strain = np.zeros_like(spectrum)
    strain[inside] = spectrum[inside] / response[inside]
    return np.fft.irfft(strain, n), np.fft.irfft(white, n)


def survivors(graph, labels):
    """Independent surviving coefficients of every event's reconstruction:
    the survivors of its members of the length `event_waveform` reconstructs,
    times their stride over their duration (overlapped windows describe the
    same samples).

    :return: function -- ``(cluster_id, window) -> count``.
    """
    from wdf.analysis.detector_graph import window_strides

    nodes = graph.nodes
    labels = np.asarray(labels)
    kept = np.fromiter((len(v) for v in nodes["wt_index"]), dtype=float, count=len(nodes))
    length = nodes["n_coeff"].to_numpy(float)
    fs = nodes["fs"].to_numpy(float)
    strides = window_strides(nodes)
    factor = np.array([min(strides[w] * f / w, 1.0) for w, f in zip(length, fs)])
    table = pd.DataFrame(dict(label=labels, length=length, count=kept * factor))
    total = table.groupby(["label", "length"])["count"].sum().to_dict()
    return lambda cid, window: float(total.get((int(cid), float(window)), 0.0))


def event_hrss(samples, fs, taps, band, sigma, n_survivors, bandpass=None, response=None):
    """hrss, its one-sigma error and its noise floor for one reconstruction.

    :param samples: the whitened reconstruction.
    :param fs: its rate, Hz.
    :param taps: the whitening taps of the model in force.
    :param band: ``(f_lo, f_hi)``, Hz.
    :param sigma: noise level of the coefficients, whitened units.
    :param n_survivors: independent surviving coefficients (`survivors`).
    :param bandpass: `BandPassDownSampling` to divide out, or None.
    :param response: `response_on_grid`, when already computed.
    :return: dict -- ``hrss``, ``hrss_err``, ``hrss_noise``.
    """
    strain, white = strain_from_whitened(samples, fs, taps, band, bandpass, response)
    e_strain = float(np.sum(strain ** 2))
    e_white = float(np.sum(white ** 2))
    hrss = np.sqrt(e_strain / fs)
    if not (e_white > 0.0 and np.isfinite(sigma) and sigma > 0.0):
        return dict(hrss=hrss, hrss_err=np.nan, hrss_noise=np.nan)
    gain = e_strain / e_white
    noise = float(n_survivors) * float(sigma) ** 2
    spread = np.sqrt(NONCENTRAL_ENERGY_SIGNAL_FACTOR * max(e_white - noise, 0.0) * sigma ** 2
                     + 2.0 * n_survivors * sigma ** NONCENTRAL_ENERGY_NOISE_POWER)
    return dict(hrss=hrss, hrss_err=hrss * spread / (2.0 * e_white),
                hrss_noise=np.sqrt(noise * gain / fs))


def add_hrss(events, graph, labels, history, settings, instant="gpsPeak"):
    """The event table with `HRSS_COLUMNS` added.

    :param events: one detector's events (`detector_events`), with
        ``cluster_id``, ``sigma`` and the `instant` column.
    :param graph: the detector's `DetectorGraph`.
    :param labels: its component labels.
    :param history: the detector's `ModelHistory`.
    :param settings: `HrssSettings`.
    :param instant: event column whose time picks the model in force.
    :return: pandas.DataFrame -- a copy of `events` with `HRSS_COLUMNS` and
        `HRSS_SPAN_COLUMNS`.
    """
    from wdf.analysis.detector_graph import event_waveform

    out = events.copy()
    values = {c: np.full(len(out), np.nan) for c in HRSS_COLUMNS + HRSS_SPAN_COLUMNS}
    if len(out):
        rate = history.resampling
        band = settings.band(rate)
        bandpass = settings.bandpass(rate) if settings.bandpass_correction else None
        held = _held(settings, rate)
        when = pd.to_numeric(out[instant]).to_numpy(float)
        sigma = pd.to_numeric(out["sigma"]).to_numpy(float)
        fs = float(pd.to_numeric(graph.nodes["fs"]).iloc[0])
        if abs(fs - rate) > _D["rate_tolerance"] * rate:
            raise ValueError(f"events at {fs} Hz, models at {rate} Hz")
        waves = event_waveform(graph, labels=labels)
        count = survivors(graph, labels)
        responses = {}
        for k, cid in enumerate(out["cluster_id"].astype(int).to_numpy()):
            got = waves.get(int(cid))
            if got is None:
                continue
            start, samples, window = got
            values["hrss_start"][k], values["hrss_end"][k] = start, start + samples.size / fs
            index = history.index_at(when[k])
            taps = history.taps(index, held)
            n = grid_length(samples.size, taps.size)
            if (index, n) not in responses:
                responses[(index, n)] = response_on_grid(taps, n, fs, bandpass)
            got = event_hrss(samples, fs, taps, band, sigma[k], count(cid, window), bandpass,
                             responses[(index, n)])
            for c in HRSS_COLUMNS:
                values[c][k] = got[c]
    for c in HRSS_COLUMNS + HRSS_SPAN_COLUMNS:
        out[c] = values[c]
    return out


def network_hrss(candidates, ifos):
    """`NETWORK_HRSS_COLUMN`: the root-sum-square of ``hrss_<ifo>`` over the
    detectors of each candidate (detector frame; NaN detectors skipped), when
    the table carries them."""
    columns = [f"hrss_{i}" for i in ifos if f"hrss_{i}" in candidates]
    if candidates.empty or not columns:
        return candidates
    values = candidates[columns].to_numpy(float)
    total = np.nansum(values ** 2, axis=1)
    total[np.all(~np.isfinite(values), axis=1)] = np.nan
    candidates[NETWORK_HRSS_COLUMN] = np.sqrt(total)
    return candidates
