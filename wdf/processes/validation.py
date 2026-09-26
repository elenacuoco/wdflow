"""The check a detector's whitened stream must pass before it is searched.

The search's threshold means the same thing at every frequency and at every
time only on a stream that is white, Gaussian and stationary, on the scale the
search divides by. Each of those is a property the conditioning is meant to
give the stream and can fail to give it, band by band: a noise model that
misfits one octave, a line left in, a glitch the gates did not catch, noise
that is not Gaussian in one band whatever the model. The check measures the
whitened stream the search is about to read -- gated, on the search's scale,
over the whole stretch it will search -- and states, per detector, which of
its criteria hold and which do not.

The criteria, each read in every octave the search reads (`octave_bands`, from
the detector's search low frequency to the Nyquist frequency):

- **White.** The power of the stream over the octave, as the median of short
  periodograms, divided by the power of unit white noise, is within
  `power_tolerance` of one. The median is read rather than the mean so that a
  transient does not lift it; transients are the third criterion's.
- **Gaussian.** The kurtosis of the stream band-passed into the octave, in
  windows of `window_s` seconds, has its median within `kurtosis_tolerance` of
  the median the same estimator gives on Gaussian white noise of the same
  length passed through the same filter. Comparing with the estimator's own
  value on Gaussian noise, rather than with three, removes its bias on a
  finite, band-limited window by construction: the bias is the same in both.
  The band above the conditioning's high edge, which the band-pass empties and
  the whitening lifts back up, is read as well. Windows a gate touches are left
  out, since they are not searched there.
- **Clean.** The transients the census flags, each widened by one analysis
  window on either side, together with the gated stretches and their tapers,
  cover no more than `transient_tolerance` of the stretch.
- **Stationary.** In each third of the stretch the power of every octave is
  within `stationarity_tolerance` of its power over the whole stretch.

The report carries the detector's sensitivity as well, as the angle-averaged
range for a binary neutron star, read off the strain's spectrum; it is
reported, not tested.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wdf.processes.gating import band_stream, merged

#: Largest departure of an octave's power from white, as a fraction.
POWER_TOLERANCE = 0.10
#: Largest departure of an octave's median window kurtosis from Gaussian noise's.
KURTOSIS_TOLERANCE = 0.10
#: Largest fraction of the stretch covered by transients and gates.
TRANSIENT_TOLERANCE = 0.01
#: Largest departure of an octave's power in a third of the stretch from its
#: power over the whole stretch, as a fraction.
STATIONARITY_TOLERANCE = 0.10

#: Seconds per periodogram of the power estimates.
POWER_SEGMENT_S = 1.0
#: Seconds per window of the kurtosis estimates.
KURTOSIS_WINDOW_S = 8.0

# Constants of the inspiral range (SI), as LIGO-T030276 and gwpy use them.
_C = 299792458.0
_G = 6.6743e-11
_SOLAR_MASS = 1.988409870698051e30
_MEGAPARSEC = 3.085677581491367e22


class ConditioningRejected(RuntimeError):
    """A detector's whitened stream failed the check, and was not searched.

    :param report: the `ValidationReport` that failed.
    """

    def __init__(self, report):
        super().__init__(report.message())
        self.report = report


def band_power(samples, rate, bands, segment_s=POWER_SEGMENT_S):
    """The power of a stream in each band, as a fraction of unit white noise's.

    The mean, over the band's bins, of the median of periodograms `segment_s`
    long, divided by `2 / rate`, the one-sided spectral density of white noise
    of unit variance.

    :type samples: numpy.ndarray
    :param samples: the stream, on the scale it is judged on.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :type bands: tuple
    :param bands: `(low, high)` per band, Hz.
    :type segment_s: float
    :param segment_s: seconds per periodogram.
    :return: numpy.ndarray -- one value per band.
    """
    from scipy.signal import welch

    nperseg = int(round(segment_s * rate))
    frequency, psd = welch(np.asarray(samples, dtype=float), fs=float(rate),
                           window="hann", nperseg=nperseg, noverlap=nperseg // 2,
                           average="median", detrend="constant")
    return np.array([np.mean(psd[(frequency >= low) & (frequency < high)])
                     for low, high in bands]) * float(rate) / 2.0


def window_kurtosis(samples, rate, band, window_s=KURTOSIS_WINDOW_S, keep=None):
    """The median kurtosis of one band of a stream, over windows of it.

    :type samples: numpy.ndarray
    :param samples: the stream.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :type band: tuple
    :param band: `(low, high)`, Hz; a high edge at the Nyquist frequency keeps
        everything above `low`.
    :type window_s: float
    :param window_s: seconds per window.
    :type keep: numpy.ndarray or None
    :param keep: one flag per window, False for a window to leave out; all
        windows when None.
    :return: float -- the median over the windows of the Pearson kurtosis,
        three for a Gaussian of infinite length.
    """
    from scipy.stats import kurtosis

    size = int(round(window_s * rate))
    stream = band_stream(samples, rate, band)
    count = stream.size // size
    values = kurtosis(stream[:count * size].reshape(count, size), axis=1, fisher=False)
    return float(np.median(values if keep is None else values[keep]))


def gaussian_window_kurtosis(size, rate, band, window_s=KURTOSIS_WINDOW_S, seed=0):
    """What `window_kurtosis` gives on Gaussian white noise of the same length.

    :type size: int
    :param size: samples of the stream it stands in for.
    :type rate: float
    :param rate: sampling rate, Hz.
    :type band: tuple
    :param band: `(low, high)`, Hz.
    :type window_s: float
    :param window_s: seconds per window.
    :type seed: int
    :param seed: seed of the noise, so that the value is reproducible.
    :return: float
    """
    noise = np.random.default_rng(seed).standard_normal(int(size))
    return window_kurtosis(noise, rate, band, window_s)


def bns_range(frequency, psd, snr=8.0, mass1=1.4, mass2=1.4, fmin=10.0, fmax=None):
    """The angle-averaged distance to which a compact binary inspiral is seen.

    The sensitive distance of LIGO-T030276, as gwpy's `sensemon_range` computes
    it: the integral of `f^(-7/3) / S(f)` from `fmin` to the innermost stable
    circular orbit of the binary (or `fmax`, if lower), scaled by the chirp
    mass and by the signal-to-noise ratio a detection needs, and averaged over
    the sky and the orientation.

    :type frequency: numpy.ndarray
    :param frequency: frequencies of `psd`, Hz.
    :type psd: numpy.ndarray
    :param psd: one-sided power spectral density of the strain, 1/Hz.
    :type snr: float
    :param snr: signal-to-noise ratio at the range.
    :type mass1: float
    :param mass1: first component mass, solar masses.
    :type mass2: float
    :param mass2: second component mass, solar masses.
    :type fmin: float
    :param fmin: lowest frequency integrated, Hz.
    :type fmax: float or None
    :param fmax: highest frequency integrated, Hz; the innermost stable orbit's
        frequency when None or higher.
    :return: float -- the range, Mpc.
    """
    m1, m2 = mass1 * _SOLAR_MASS, mass2 * _SOLAR_MASS
    total = m1 + m2
    chirp = (m1 * m2) ** 0.6 / total ** 0.2
    isco = _C ** 3 / (_G * 6.0 ** 1.5 * np.pi * total)
    top = isco if fmax is None else min(float(fmax), isco)
    prefactor = (1.77 ** 2 * 5.0 * _C ** (1.0 / 3.0) * (chirp * _G / _C ** 2) ** (5.0 / 3.0)
                 / (96.0 * np.pi ** (4.0 / 3.0) * snr ** 2))
    frequency = np.asarray(frequency, dtype=float)
    psd = np.asarray(psd, dtype=float)
    inside = (frequency >= fmin) & (frequency < top) & (frequency > 0.0)
    integrand = prefactor * frequency[inside] ** (-7.0 / 3.0) / psd[inside]
    return float(np.sqrt(np.trapezoid(integrand, frequency[inside])) / _MEGAPARSEC)


@dataclass
class ValidationReport:
    """What the check measured on one detector's whitened stream, and its verdict.

    :param detector: the detector.
    :param start: GPS start of the stretch checked.
    :param stop: GPS end of it.
    :param bands: the octaves, `(low, high)` in Hz.
    :param power: power of each octave, as a fraction of white.
    :param kurtosis: median window kurtosis of each octave, then of the band
        above the conditioning's high edge.
    :param gaussian: the same estimator on Gaussian noise, band by band.
    :param thirds: power of each octave in each third of the stretch, over its
        power in the whole, shape `(3, n_bands)`.
    :param stop_band: the band above the conditioning's high edge, Hz.
    :param transient_fraction: fraction of the stretch covered by transients
        and gates.
    :param transients: number of transients the census flagged.
    :param gates: GPS start and stop of each gate, shape `(n, 2)`.
    :param range_mpc: binary neutron star range, Mpc; nan when not measured.
    :param failures: one sentence per criterion not met.
    """

    detector: str
    start: float
    stop: float
    bands: tuple
    power: np.ndarray
    kurtosis: np.ndarray
    gaussian: np.ndarray
    thirds: np.ndarray
    stop_band: tuple
    transient_fraction: float
    transients: int
    gates: np.ndarray
    range_mpc: float = float("nan")
    failures: list = field(default_factory=list)

    @property
    def passed(self):
        """True when every criterion holds."""
        return not self.failures

    def message(self):
        """Which detector, and which band and criterion failed, in one text."""
        verdict = "passes" if self.passed else "fails"
        head = f"{self.detector} {verdict} the conditioning check over {self.start:.3f}-{self.stop:.3f}"
        return head if self.passed else head + ": " + "; ".join(self.failures)

    def table(self):
        """The measurements, one line per band, as text."""
        names = [f"{lo:g}-{hi:g}" for lo, hi in self.bands] + \
                [f"{self.stop_band[0]:g}-{self.stop_band[1]:g}"]
        lines = [f"{self.detector}  {self.start:.3f}-{self.stop:.3f}  "
                 f"BNS range {self.range_mpc:.1f} Mpc",
                 f"  {'band (Hz)':>14} {'power/white':>11} {'kurtosis':>9} "
                 f"{'gaussian':>9} {'thirds min-max':>15}"]
        for k, name in enumerate(names):
            power = f"{self.power[k]:11.3f}" if k < len(self.bands) else f"{'':>11}"
            thirds = (f"{self.thirds[:, k].min():7.3f}-{self.thirds[:, k].max():.3f}"
                      if k < len(self.bands) else "")
            lines.append(f"  {name:>14} {power} {self.kurtosis[k]:9.3f} "
                         f"{self.gaussian[k]:9.3f} {thirds:>15}")
        lines.append(f"  transients {self.transients}, gates {len(self.gates)}, "
                     f"covering {100 * self.transient_fraction:.2f}% of the stretch")
        lines.append("  " + ("PASS" if self.passed else "FAIL: " + "; ".join(self.failures)))
        return "\n".join(lines)

    def to_dict(self):
        """The report as plain values, for a JSON record."""
        return dict(detector=self.detector, start=self.start, stop=self.stop,
                    bands=[list(b) for b in self.bands], power=self.power.tolist(),
                    kurtosis=self.kurtosis.tolist(), gaussian=self.gaussian.tolist(),
                    thirds=self.thirds.tolist(), stop_band=list(self.stop_band),
                    transient_fraction=self.transient_fraction,
                    transients=self.transients, gates=np.asarray(self.gates).tolist(),
                    range_mpc=self.range_mpc, failures=list(self.failures),
                    passed=self.passed)


def validate(detector, samples, rate, start, bands, stop_band, found, gates,
             taper_s, window_pad_s, range_mpc=float("nan"),
             power_tolerance=POWER_TOLERANCE, kurtosis_tolerance=KURTOSIS_TOLERANCE,
             transient_tolerance=TRANSIENT_TOLERANCE,
             stationarity_tolerance=STATIONARITY_TOLERANCE,
             window_s=KURTOSIS_WINDOW_S):
    """Check a whitened, gated stream against the four criteria.

    :type detector: str
    :param detector: the detector, for the report.
    :type samples: numpy.ndarray
    :param samples: the stream as the search reads it -- whitened, gated -- and
        divided by the scale the search divides it by.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :type start: float
    :param start: GPS time of its first sample.
    :type bands: tuple
    :param bands: the octaves the search reads, as `octave_bands` returns them.
    :type stop_band: tuple
    :param stop_band: the band above the conditioning's high edge, Hz.
    :type found: wdf.processes.gating.Transients
    :param found: the census of the stream's transients.
    :type gates: numpy.ndarray
    :param gates: GPS start and stop of each gated stretch, shape `(n, 2)`.
    :type taper_s: float
    :param taper_s: the gates' taper on each side, seconds.
    :type window_pad_s: float
    :param window_pad_s: the analysis window, seconds, by which each flagged
        transient is widened on either side.
    :type range_mpc: float
    :param range_mpc: the detector's binary neutron star range, Mpc, reported.
    :return: ValidationReport
    """
    samples = np.asarray(samples, dtype=float)
    rate = float(rate)
    stop = start + samples.size / rate
    failures = []

    power = band_power(samples, rate, bands)
    for (low, high), value in zip(bands, power):
        if abs(value - 1.0) > power_tolerance:
            failures.append(f"{low:g}-{high:g} Hz: power {value:.3f} of white "
                            f"(tolerance {power_tolerance:g})")

    tapered = merged(np.asarray(gates, dtype=float).reshape(-1, 2) + [-taper_s, taper_s])
    lost = merged(np.vstack([
        tapered, np.column_stack([start + found.start / rate - window_pad_s,
                                  start + found.stop / rate + window_pad_s])]))
    # A window is touched by a gate when a gate starts before the window ends
    # and has not ended by the time it starts; the gates are sorted and
    # disjoint, so the two counts differ exactly by the gates it overlaps.
    size = int(round(window_s * rate))
    count = samples.size // size
    window_start = start + np.arange(count) * size / rate
    touched = (np.searchsorted(tapered[:, 0], window_start + size / rate, side="left")
               - np.searchsorted(tapered[:, 1], window_start, side="right")) > 0
    everything = tuple(bands) + (tuple(stop_band),)
    kurt = np.array([window_kurtosis(samples, rate, band, window_s, keep=~touched)
                     for band in everything])
    gaussian = np.array([gaussian_window_kurtosis(samples.size, rate, band, window_s)
                         for band in everything])
    for (low, high), value, reference in zip(everything, kurt, gaussian):
        if abs(value - reference) > kurtosis_tolerance:
            failures.append(f"{low:g}-{high:g} Hz: window kurtosis {value:.3f} against "
                            f"{reference:.3f} for Gaussian noise (tolerance "
                            f"{kurtosis_tolerance:g})")

    clipped = np.clip(lost, start, stop)
    fraction = float(np.sum(clipped[:, 1] - clipped[:, 0]) / (stop - start))
    if fraction > transient_tolerance:
        failures.append(f"transients and gates cover {100 * fraction:.2f}% of the "
                        f"stretch (tolerance {100 * transient_tolerance:g}%)")

    third = samples.size // 3
    thirds = np.array([band_power(samples[k * third:(k + 1) * third], rate, bands)
                       for k in range(3)]) / power
    for k, (low, high) in enumerate(bands):
        worst = int(np.argmax(np.abs(thirds[:, k] - 1.0)))
        if abs(thirds[worst, k] - 1.0) > stationarity_tolerance:
            failures.append(f"{low:g}-{high:g} Hz: power in third {worst + 1} is "
                            f"{thirds[worst, k]:.3f} of the whole stretch's (tolerance "
                            f"{stationarity_tolerance:g})")

    return ValidationReport(
        detector=str(detector), start=float(start), stop=float(stop), bands=tuple(bands),
        power=power, kurtosis=kurt, gaussian=gaussian, thirds=thirds,
        stop_band=tuple(stop_band), transient_fraction=fraction,
        transients=len(found), gates=np.asarray(gates, dtype=float).reshape(-1, 2),
        range_mpc=float(range_mpc), failures=failures)
