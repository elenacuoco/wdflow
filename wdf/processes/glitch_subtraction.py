"""Subtracting glitches from the conditioned stream with whitened sine-Gaussians.

A glitch is modelled as a sum of sine-Gaussian atoms in the **conditioned**
domain (the band-passed, decimated stream the whitening reads) and fitted in
the **whitened** domain, where the noise is white and least squares is the
right fit. The fitted atoms are then subtracted from the conditioned stream
before the whitening, which by linearity is the same as subtracting their
whitened images from the whitened stream: nothing is zeroed, so no gate edge
is spread by the whitening, and the data around and under the glitch -- a
chirp included -- is kept.

The algorithm (greedy matching pursuit, "M4"):

1. The bank. Each atom is a complex sine-Gaussian

   ``g(t) = exp(-t^2 / (2 tau^2)) exp(2 pi i f t)``,
   ``tau = Q / (2 pi f) / sqrt(2)``,

   cut at ``|t| <= ceil(cut_tau tau fs) / fs``, for every quality factor of
   `GlitchQs` and every frequency of `bank_frequencies`: `GlitchFrequencyCount`
   frequencies log-spaced from `LowFrequencyCut` to `GlitchTopFraction` of the
   Nyquist frequency, so that the bank follows the rate. Its whitened image is ``h * g`` with ``h`` the
   whitening's zero-phase taps divided by its scale ``sigma``
   (`MagnitudeWhitening.taps` and `.sigma`), so that the whitened noise has
   unit variance.
2. The fit. At every candidate centre the residual's projections on the
   whitened real and imaginary parts, ``c = (<r, h*Re g>, <r, h*Im g>)``, give
   the exact two-parameter least-squares amplitude ``(alpha, beta) = G^-1 c``,
   ``G`` the 2x2 Gram matrix of the two whitened parts, and the atom's
   signal-to-noise ratio ``sqrt(c^T G^-1 c)``: the phase is fitted exactly.
3. The pursuit. The atom with the largest SNR over the bank and over the
   centres allowed (within ``half_width`` of the seed) is subtracted from the
   residual; this repeats until the best SNR is below `stop_snr`
   (`GlitchStopSNR`) or `max_atoms` (`GlitchMaxAtoms`) atoms have been taken.

The projections ``<r, h * g_j>`` equal ``<h * r, g_j>`` because the taps are
symmetric, and the residual is kept as the conditioned stream minus the model,
so the pursuit correlates the doubly whitened stream ``k * x`` (``k = h * h``)
with the short conditioned atoms and updates it by ``k * a`` after each atom.
That is exact in the infinite domain and needs no truncation of the whitened
atoms' tails.

Where to fit (`glitch_seeds`). The census reads the whitened stream: its level
is the largest ratio of the stream to its robust standard deviation, broadband
and in each octave from `GlitchCensusLow` (the search's `MinFrequency`) up
(`wdf.processes.gating.transients`). A transient whose peak reaches the
trigger (`GlitchTrigger`, the gate's threshold) seeds a fit. Its
**extent** is taken from the broadband stream and the octaves from
`GlitchExtentLow` (by default `extent_low_factor` times the census's lowest frequency)
only: the 16-32 Hz octave carries the
whitening's own memory (2.7 s with the 16 s truncated filter) and would size
the fit by the filter rather than the glitch. The seed is the time of the
largest of those levels, and atoms may be centred within
``extent / 2 + GlitchSeedPad`` of it.

Protection. A seed found in two or more detectors within their light travel
time plus `GlitchCoincidenceJitter` may be a gravitational wave and is not
subtracted (`protect_coincident`, `GlitchProtectCoincident`, on by default).

Every choice is a run parameter read by `M4Settings.from_parameters`, whose
defaults are table ``[glitch_subtraction]`` of wdf/config/defaults.toml.

Public API: `M4Settings`, `bank_frequencies`, `Atom`, `SineGaussianBank`,
`matching_pursuit`, `census_levels`, `glitch_seeds`, `protect_coincident`,
`whitened_census`, `fit_seeds`, `GlitchModel`, `fit_glitches`. Times are GPS seconds; arrays
are conditioned samples at the stream's own rate unless the name says
whitened.
"""
__author__ = "Elena Cuoco"
__license__ = "GPL"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import fftconvolve

from wdf.analysis.detectors import light_travel_time
from wdf.config.defaults import defaults
from wdf.processes.census import (DEFAULT_OCTAVE_FILTER_ORDER, band_stream, octave_bands,
                                  robust_sigma, transients)

#: The defaults of the M4 stage: table ``[glitch_subtraction]`` of
#: wdf/config/defaults.toml. The names below are their aliases.
_GS = defaults("glitch_subtraction")

#: Quality factors of the bank (`GlitchQs`).
DEFAULT_QS = _GS["GlitchQs"]
#: Top frequency of the bank as a fraction of the stream's Nyquist frequency
#: (`GlitchTopFraction`).
DEFAULT_TOP_FRACTION = _GS["GlitchTopFraction"]
#: Number of frequencies of the bank, log-spaced from its bottom (the run's
#: `LowFrequencyCut`) to its top (`GlitchFrequencyCount`).
DEFAULT_FREQUENCY_COUNT = _GS["GlitchFrequencyCount"]
#: The pursuit stops when the best atom's whitened SNR is below this
#: (`GlitchStopSNR`).
DEFAULT_STOP_SNR = _GS["GlitchStopSNR"]
#: ... or when this many atoms have been fitted to one seed (`GlitchMaxAtoms`).
DEFAULT_MAX_ATOMS = _GS["GlitchMaxAtoms"]
#: Smallest census peak, in robust standard deviations, that seeds a fit
#: (`GlitchTrigger`).
DEFAULT_TRIGGER = _GS["GlitchTrigger"]
#: Level, robust standard deviations, a sample must exceed to belong to a
#: transient (the census's flag, `GlitchFlag`).
DEFAULT_FLAG = _GS["GlitchFlag"]
#: Largest gap inside one transient, s (`GlitchMergeGap`).
DEFAULT_MERGE_S = _GS["GlitchMergeGap"]
#: Seconds added to half the extent to give the half-width of the centres
#: (`GlitchSeedPad`).
DEFAULT_SEED_PAD_S = _GS["GlitchSeedPad"]
#: The lowest octave a transient's extent is read from, as a multiple of the
#: census's lowest frequency, when `GlitchExtentLow` is unset.
DEFAULT_EXTENT_LOW_FACTOR = _GS["extent_low_factor"]
#: An atom is cut at this many ``tau`` on each side (`GlitchAtomCutTau`).
DEFAULT_ATOM_CUT_TAU = _GS["GlitchAtomCutTau"]
#: Seconds added to the light travel time between two detectors within which
#: their seeds are coincident (`GlitchCoincidenceJitter`).
DEFAULT_COINCIDENCE_JITTER_S = _GS["GlitchCoincidenceJitter"]


def bank_frequencies(rate, low, top_fraction=DEFAULT_TOP_FRACTION,
                     count=DEFAULT_FREQUENCY_COUNT):
    """The bank's frequencies: `count`, log-spaced from `low` to
    `top_fraction` of the Nyquist frequency.

    The bank follows the rate it is built at, so a stream at another rate gets
    a bank over its own band rather than the 2048 Hz one.

    :type rate: float
    :param rate: sampling rate of the conditioned stream, Hz.
    :type low: float
    :param low: lowest frequency, Hz (the run's `LowFrequencyCut`).
    :type top_fraction: float
    :param top_fraction: highest frequency over the Nyquist frequency, < 1.
    :type count: int
    :param count: number of frequencies.
    :return: tuple of float -- Hz, ascending.
    :raises ValueError: if the band is empty or `count` is below 1.
    """
    top = float(top_fraction) * 0.5 * float(rate)
    if not 0.0 < float(low) < top < 0.5 * float(rate) or int(count) < 1:
        raise ValueError(f"no bank between {low} Hz and {top} Hz with {count} frequencies "
                         f"at {rate} Hz")
    return tuple(np.geomspace(float(low), top, int(count)))


@dataclass
class M4Settings:
    """Every choice of the M4 stage, defaults relative to the rate.

    :param qs: quality factors of the bank.
    :param low: bottom of the bank, Hz (the run's `LowFrequencyCut`).
    :param top_fraction: top of the bank over the Nyquist frequency.
    :param count: frequencies of the bank.
    :param atom_cut_tau: an atom is cut at this many tau.
    :param trigger: smallest census peak fitted, robust sigma.
    :param flag: census flag level, robust sigma.
    :param merge_s: largest gap inside one transient, s.
    :param census_low: lowest octave of the census, Hz.
    :param extent_low: lowest octave a transient's extent is read from, Hz.
    :param pad: added to half the extent, s.
    :param stop_snr: the pursuit stops below this atom SNR.
    :param max_atoms: most atoms per seed.
    :param octave_order: order of the census's octave filters.
    :param protect_coincident: leave seeds coincident in two or more detectors
        unsubtracted.
    :param jitter_s: added to the light travel time for coincidence, s.
    """

    qs: tuple
    low: float
    top_fraction: float
    count: int
    atom_cut_tau: float
    trigger: float
    flag: float
    merge_s: float
    census_low: float
    extent_low: float
    pad: float
    stop_snr: float
    max_atoms: int
    octave_order: int
    protect_coincident: bool
    jitter_s: float

    #: The run parameter each field is read from.
    KEYS = dict(qs="GlitchQs", low="LowFrequencyCut", top_fraction="GlitchTopFraction",
                count="GlitchFrequencyCount", atom_cut_tau="GlitchAtomCutTau",
                trigger="GlitchTrigger", flag="GlitchFlag", merge_s="GlitchMergeGap",
                census_low="GlitchCensusLow", extent_low="GlitchExtentLow", pad="GlitchSeedPad",
                stop_snr="GlitchStopSNR", max_atoms="GlitchMaxAtoms",
                octave_order="GlitchOctaveOrder", protect_coincident="GlitchProtectCoincident",
                jitter_s="GlitchCoincidenceJitter")

    @classmethod
    def from_parameters(cls, par):
        """The settings a run asks for, defaults filled in and written back.

        `GlitchCensusLow` defaults to the search's `MinFrequency` when it is
        set, else to `LowFrequencyCut`; `GlitchExtentLow` to
        `DEFAULT_EXTENT_LOW_FACTOR` times the census's lowest frequency.

        :param par: run configuration; it receives every value used.
        :return: M4Settings
        """
        from wdf.config.Parameters import low_frequency_cut

        def get(key, default):
            value = getattr(par, key, None)
            return default if value is None else value

        low = low_frequency_cut(par)
        census_low = get("GlitchCensusLow", _GS["GlitchCensusLow"])
        census_low = float(getattr(par, "MinFrequency", None) or low if census_low is None
                           else census_low)
        extent_low = get("GlitchExtentLow", _GS["GlitchExtentLow"])
        out = cls(
            qs=tuple(float(q) for q in get("GlitchQs", DEFAULT_QS)),
            low=low,
            top_fraction=float(get("GlitchTopFraction", DEFAULT_TOP_FRACTION)),
            count=int(get("GlitchFrequencyCount", DEFAULT_FREQUENCY_COUNT)),
            atom_cut_tau=float(get("GlitchAtomCutTau", DEFAULT_ATOM_CUT_TAU)),
            trigger=float(get("GlitchTrigger", DEFAULT_TRIGGER)),
            flag=float(get("GlitchFlag", DEFAULT_FLAG)),
            merge_s=float(get("GlitchMergeGap", DEFAULT_MERGE_S)),
            census_low=census_low,
            extent_low=float(DEFAULT_EXTENT_LOW_FACTOR * census_low if extent_low is None
                             else extent_low),
            pad=float(get("GlitchSeedPad", DEFAULT_SEED_PAD_S)),
            stop_snr=float(get("GlitchStopSNR", DEFAULT_STOP_SNR)),
            max_atoms=int(get("GlitchMaxAtoms", DEFAULT_MAX_ATOMS)),
            octave_order=int(get("GlitchOctaveOrder", DEFAULT_OCTAVE_FILTER_ORDER)),
            protect_coincident=bool(get("GlitchProtectCoincident", _GS["GlitchProtectCoincident"])),
            jitter_s=float(get("GlitchCoincidenceJitter", DEFAULT_COINCIDENCE_JITTER_S)))
        for field, key in cls.KEYS.items():
            value = getattr(out, field)
            setattr(par, key, list(value) if isinstance(value, tuple) else value)
        return out

    def frequencies(self, rate):
        """The bank's frequencies at `rate` (`bank_frequencies`)."""
        return bank_frequencies(rate, self.low, self.top_fraction, self.count)

    def census(self):
        """The keyword arguments of `glitch_seeds`."""
        return dict(trigger=self.trigger, low=self.census_low, extent_low=self.extent_low,
                    flag=self.flag, merge_s=self.merge_s, pad=self.pad,
                    order=self.octave_order)


def _half_length(q, frequency, rate, cut_tau=DEFAULT_ATOM_CUT_TAU):
    """Half-length, in samples, of the atom of quality `q` at `frequency`."""
    tau = q / (2.0 * np.pi * frequency) / np.sqrt(2.0)
    return tau, int(np.ceil(cut_tau * tau * rate))


@dataclass
class Atom:
    """One fitted sine-Gaussian, in the conditioned domain.

    The atom is ``alpha * Re g + beta * Im g`` with ``g`` the complex
    sine-Gaussian of `q` and `frequency` centred at `centre`, cut at
    `half_length` samples on each side at the sampling rate `rate`.

    :param centre: GPS time of the centre, s (a sample of the stream it was
        fitted on).
    :param frequency: Hz.
    :param q: quality factor.
    :param alpha: amplitude of the cosine part, conditioned units.
    :param beta: amplitude of the sine part, conditioned units.
    :param snr: whitened signal-to-noise ratio of the atom when it was taken.
    :param rate: sampling rate of the stream, Hz.
    :param half_length: half-length, samples.
    """

    centre: float
    frequency: float
    q: float
    alpha: float
    beta: float
    snr: float
    rate: float
    half_length: int

    @property
    def tau(self):
        """Gaussian width, s."""
        return self.q / (2.0 * np.pi * self.frequency) / np.sqrt(2.0)

    @property
    def span(self):
        """``(first, last)`` GPS times the atom touches."""
        h = self.half_length / self.rate
        return self.centre - h, self.centre + h

    def evaluate(self, start, size):
        """The atom on a run of samples at its own rate.

        Each sample is evaluated at its own time, so a stream cleaned block by
        block is the stream cleaned whole.

        :type start: float
        :param start: GPS time of the first sample.
        :type size: int
        :param size: number of samples.
        :return: numpy.ndarray -- `size` values, zero outside the atom.
        """
        out = np.zeros(int(size))
        offset = (self.centre - start) * self.rate
        centre = int(np.round(offset))
        first = max(centre - self.half_length, 0)
        last = min(centre + self.half_length + 1, int(size))
        if last <= first:
            return out
        dt = (np.arange(first, last) - offset) / self.rate
        envelope = np.exp(-dt ** 2 / (2.0 * self.tau ** 2))
        phase = 2.0 * np.pi * self.frequency * dt
        out[first:last] = envelope * (self.alpha * np.cos(phase) + self.beta * np.sin(phase))
        return out


class SineGaussianBank:
    """The atoms a pursuit chooses from, with their whitened Gram matrices.

    :type taps: numpy.ndarray
    :param taps: the whitening's symmetric, odd-length taps
        (`MagnitudeWhitening.taps`), applied as ``fftconvolve(x, taps, "same")``.
    :type sigma: float
    :param sigma: the whitening's scale (`MagnitudeWhitening.sigma`); the
        whitened stream is the convolution divided by it.
    :type rate: float
    :param rate: sampling rate of the conditioned stream, Hz.
    :type qs: iterable of float
    :param qs: quality factors.
    :type frequencies: iterable of float
    :param frequencies: Hz; those at or above the Nyquist frequency are left out.
        `bank_frequencies(rate, low)` builds them relative to the rate.
    :type cut_tau: float
    :param cut_tau: an atom is cut at this many tau on each side.
    """

    def __init__(self, taps, sigma, rate, frequencies, qs=DEFAULT_QS, cut_tau=DEFAULT_ATOM_CUT_TAU):
        taps = np.asarray(taps, dtype=float)
        if taps.size % 2 == 0 or not np.allclose(taps, taps[::-1], rtol=0, atol=1e-12 * np.abs(taps).max()):
            raise ValueError("the whitening taps must be symmetric with an odd length (zero phase)")
        self.taps, self.sigma, self.rate = taps, float(sigma), float(rate)
        #: The doubly whitening kernel ``h * h``, ``h = taps / sigma``.
        self.kernel = fftconvolve(taps, taps) / self.sigma ** 2
        self.atoms = []
        for q in qs:
            for f in frequencies:
                if f >= 0.5 * rate:
                    continue
                tau, half = _half_length(float(q), float(f), self.rate, float(cut_tau))
                t = np.arange(-half, half + 1) / self.rate
                g = np.exp(-t ** 2 / (2.0 * tau ** 2)) * np.exp(2j * np.pi * f * t)
                white = fftconvolve(g, taps) / self.sigma
                re, im = white.real, white.imag
                gram = np.array([[re @ re, re @ im], [re @ im, im @ im]])
                self.atoms.append(dict(q=float(q), frequency=float(f), half=half, g=g,
                                       gram_inv=np.linalg.inv(gram)))

    @property
    def longest(self):
        """Largest half-length of an atom, samples."""
        return max(a["half"] for a in self.atoms)

    @property
    def context(self):
        """Samples a fit needs on each side of its centres: the longest atom
        and the doubly whitening kernel's half-length."""
        return self.longest + (self.kernel.size - 1) // 2

    def whiten(self, samples):
        """The whitened stream, ``fftconvolve(samples, taps, "same") / sigma``."""
        return fftconvolve(np.asarray(samples, dtype=float), self.taps, mode="same") / self.sigma


def matching_pursuit(conditioned, start, bank, centre, half_width,
                     stop_snr=DEFAULT_STOP_SNR, max_atoms=DEFAULT_MAX_ATOMS):
    """Greedy whitened matching pursuit around one seed.

    :type conditioned: numpy.ndarray
    :param conditioned: the conditioned stream, at the bank's rate; it must
        hold `bank.context` samples beyond the centres allowed on each side.
    :type start: float
    :param start: GPS time of its first sample.
    :type bank: SineGaussianBank
    :param bank: the atoms and the whitening.
    :type centre: float
    :param centre: GPS time of the seed.
    :type half_width: float
    :param half_width: atoms are centred within this many seconds of `centre`.
    :type stop_snr: float
    :param stop_snr: stop when the best atom's whitened SNR is below it.
    :type max_atoms: int
    :param max_atoms: stop after this many atoms.
    :return: tuple -- (list of `Atom`, model): the model is the sum of the
        atoms on the samples of `conditioned`.
    :raises ValueError: if `conditioned` lacks the context the fit needs.
    """
    x = np.asarray(conditioned, dtype=float)
    rate = bank.rate
    lo = int(np.ceil((centre - half_width - start) * rate - 1e-9))
    hi = int(np.floor((centre + half_width - start) * rate + 1e-9))
    if lo - bank.context < 0 or hi + bank.context >= x.size:
        raise ValueError(
            f"the stream around {centre} lacks the {bank.context / rate:.1f} s of "
            "context the fit needs on each side")
    kernel = bank.kernel
    kh = (kernel.size - 1) // 2
    # Only the doubly whitened stream over the centres and the atoms' reach is
    # ever read; compute it on that span from the samples it depends on.
    reach0, reach1 = lo - bank.longest, hi + bank.longest + 1
    q = fftconvolve(x[reach0 - kh:reach1 + kh], kernel, mode="valid")    # samples reach0 .. reach1-1
    model = np.zeros(x.size)
    atoms = []
    for _ in range(int(max_atoms)):
        best = (0.0, None)
        for k, a in enumerate(bank.atoms):
            half = a["half"]
            seg = q[lo - half - reach0:hi + half + 1 - reach0]
            c = fftconvolve(seg, a["g"][::-1], mode="valid")           # centres lo .. hi
            cr, ci = c.real, c.imag
            gi = a["gram_inv"]
            snr2 = gi[0, 0] * cr * cr + 2.0 * gi[0, 1] * cr * ci + gi[1, 1] * ci * ci
            j = int(np.argmax(snr2))
            if snr2[j] > best[0] ** 2:
                best = (float(np.sqrt(max(snr2[j], 0.0))), (k, j, cr[j], ci[j]))
        if best[1] is None or best[0] < stop_snr:
            break
        k, j, cr, ci = best[1]
        a = bank.atoms[k]
        alpha, beta = a["gram_inv"] @ np.array([cr, ci])
        at = lo + j
        half = a["half"]
        wave = alpha * a["g"].real + beta * a["g"].imag
        model[at - half:at + half + 1] += wave
        # q -= k * wave, on the span q holds
        upd = fftconvolve(wave, kernel)                                  # samples at - half - kh ...
        first = at - half - kh
        a0, a1 = max(first, reach0), min(first + upd.size, reach1)
        if a1 > a0:
            q[a0 - reach0:a1 - reach0] -= upd[a0 - first:a1 - first]
        atoms.append(Atom(centre=start + at / rate, frequency=a["frequency"], q=a["q"],
                          alpha=float(alpha), beta=float(beta), snr=best[0], rate=rate,
                          half_length=int(half)))
    return atoms, model


def census_levels(whitened, rate, low, extent_low, order=DEFAULT_OCTAVE_FILTER_ORDER):
    """The census's level, and the level a transient's extent is read from.

    :type whitened: numpy.ndarray
    :param whitened: the whitened stream.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :type low: float
    :param low: lowest octave of the census, Hz.
    :type extent_low: float
    :param extent_low: lowest octave of the extent, Hz.
    :type order: int
    :param order: order of the octave filters.
    :return: tuple -- (level, extent level): per sample, the largest ratio of
        the stream to its robust standard deviation, broadband and over the
        octaves from `low` (resp. from `extent_low`).
    """
    w = np.asarray(whitened, dtype=float)
    broadband = np.abs(w) / robust_sigma(w)
    level, extent = broadband.copy(), broadband.copy()
    for band in octave_bands(rate, low):
        s = band_stream(w, rate, band, order)
        r = np.abs(s) / robust_sigma(s)
        np.maximum(level, r, out=level)
        if band[0] >= extent_low:
            np.maximum(extent, r, out=extent)
    return level, extent


def glitch_seeds(whitened, start, rate, trigger, low, extent_low, flag, merge_s, pad,
                 order=DEFAULT_OCTAVE_FILTER_ORDER):
    """Where to fit: the census's transients that reach `trigger`.

    A transient's **extent** is read from the broadband stream and the octaves
    from `extent_low` only: the lowest octaves carry the whitening's own memory
    (2.7 s in 16-32 Hz with the 16 s truncated filter) and would size the fit
    by the filter rather than the glitch (the census fix of a1bff75).

    :type whitened: numpy.ndarray
    :param whitened: the whitened stream (unit variance in noise).
    :type start: float
    :param start: GPS time of its first sample.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :type trigger: float
    :param trigger: smallest census peak that seeds a fit, robust sigma.
    :type low: float
    :param low: lowest octave of the census, Hz.
    :type extent_low: float
    :param extent_low: lowest octave a transient's extent is read from, Hz.
    :type flag: float
    :param flag: level a sample must exceed to belong to a transient.
    :type merge_s: float
    :param merge_s: largest gap inside one transient, s.
    :type pad: float
    :param pad: added to half the extent, s.
    :type order: int
    :param order: order of the octave filters.
    :return: list[dict] -- per seed: ``centre`` (GPS of the largest extent
        level, or of the census peak when no sample of the broadband stream or
        of the octaves from `extent_low` exceeds `flag`), ``half_width``
        (``extent / 2 + pad``), ``extent`` (s), ``peak`` (census peak),
        ``first`` and ``last`` (GPS of the census transient), and
        ``protected`` False.
    """
    w = np.asarray(whitened, dtype=float)
    found = transients(w, rate, octave_bands(rate, low), flag=flag, merge_s=merge_s, order=order)
    if not np.any(np.asarray(found.peak) >= trigger):
        return []
    level, extent = census_levels(w, rate, low, extent_low, order)
    seeds = []
    for a, b, peak in zip(found.start, found.stop, found.peak):
        if peak < trigger:
            continue
        ext = extent[a:b]
        above = np.flatnonzero(ext > flag)
        if above.size:
            length = (above[-1] - above[0] + 1) / rate
            at = a + int(np.argmax(ext))
        else:
            length = 0.0
            at = a + int(np.argmax(level[a:b]))
        seeds.append(dict(centre=start + at / rate, half_width=0.5 * length + pad, extent=length,
                          peak=float(peak), first=start + a / rate, last=start + b / rate,
                          protected=False))
    return seeds


def protect_coincident(seeds_by_ifo, jitter_s=DEFAULT_COINCIDENCE_JITTER_S):
    """Mark the seeds found in coincidence in two or more detectors.

    A loud transient seen in two detectors within their light travel time may
    be a gravitational wave, and subtracting it as a glitch would remove the
    signal. Two seeds of detectors A and B are coincident when their census
    transients, each widened by the light travel time between A and B plus
    `jitter_s` on each side, overlap. Every seed coincident with a seed of
    another detector gets ``protected`` True and the list of those detectors
    in ``coincident_with``; the others ``protected`` False. The seeds are
    marked in place and returned.

    :type seeds_by_ifo: dict
    :param seeds_by_ifo: detector name ("H1", "L1", "V1", ...) to its seeds,
        as `glitch_seeds` returns them.
    :type jitter_s: float
    :param jitter_s: seconds added to the light travel time.
    :return: dict -- the same mapping, the seeds marked.
    """
    ifos = list(seeds_by_ifo)
    for ifo in ifos:
        for seed in seeds_by_ifo[ifo]:
            seed["protected"], seed["coincident_with"] = False, []
    for n, a in enumerate(ifos):
        for b in ifos[n + 1:]:
            tol = light_travel_time(a, b) + float(jitter_s)
            for sa in seeds_by_ifo[a]:
                for sb in seeds_by_ifo[b]:
                    if sa["first"] - tol <= sb["last"] and sb["first"] - tol <= sa["last"]:
                        sa["protected"] = sb["protected"] = True
                        sa["coincident_with"].append(b)
                        sb["coincident_with"].append(a)
    return seeds_by_ifo


class GlitchModel:
    """The fitted atoms of a stream, evaluated on any run of its samples.

    :type atoms: iterable of Atom
    :param atoms: the atoms.
    :type seeds: list[dict] or None
    :param seeds: the seeds they were fitted to (`glitch_seeds`), recorded.
    """

    def __init__(self, atoms=(), seeds=None):
        self.atoms = list(atoms)
        self.seeds = list(seeds or [])
        spans = np.array([a.span for a in self.atoms]).reshape(-1, 2)
        self._first, self._last = spans[:, 0], spans[:, 1]

    def __len__(self):
        return len(self.atoms)

    def evaluate(self, start, rate, size):
        """The sum of the atoms on `size` samples from `start`.

        :return: numpy.ndarray or None -- None when no atom reaches the run, so
            that a caller can leave the run as it is.
        """
        stop = start + (size - 1) / rate
        near = np.flatnonzero((self._last >= start) & (self._first <= stop))
        if near.size == 0:
            return None
        out = np.zeros(int(size))
        for i in near:
            atom = self.atoms[i]
            if abs(atom.rate - rate) > 1e-9 * rate:
                raise ValueError(f"an atom fitted at {atom.rate} Hz cannot be laid on a {rate} Hz stream")
            out += atom.evaluate(start, size)
        return out

    def subtract(self, samples, start, rate):
        """`samples` with the atoms subtracted (a copy)."""
        samples = np.array(samples, dtype=float)
        m = self.evaluate(start, rate, samples.size)
        return samples if m is None else samples - m

    def table(self):
        """The atoms as a list of dicts (one row each), for writing."""
        return [asdict(a) for a in self.atoms]


def whitened_census(conditioned, start, rate, bank, settings):
    """The seeds of a conditioned stream, through the bank's whitening.

    The census runs on the samples the whitening reached with real data on
    both sides.

    :type conditioned: numpy.ndarray
    :param conditioned: the conditioned stream.
    :type start: float
    :param start: GPS time of its first sample.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :type bank: SineGaussianBank
    :param bank: the bank, carrying the whitening's taps and scale.
    :type settings: M4Settings
    :param settings: the census's settings.
    :return: list[dict] -- the seeds, as `glitch_seeds` returns them.
    """
    x = np.asarray(conditioned, dtype=float)
    support = (bank.taps.size - 1) // 2
    w = bank.whiten(x)
    inner = w[support:x.size - support]
    if inner.size == 0:
        return []
    return glitch_seeds(inner, start + support / rate, rate, **settings.census())


def fit_seeds(conditioned, start, rate, bank, seeds, settings):
    """Fit every seed that is not protected, in time order, each on the stream
    with the atoms already found subtracted.

    :type conditioned: numpy.ndarray
    :param conditioned: the conditioned stream.
    :type start: float
    :param start: GPS time of its first sample.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :type bank: SineGaussianBank
    :param bank: the atoms and the whitening.
    :type seeds: list[dict]
    :param seeds: the seeds; each receives ``fitted`` and ``atoms``.
    :type settings: M4Settings
    :param settings: the pursuit's settings.
    :return: GlitchModel -- with the seeds recorded; a protected seed, or one
        too close to the stream's ends to be fitted, has ``fitted`` False.
    """
    x = np.asarray(conditioned, dtype=float)
    atoms = []
    model = np.zeros(x.size)
    for seed in seeds:
        if seed.get("protected", False):
            seed["fitted"], seed["atoms"] = False, 0
            continue
        try:
            found, local = matching_pursuit(x - model, start, bank, seed["centre"], seed["half_width"],
                                            stop_snr=settings.stop_snr, max_atoms=settings.max_atoms)
        except ValueError:
            seed["fitted"], seed["atoms"] = False, 0
            continue
        seed["fitted"], seed["atoms"] = True, len(found)
        model += local
        atoms += found
    return GlitchModel(atoms, seeds)


def fit_glitches(conditioned, start, rate, taps, sigma, settings, bank=None):
    """Census and pursuit over one detector's conditioned stream.

    Without the other detectors' seeds no seed can be protected here; a
    network run takes each detector's census (`whitened_census`), marks the
    coincident seeds (`protect_coincident`) and fits the rest (`fit_seeds`).

    :type conditioned: numpy.ndarray
    :param conditioned: the conditioned stream.
    :type start: float
    :param start: GPS time of its first sample.
    :type rate: float
    :param rate: its sampling rate, Hz.
    :type taps: numpy.ndarray
    :param taps: the whitening's symmetric taps.
    :type sigma: float
    :param sigma: the whitening's scale.
    :type settings: M4Settings
    :param settings: every choice of the stage.
    :type bank: SineGaussianBank or None
    :param bank: built from `taps`, `sigma`, `rate` and `settings` when None.
    :return: GlitchModel
    """
    bank = SineGaussianBank(taps, sigma, rate, settings.frequencies(rate), qs=settings.qs,
                            cut_tau=settings.atom_cut_tau) if bank is None else bank
    seeds = whitened_census(conditioned, start, rate, bank, settings)
    return fit_seeds(conditioned, start, rate, bank, seeds, settings)
