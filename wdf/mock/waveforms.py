"""Waveform generators for mock data sets: compact-binary signals and glitches.

Every generator returns a zero-padded-free time series sampled at `sample_rate`,
with an arbitrary overall amplitude: injection code rescales it to a requested
signal-to-noise ratio, so only the shape carries meaning here.
"""
from __future__ import annotations

import numpy as np

CBC_CLASSES = ("bbh", "bhns", "bns")
GLITCH_CLASSES = ("gaussian", "sine_gaussian", "blip", "chirplike", "scattered_light")


def cbc_polarisations(mass1, mass2, spin1z=0.0, spin2z=0.0, distance=100.0,
                      inclination=0.0, f_lower=20.0, sample_rate=2048,
                      approximant="IMRPhenomD"):
    """Plus and cross polarisations of a compact-binary coalescence.

    :type mass1: float
    :param mass1: primary mass, solar masses.
    :type mass2: float
    :param mass2: secondary mass, solar masses.
    :type spin1z: float
    :param spin1z: dimensionless aligned spin of the primary.
    :type spin2z: float
    :param spin2z: dimensionless aligned spin of the secondary.
    :type distance: float
    :param distance: luminosity distance, Mpc.
    :type inclination: float
    :param inclination: inclination angle, radians.
    :type f_lower: float
    :param f_lower: waveform starting frequency, Hz.
    :type sample_rate: int
    :param sample_rate: sampling rate, Hz.
    :type approximant: str
    :param approximant: any time-domain approximant known to pycbc.
    :return: tuple -- (hp, hc) as pycbc TimeSeries, merger at t = 0.
    """
    from pycbc.waveform import get_td_waveform

    return get_td_waveform(approximant=approximant, mass1=mass1, mass2=mass2,
                           spin1z=spin1z, spin2z=spin2z, distance=distance,
                           inclination=inclination, delta_t=1.0 / sample_rate,
                           f_lower=f_lower)


def gaussian(sigma_t=0.005, sample_rate=2048, n_sigma=6.0):
    """Gaussian pulse.

    :type sigma_t: float
    :param sigma_t: Gaussian width, seconds.
    :type sample_rate: int
    :param sample_rate: sampling rate, Hz.
    :type n_sigma: float
    :param n_sigma: half-length of the generated series, in units of `sigma_t`.
    :return: numpy.ndarray -- the pulse, centred in the array.
    """
    t = _symmetric_times(n_sigma * sigma_t, sample_rate)
    return np.exp(-0.5 * (t / sigma_t) ** 2)


def sine_gaussian(f0=150.0, q=12.0, sample_rate=2048, n_tau=4.0):
    """Sine-Gaussian burst.

    :type f0: float
    :param f0: central frequency, Hz.
    :type q: float
    :param q: quality factor; larger means longer and narrower in frequency.
    :type sample_rate: int
    :param sample_rate: sampling rate, Hz.
    :type n_tau: float
    :param n_tau: half-length of the generated series, in decay times.
    :return: numpy.ndarray -- the burst, centred in the array.
    """
    tau = q / (2.0 * np.pi * f0)
    t = _symmetric_times(n_tau * tau, sample_rate)
    return np.exp(-0.5 * (t / tau) ** 2) * np.sin(2.0 * np.pi * f0 * t)


def blip(f0=250.0, q=3.0, asymmetry=3.0, sample_rate=2048, n_tau=6.0):
    """Blip: a short, broadband burst with a fast rise and a slower decay.

    :type f0: float
    :param f0: central frequency, Hz.
    :type q: float
    :param q: quality factor; low values give the broadband, teardrop shape.
    :type asymmetry: float
    :param asymmetry: ratio of decay to rise time.
    :type sample_rate: int
    :param sample_rate: sampling rate, Hz.
    :type n_tau: float
    :param n_tau: half-length of the generated series, in decay times.
    :return: numpy.ndarray -- the burst, centred in the array.
    """
    tau = q / (2.0 * np.pi * f0)
    t = _symmetric_times(n_tau * tau * asymmetry, sample_rate)
    width = np.where(t < 0, tau, tau * asymmetry)
    return np.exp(-0.5 * (t / width) ** 2) * np.sin(2.0 * np.pi * f0 * t)


def chirplike(f_start=40.0, f_end=400.0, duration=0.5, sample_rate=2048):
    """Frequency sweep under a Gaussian envelope, mimicking a chirping artefact.

    :type f_start: float
    :param f_start: frequency at the start of the sweep, Hz.
    :type f_end: float
    :param f_end: frequency at the end of the sweep, Hz.
    :type duration: float
    :param duration: sweep duration, seconds.
    :type sample_rate: int
    :param sample_rate: sampling rate, Hz.
    :return: numpy.ndarray -- the sweep, tapered at both ends.
    """
    t = np.arange(0.0, duration, 1.0 / sample_rate)
    rate = (f_end - f_start) / duration
    phase = 2.0 * np.pi * (f_start * t + 0.5 * rate * t ** 2)
    envelope = np.exp(-0.5 * ((t - 0.5 * duration) / (0.25 * duration)) ** 2)
    return envelope * np.sin(phase)


def scattered_light(f_peak=35.0, arch_period=1.0, n_arches=4, sample_rate=2048):
    """Scattered-light artefact: repeated low-frequency arches.

    The instantaneous frequency follows `f_peak * |sin(pi t / arch_period)|`,
    which traces the arch shape these artefacts show in a spectrogram.

    :type f_peak: float
    :param f_peak: highest instantaneous frequency reached by an arch, Hz.
    :type arch_period: float
    :param arch_period: duration of a single arch, seconds.
    :type n_arches: int
    :param n_arches: number of consecutive arches.
    :type sample_rate: int
    :param sample_rate: sampling rate, Hz.
    :return: numpy.ndarray -- the artefact, tapered at both ends.
    """
    duration = arch_period * n_arches
    t = np.arange(0.0, duration, 1.0 / sample_rate)
    f_inst = f_peak * np.abs(np.sin(np.pi * t / arch_period))
    phase = 2.0 * np.pi * np.cumsum(f_inst) / sample_rate
    envelope = np.sin(np.pi * t / duration) ** 2
    return envelope * np.sin(phase)


GLITCH_GENERATORS = {
    "gaussian": gaussian,
    "sine_gaussian": sine_gaussian,
    "blip": blip,
    "chirplike": chirplike,
    "scattered_light": scattered_light,
}


# A core-collapse supernova is not written down in closed form. Its waveform is
# the output of a hydrodynamic simulation, read from a catalogue, and what
# distinguishes one entry from another is the progenitor model and the direction
# the observer sits in relative to the simulation's axes --- the same explosion
# radiates differently along each. Both are drawn, since neither is known in
# advance for a real source.
CCSN_FILE_PATTERN = "*_strains_*.txt"


def ccsn_catalogue(directory):
    """Index the core-collapse supernova waveforms a directory holds.

    The index is built from what is on disk rather than from a list written
    here: a catalogue is data, it is revised upstream, and an entry missing for
    one direction is a fact about the catalogue and not an error. Drawing from
    the index therefore draws only from waveforms that exist.

    File names are read as ``<model>_strains_<direction>.txt``.

    :type directory: str
    :param directory: the directory holding the strain files.
    :return: dict -- ``{model: {direction: path}}``, both keys sorted by the
        caller when a deterministic draw is needed.
    :raises FileNotFoundError: if the directory holds no file of that form.
    """
    import glob
    import os

    found = {}
    for path in sorted(glob.glob(os.path.join(directory, CCSN_FILE_PATTERN))):
        stem = os.path.basename(path)[: -len(".txt")]
        model, _, direction = stem.rpartition("_strains_")
        if not model or not direction:
            continue
        found.setdefault(model, {})[direction] = path

    if not found:
        raise FileNotFoundError(
            f"no file matching {CCSN_FILE_PATTERN} under {directory}")
    return found


def ccsn_polarisations(path, sample_rate=2048):
    """Plus and cross polarisations of one core-collapse supernova model.

    The file carries the time steps the simulation took, which are neither
    uniform nor the analysis rate: on this catalogue they vary by more than an
    order of magnitude within one waveform. Interpolating straight onto the
    analysis grid would fold everything above its Nyquist frequency back into
    the band, where it would be indistinguishable from signal. The series is
    therefore put first on a uniform grid fine enough to resolve the steps the
    simulation actually took, band limited there, and only then read off at the
    requested rate.

    The overall scale is discarded, as for every generator here: the injection
    code rescales to a requested signal-to-noise ratio, so only the shape and
    the relation between the two polarisations carry meaning. Unlike a circular
    binary's, these two are neither in phase nor scaled copies of one another,
    which is why the amplitude a detector receives cannot be written as a
    function of an inclination.

    :type path: str
    :param path: the strain file, three columns --- time in seconds from core
        bounce, then the two polarisations.
    :type sample_rate: int
    :param sample_rate: rate to return the series at, Hz.
    :return: tuple -- ``(hp, hc, start_offset)``, the two polarisations as
        arrays on a uniform grid at `sample_rate`, and the time of their first
        sample relative to core bounce, seconds. The reference time of the
        injection is the bounce, so `start_offset` is positive when the
        catalogue begins after it.
    :raises ValueError: if the file does not hold at least three columns, or
        holds fewer than two rows, or its time column does not increase.
    """
    from scipy.signal import butter, sosfiltfilt

    table = np.loadtxt(path)
    if table.ndim != 2 or table.shape[1] < 3 or table.shape[0] < 2:
        raise ValueError(
            f"{path} must hold at least two rows of (time, h_plus, h_cross)")

    times = np.asarray(table[:, 0], dtype=float)
    steps = np.diff(times)
    if not np.all(steps > 0.0):
        raise ValueError(f"{path} has a time column that does not increase")

    sample_rate = float(sample_rate)

    # Fine enough to resolve the simulation's own typical step, and a whole
    # multiple of the target so the final read-off is a decimation of it.
    oversample = max(int(np.ceil(1.0 / (float(np.median(steps)) * sample_rate))), 1)
    fine_rate = oversample * sample_rate

    fine = np.arange(times[0], times[-1], 1.0 / fine_rate)
    columns = [np.interp(fine, times, np.asarray(table[:, k], dtype=float))
               for k in (1, 2)]

    if oversample > 1:
        # Below the target Nyquist, with the transition band inside it: what
        # leaving it in would add is power folded to a frequency it was never
        # emitted at.
        sos = butter(8, 0.45 * sample_rate / (0.5 * fine_rate),
                     btype="lowpass", output="sos")
        columns = [sosfiltfilt(sos, column) for column in columns]

    coarse = np.arange(times[0], times[-1], 1.0 / sample_rate)
    hp, hc = (np.interp(coarse, fine, column) for column in columns)
    return hp, hc, float(times[0])


def _symmetric_times(half_length, sample_rate):
    """Time array symmetric about zero, spanning +/- `half_length` seconds."""
    n = int(round(half_length * sample_rate))
    return np.arange(-n, n + 1) / float(sample_rate)
