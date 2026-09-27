"""Gaussian detector noise for mock data sets: coloured, or already whitened.

Coloured noise is what a detector records and what a search has to condition.
White noise is what the conditioning is meant to produce: unit variance, flat
at every frequency, the same in every detector. A set written in white noise
therefore measures the search alone, with the whitening taken out of the
comparison --- a difference between two search configurations on it cannot be
a difference in how well the data were conditioned.
"""
from __future__ import annotations

import numpy as np

DEFAULT_PSD = "aLIGOZeroDetHighPower"


def analytic_psd(length, delta_f, low_frequency_cutoff, psd_name=DEFAULT_PSD):
    """Analytic detector power spectral density.

    :type length: int
    :param length: number of frequency samples.
    :type delta_f: float
    :param delta_f: frequency resolution, Hz.
    :type low_frequency_cutoff: float
    :param low_frequency_cutoff: frequency below which the PSD is not defined, Hz.
    :type psd_name: str
    :param psd_name: name of any analytic PSD provided by `pycbc.psd`.
    :return: pycbc.types.FrequencySeries -- the PSD.
    """
    from pycbc.psd import from_string

    return from_string(psd_name, length, delta_f, low_frequency_cutoff)


def coloured_noise(start_time, end_time, seed=0, sample_rate=2048,
                   low_frequency_cutoff=5.0, psd_name=DEFAULT_PSD,
                   filter_duration=128):
    """Gaussian noise coloured by an analytic PSD, reproducible from `seed`.

    Generation is continuous across the whole span, so there are no
    discontinuities that a transient search would flag.

    :type start_time: float
    :param start_time: GPS time of the first sample.
    :type end_time: float
    :param end_time: GPS time at which the series ends.
    :type seed: int
    :param seed: seed fixing the noise realisation.
    :type sample_rate: int
    :param sample_rate: sampling rate, Hz.
    :type low_frequency_cutoff: float
    :param low_frequency_cutoff: frequency below which no noise is generated, Hz.
    :type psd_name: str
    :param psd_name: name of any analytic PSD provided by `pycbc.psd`.
    :type filter_duration: float
    :param filter_duration: length of the colouring filter, seconds.
    :return: pycbc.types.TimeSeries -- the noise, starting at `start_time`.
    """
    from pycbc.noise.reproduceable import colored_noise

    delta_f = 1.0 / filter_duration
    length = int(sample_rate / 2 / delta_f) + 1
    psd = analytic_psd(length, delta_f, low_frequency_cutoff, psd_name)
    return colored_noise(psd, start_time, end_time, seed=seed,
                         sample_rate=sample_rate,
                         low_frequency_cutoff=low_frequency_cutoff,
                         filter_duration=filter_duration)


def white_psd(length, delta_f, sample_rate):
    """One-sided power spectral density of unit-variance white noise.

    A variance of one spread evenly over ``[0, sample_rate / 2]`` is a density
    of ``2 / sample_rate`` at every frequency. This is the spectrum an injection
    into :func:`white_noise` is scaled against, so the signal-to-noise ratio a
    truth table records is the one the samples carry.

    :type length: int
    :param length: number of frequency samples.
    :type delta_f: float
    :param delta_f: frequency resolution, Hz.
    :type sample_rate: float
    :param sample_rate: rate of the white series, Hz.
    :return: pycbc.types.FrequencySeries -- the flat density.
    """
    from pycbc.types import FrequencySeries

    return FrequencySeries(np.full(int(length), 2.0 / float(sample_rate)),
                           delta_f=float(delta_f))


def white_noise(start_time, end_time, seed=0, sample_rate=2048):
    """Unit-variance white Gaussian noise, reproducible from `seed`.

    Independent samples of a standard normal, which is what a perfectly
    whitened detector stream is at the rate it is searched at. Nothing is
    filtered and nothing is band limited: the spectrum is flat up to Nyquist
    and the samples are Gaussian in every band by construction, so what a
    search finds in it that is not injected is the noise floor and nothing
    else.

    :type start_time: float
    :param start_time: GPS time of the first sample.
    :type end_time: float
    :param end_time: GPS time at which the series ends.
    :type seed: int
    :param seed: seed fixing the noise realisation.
    :type sample_rate: int
    :param sample_rate: sampling rate, Hz.
    :return: pycbc.types.TimeSeries -- the noise, starting at `start_time`.
    """
    from pycbc.types import TimeSeries

    n = int(round((float(end_time) - float(start_time)) * float(sample_rate)))
    rng = np.random.default_rng(seed)
    return TimeSeries(rng.standard_normal(n), delta_t=1.0 / float(sample_rate),
                      epoch=float(start_time))
