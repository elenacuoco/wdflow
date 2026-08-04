"""Coloured Gaussian detector noise for mock data sets."""
from __future__ import annotations

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
