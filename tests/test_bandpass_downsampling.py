"""Regression tests for BandPassDownSampling actually honoring the band-pass
parameters a run configures -- Parameters.LowFrequencyCut and
Parameters.FilterOrder -- instead of silently using its own signature
defaults regardless.

Both had the same failure mode: the value was written into the run config (and
saved into parametersUsed.json) but never read back by any code. With the
highpass silently pinned at 4 Hz instead of a configured 10 Hz, seismic-band
power below the intended cutoff leaked into the coarsest wavelet coefficient
band, dominating basis selection on real H1/L1 data. The filter order matters
for the same reason: the highpass corner sits on the steep part of the seismic
wall, so the roll-off rate sets how much sub-cutoff power survives into
AR whitening.
"""
from wdf.config.Parameters import Parameters
from wdf.processes.BandPassDownSampling import BandPassDownSampling

BASE_KWARGS = dict(sampling=2048, resampling=256, ResamplingFactor=8)


def test_low_freq_hp_uses_parameters_low_frequency_cut_when_present():
    par = Parameters(LowFrequencyCut=10.0, **BASE_KWARGS)
    ds = BandPassDownSampling(par, estimation=True)
    assert ds.low_freq_hp == 10.0


def test_low_freq_hp_falls_back_to_default_when_not_configured():
    par = Parameters(**BASE_KWARGS)
    ds = BandPassDownSampling(par, estimation=True)
    assert ds.low_freq_hp == 4.0


def test_order_uses_parameters_filter_order_when_present():
    par = Parameters(FilterOrder=8, **BASE_KWARGS)
    ds = BandPassDownSampling(par, estimation=True)
    assert ds.order == 8


def test_order_falls_back_to_default_when_not_configured():
    par = Parameters(**BASE_KWARGS)
    ds = BandPassDownSampling(par, estimation=True)
    assert ds.order == 5


def test_explicit_order_argument_overrides_configured_filter_order():
    par = Parameters(FilterOrder=8, **BASE_KWARGS)
    ds = BandPassDownSampling(par, order=3, estimation=True)
    assert ds.order == 3


def test_higher_order_attenuates_more_below_the_highpass_corner():
    """The point of wiring FilterOrder up: a steeper band-pass leaves less
    sub-cutoff (seismic-band) power for AR whitening to fight."""
    from scipy.signal import sosfreqz

    def gain_at(par, freq_hz):
        ds = BandPassDownSampling(par, estimation=True)
        w, h = sosfreqz(ds.sos, worN=[freq_hz], fs=ds.sampling)
        return abs(h[0])

    low = Parameters(LowFrequencyCut=10.0, FilterOrder=4, **BASE_KWARGS)
    high = Parameters(LowFrequencyCut=10.0, FilterOrder=10, **BASE_KWARGS)
    assert gain_at(high, 2.0) < gain_at(low, 2.0)
