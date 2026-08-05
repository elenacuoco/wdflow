import logging
import numpy as np
 
from pytsa.tsa import  WaveletTransform
from wdf.observers.observable import Observable
from wdf.observers.observer import Observer
from wdf.structures.array2SeqView import array2SeqView
from wdf.structures.eventPE import eventPE
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 


def _as_finite_1d(signal):
    """Return a finite one-dimensional float array."""
    x = np.asarray(signal, dtype=float).reshape(-1)

    if x.size == 0:
        return x

    return np.where(np.isfinite(x), x, 0.0)


def energy_interval(
    signal,
    energy_fraction=0.90,
):
    """
    Locate the central interval containing ``energy_fraction`` of the
    reconstructed time-domain energy.

    The default corresponds to the interval between the 5% and 95%
    cumulative-energy quantiles.

    Parameters
    ----------
    signal : array-like
        Reconstructed time-domain signal.
    energy_fraction : float
        Fraction of total energy included in the interval.

    Returns
    -------
    tuple
        ``i_start, i_end``, both inclusive. ``None, None`` when the signal is
        empty or carries no energy.
    """
    x = _as_finite_1d(signal)

    if x.size == 0:
        return None, None

    energy = x * x
    total_energy = float(np.sum(energy))

    if total_energy <= np.finfo(float).tiny:
        return None, None

    fraction = float(np.clip(energy_fraction, 0.0, 1.0))
    tail = 0.5 * (1.0 - fraction)

    cumulative = np.cumsum(energy) / total_energy

    i_start = int(np.searchsorted(cumulative, tail, side="left"))
    i_end = int(np.searchsorted(cumulative, 1.0 - tail, side="left"))

    i_start = int(np.clip(i_start, 0, x.size - 1))
    i_end = int(np.clip(i_end, i_start, x.size - 1))

    return i_start, i_end


def estimate_duration(
    signal,
    fs,
    energy_fraction=0.90,
):
    """
    Estimate the signal duration from the central interval containing
    ``energy_fraction`` of the reconstructed time-domain energy.

    Parameters
    ----------
    signal : array-like
        Reconstructed time-domain signal.
    fs : float
        Sampling frequency in Hz.
    energy_fraction : float
        Fraction of total energy included in the duration estimate.

    Returns
    -------
    float
        Duration in seconds.
    """
    if not np.isfinite(fs) or fs <= 0:
        return 0.0

    i_start, i_end = energy_interval(signal, energy_fraction=energy_fraction)

    if i_start is None:
        return 0.0

    # Add one sample because the selected interval contains both endpoints.
    return float((i_end - i_start + 1) / fs)


def get_most_important_frequencies(
    signal,
    fs,
    alpha=0.10,
    f_low=0.0,
    f_high=None,
):
    """
    Estimate robust frequency metaparameters from the reconstructed signal.

    ``freqMin`` and ``freqMax`` delimit the central ``1 - alpha`` fraction
    of spectral energy. ``freqMean`` is the energy-weighted frequency and
    ``freqPeak`` is the maximum-power frequency.

    The mean is removed before the FFT, so the DC bin cannot produce an
    artificial ``freqMin = 0``.

    Parameters
    ----------
    signal : array-like
        Reconstructed time-domain signal.
    fs : float
        Sampling frequency in Hz.
    alpha : float
        Fraction of spectral energy excluded from the two tails.
        With alpha=0.10, freqMin/freqMax contain 90% of the energy.
    f_low : float
        Lowest physically allowed frequency.
    f_high : float or None
        Highest physically allowed frequency. Defaults to Nyquist.

    Returns
    -------
    tuple
        ``freqMean, freqMin, freqMax, freqPeak``.
    """
    x = _as_finite_1d(signal)

    if x.size < 2 or not np.isfinite(fs) or fs <= 0:
        value = max(float(f_low), 0.0)
        return value, value, value, value

    nyquist = 0.5 * float(fs)

    f_low = float(np.clip(f_low, 0.0, nyquist))

    if f_high is None:
        f_high = nyquist
    else:
        f_high = float(np.clip(f_high, f_low, nyquist))

    # Remove DC and reduce spectral leakage.
    x = x - np.mean(x)

    total_time_energy = float(np.dot(x, x))
    if total_time_energy <= np.finfo(float).tiny:
        return f_low, f_low, f_low, f_low

    if x.size > 2:
        window = np.hanning(x.size)
        x_fft = x * window
    else:
        x_fft = x

    spectrum = np.fft.rfft(x_fft)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)

    valid = (
        np.isfinite(freqs)
        & np.isfinite(power)
        & (freqs >= f_low)
        & (freqs <= f_high)
    )

    freqs = freqs[valid]
    power = power[valid]

    if freqs.size == 0:
        return f_low, f_low, f_low, f_low

    total_power = float(np.sum(power))

    if total_power <= np.finfo(float).tiny:
        return f_low, f_low, f_low, f_low

    power = power / total_power
    cumulative = np.cumsum(power)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    lower_quantile = 0.5 * alpha
    upper_quantile = 1.0 - lower_quantile

    i_min = int(np.searchsorted(cumulative, lower_quantile, side="left"))
    i_max = int(np.searchsorted(cumulative, upper_quantile, side="left"))

    i_min = int(np.clip(i_min, 0, freqs.size - 1))
    i_max = int(np.clip(i_max, i_min, freqs.size - 1))

    freq_min = float(freqs[i_min])
    freq_max = float(freqs[i_max])
    freq_mean = float(np.sum(freqs * power))
    freq_peak = float(freqs[int(np.argmax(power))])

    # Numerical safety and ordering.
    freq_min = float(np.clip(freq_min, f_low, f_high))
    freq_max = float(np.clip(freq_max, freq_min, f_high))
    freq_mean = float(np.clip(freq_mean, freq_min, freq_max))
    freq_peak = float(np.clip(freq_peak, freq_min, freq_max))

    return freq_mean, freq_min, freq_max, freq_peak


def extract_meta_features(
    sigIn,
    fs,
    sigma=None,
    EnWDF=None,
    f_low=0.0,
    f_high=None,
    energy_fraction=0.90,
    spectral_alpha=0.10,
):
    """
    Extract robust time-domain and frequency-domain trigger metaparameters.

    ``EnWDF`` remains the detection statistic. ``snrMean`` and ``snrPeak``
    are derived from EnWDF and the shape of the reconstructed waveform,
    rather than independently dividing by a potentially inconsistent sigma.

    ``snrMean`` is the root-mean-square amplitude over the interval the
    duration is measured on, not over the whole analysis window: a transient
    occupying a few samples of the window has a mean amplitude set by its own
    support, and averaging over the window instead dilutes it by the square
    root of the ratio of the two lengths.

    This guarantees, up to numerical precision,

        snrMean <= snrPeak <= EnWDF

    for a nonzero reconstruction.

    Parameters
    ----------
    sigIn : array-like
        Reconstructed time-domain signal.
    fs : float
        Sampling frequency in Hz.
    sigma : float or None
        Kept for API compatibility. It is used only as a fallback when
        EnWDF is not supplied.
    EnWDF : float or None
        WDF normalized wavelet-energy detection statistic.
    f_low, f_high : float
        Analysis band.
    energy_fraction : float
        Fraction of time-domain energy used for the duration.
    spectral_alpha : float
        Fraction of spectral energy excluded from the frequency tails.

    Returns
    -------
    tuple
        tPeak, duration, freqMin, freqMean, freqMax, freqPeak,
        snrMean, snrPeak
    """
    x = _as_finite_1d(sigIn)

    if x.size == 0 or not np.isfinite(fs) or fs <= 0:
        base_frequency = max(float(f_low), 0.0)
        return (
            0.0,
            0.0,
            base_frequency,
            base_frequency,
            base_frequency,
            base_frequency,
            0.0,
            0.0,
        )

    absolute = np.abs(x)
    peak_index = int(np.argmax(absolute))
    t_peak = float(peak_index / fs)

    i_start, i_end = energy_interval(x, energy_fraction=energy_fraction)

    if i_start is None:
        duration = 0.0
        support = x
    else:
        duration = float((i_end - i_start + 1) / fs)
        support = x[i_start:i_end + 1]

    freq_mean, freq_min, freq_max, freq_peak = (
        get_most_important_frequencies(
            x,
            fs,
            alpha=spectral_alpha,
            f_low=f_low,
            f_high=f_high,
        )
    )

    l2_norm = float(np.linalg.norm(x))

    if EnWDF is not None and np.isfinite(EnWDF) and EnWDF >= 0:
        normalized_energy = float(EnWDF)
    elif (
        sigma is not None
        and np.isfinite(sigma)
        and sigma > np.finfo(float).tiny
    ):
        # Backward-compatible fallback.
        normalized_energy = l2_norm / float(sigma)
    else:
        normalized_energy = 0.0

    if l2_norm <= np.finfo(float).tiny or normalized_energy == 0.0:
        snr_mean = 0.0
        snr_peak = 0.0
    else:
        # Map the reconstructed waveform shape onto the EnWDF scale.
        normalization = normalized_energy / l2_norm

        snr_mean = float(
            np.sqrt(np.mean(support * support)) * normalization
        )
        snr_peak = float(
            np.max(absolute) * normalization
        )

    return (
        t_peak,
        duration,
        freq_min,
        freq_mean,
        freq_max,
        freq_peak,
        snr_mean,
        snr_peak,
    )

class ParameterEstimation(Observer, Observable):

    """
    This class stands for the parameter estimation of the Sequence View data

    """

    def __init__(self, parameters):
        Observable.__init__(self)
        Observer.__init__(self)

        if parameters.ResamplingFactor is not None:
            self.sampling = (
                parameters.sampling / parameters.ResamplingFactor
            )
        else:
            self.sampling = parameters.sampling

        self.Ncoeff = parameters.Ncoeff
        self.sigma = parameters.sigma

        # The current downsampling stage uses low_freq_hp.
        self.f_low = float(
            getattr(
                parameters,
                "low_freq_hp",
                getattr(parameters, "f_low", 0.0),
            )
        )

        configured_f_high = getattr(
            parameters,
            "f_high",
            None,
        )

        nyquist = 0.5 * float(self.sampling)

        if configured_f_high is None:
            self.f_high = nyquist
        else:
            self.f_high = min(
                float(configured_f_high),
                nyquist,
            ) 

    def update(self, event):
        """
        This method estimates parameters of the triggers from the Sequence View data

        :type event: object
        :param event: An object to be analysed to get triggers
        :return: An object storing triggers
        """
        wave = event.mWave
        t0 = event.mTime
        coeff = np.zeros(self.Ncoeff)
        Icoeff = np.zeros(self.Ncoeff)
        for i in range(self.Ncoeff):
            coeff[i] = event.GetCoeff(i)

        data = array2SeqView(t0, self.sampling, self.Ncoeff)
        data = data.Fill(t0, coeff)
    
        wt = getattr(WaveletTransform, wave)
        WT = WaveletTransform(self.Ncoeff, wt)
        WT.Inverse(data)
        for i in range(self.Ncoeff):
            Icoeff[i] = data.GetY(0, i)
        

        EnWDF = float(event.mSNR)
        sigma = float(event.mSigma)

        (
            tPeak,
            duration,
            freqMin,
            freqMean,
            freqMax,
            freqPeak,
            snrMean,
            snrPeak,
        ) = extract_meta_features(
            Icoeff,
            fs=self.sampling,
            sigma=sigma,
            EnWDF=EnWDF,
            f_low=self.f_low,
            f_high=self.f_high,
            energy_fraction=0.90,
            spectral_alpha=0.10,
        )
        
        # the gps of the signal is identified by WDF as the t0 of analyzing window
        gps=t0
        
        #the gpsPeak of the signal is identified by WDF as the time  of analyzing window at which the signal is maximum
        gpsPeak=t0+tPeak
        
       
         
        eventParameters = eventPE(
            gps, gpsPeak, duration, EnWDF, sigma, snrMean, snrPeak, freqMin, freqMean, freqMax, freqPeak, wave, coeff, Icoeff)

        self.update_observers(eventParameters)
