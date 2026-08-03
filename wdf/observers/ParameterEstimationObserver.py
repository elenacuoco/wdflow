import logging
import numpy as np
 
from pytsa.tsa import  WaveletTransform
from wdf.observers.observable import Observable
from wdf.observers.observer import Observer
from wdf.processes.wavelet_energy import wavelet_energy_snr
from wdf.structures.array2SeqView import array2SeqView
from wdf.structures.eventPE import eventPE
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 

def estimate_duration(signal, fs):
    """Estimates the duration of the most relevant part of a signal around its maximum amplitude.

    Args:
        signal (array-like): Input signal.
        fs (float): Sampling frequency in Hz.

    Returns:
        float: Duration in seconds of the most relevant part of the signal around its maximum amplitude.
    """
    # Find the index of the maximum amplitude in the signal
    max_index = np.argmax(np.abs(signal))

    # Compute the threshold for the relevant part of the signal
    threshold = 0.1 * np.abs(signal[max_index])

    # Find the indices where the signal exceeds the threshold
    indices = np.where(np.abs(signal) >= threshold)[0]

    # Compute the duration of the relevant part of the signal
    if len(indices)>0:
        duration = (indices[-1] - indices[0]) / fs
    else:
        duration=0.    

    return duration

def get_most_important_frequencies(signal, fs, alpha=.1):
    """
    Returns the most important frequencies present in a signal with sampling rate fs.
    signal: array_like
        The input signal.
    fs: float
        The sampling rate of the signal.
    alpha: float
        the percentile to select frequencies 
    Returns
    -------
    ...
    """
    # Compute the FFT of the signal
    signal_fft = np.fft.rfft(signal)
    # Compute the corresponding frequency values
    freqs = np.fft.rfftfreq(len(signal)) * fs
    # Find the magnitudes of the FFT coefficients
    magnitudes = np.abs(signal_fft)
    # Find the frequency at maximum magnitude
    freqPeak = freqs[np.argmax(magnitudes)]
    # Significance region. >= (not >): sig_mag is itself within [min(magnitudes),
    # max(magnitudes)], so the bin(s) at the maximum always qualify -- with a
    # strict >, a degenerate spectrum (e.g. every bin tied at the same
    # magnitude, such as for an all-zero or single-sample signal) selects zero
    # bins instead.
    sig_mag = np.percentile(magnitudes,100*(1-alpha))
    sig_freqs = freqs[magnitudes>=sig_mag]
    # Calculate the mean, minimum, and maximum frequencies in the significance region
    freqMin = np.min(sig_freqs)
    freqMax = np.max(sig_freqs)
    freqMean= np.mean(sig_freqs)
    # Return the top num_freqs frequencies and their magnitudes
    return freqMean, freqMin, freqMax, freqPeak 
 

def extract_meta_features(sigIn, fs):
    """Timing/frequency features of the reconstructed (inverse-transformed)
    trigger waveform `sigIn`: time and duration of its most relevant part
    around its peak amplitude, and the frequency band containing its
    strongest FFT content. Does not include an SNR estimate -- see
    `wdf.processes.wavelet_energy.wavelet_energy_snr` for that, computed
    directly from the wavelet-domain coefficients instead.
    """
    # tPeak: The time at which the signal has its maximum absolute value.
    tPeak = np.argmax(np.abs(sigIn)) / fs

    # Calculate the duration of the signal in seconds
    duration = estimate_duration(sigIn, fs)


    # freqMean: The mean frequency of the signal.
    # freqMin: The minimum frequency of the signal.
    #freqMax: The maximum frequency of the signal.
    freqMean, freqMin, freqMax, freqPeak =  get_most_important_frequencies(sigIn, fs, alpha=.1)

    return  tPeak, duration, freqMin, freqMean, freqMax, freqPeak


class ParameterEstimation(Observer, Observable):

    """
    This class stands for the parameter estimation of the Sequence View data

    """

    def __init__(self, parameters):
        """
        :type parameters: class Parameters object
        """

        Observable.__init__(self)
        Observer.__init__(self)
        
       
        if parameters.ResamplingFactor is not None:
            self.sampling = parameters.sampling/parameters.ResamplingFactor
        else:
            self.sampling = parameters.sampling
            
        self.Ncoeff = parameters.Ncoeff

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
        

        # EnWDF is WDF's own internal detection statistic: the thresholded,
        # per-window/per-basis RMS-over-local-sigma value that decided
        # whether this trigger fired. snrMean/snrPeak recompute the same
        # energy statistic here using event.mSigma -- the winning basis's
        # own per-window sigma, exactly the value EnWDF was normalized by
        # (see WDF2Classify::GetDataVector) -- instead of self.sigma (a
        # single sigma frozen at Learn time from time-domain whitened
        # residuals). Using self.sigma here used to silently drift out of
        # sync with EnWDF's convention as real noise moved away from what
        # Learn saw at t=0.
        EnWDF = event.mSNR
        tPeak, duration, freqMin, freqMean, freqMax, freqPeak = extract_meta_features(Icoeff, self.sampling)

        snr = wavelet_energy_snr(coeff, event.mSigma)["snr"]
        snrMean = snrPeak = snr

        # the gps of the signal is identified by WDF as the t0 of analyzing window
        gps=t0
        
        #the gpsPeak of the signal is identified by WDF as the time  of analyzing window at which the signal is maximum
        gpsPeak=t0+tPeak
        
       
         
        eventParameters = eventPE(
            gps, gpsPeak, duration, EnWDF, snrMean, snrPeak, freqMin, freqMean, freqMax, freqPeak, wave, coeff, Icoeff)

        self.update_observers(eventParameters)
