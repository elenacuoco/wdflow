"""Whiten a stream at zero phase, and check that it worked.

Run it on any frame file::

    python examples/zero_phase_whitening_example.py <frame-or-ffl> <channel> <gps>

The example fits the noise model once, builds the square-root filter from it,
streams the data through the whitening in blocks, and reports the three numbers
that say whether the conditioning is sound: the standard deviation on the noise
scale, the kurtosis, and the spectral flatness.
"""
import sys

import numpy as np
from scipy.signal import welch

from pytsa.tsa import FrameIChannel, SeqView_double_t as SV
from wdf.config.Parameters import Parameters
from wdf.processes.BandPassDownSampling import BandPassDownSampling, SV_to_array
from wdf.processes.Whitening import Whitening
from wdf.processes.zero_phase_whitening import ZeroPhaseWhitening

LEARN_S = 120.0
SPAN_S = 60.0
SQRT_ORDER = 256


def make_parameters(frame, channel, gps, sampling, resampling_factor=2):
    """Minimal run configuration for the conditioning front end."""
    par = Parameters()
    par.file, par.channel = frame, channel
    par.gps = par.gpsStart = gps
    par.sampling = sampling
    par.ResamplingFactor = resampling_factor
    par.resampling = sampling // resampling_factor
    par.ARorder, par.learn, par.preWhite = 1000, int(LEARN_S), 30
    par.LowFrequencyCut, par.FilterOrder = 10.0, 6
    return par


def main(frame, channel, gps):
    probe = SV()
    FrameIChannel(frame, channel, 1.0, gps).GetData(probe)
    sampling = int(round(1.0 / probe.GetSampling()))
    par = make_parameters(frame, channel, gps, sampling)
    fs = float(par.resampling)
    print(f"{channel} at {sampling} Hz, analysed at {fs:.0f} Hz")

    # 1. Fit the noise model once, on data preceding the span.
    whitening_model = Whitening(par.ARorder)
    learn = SV()
    FrameIChannel(frame, channel, LEARN_S, gps - LEARN_S - 10.0).GetData(learn)
    whitening_model.ParametersEstimate(
        BandPassDownSampling(par, estimation=True).Process(learn))
    ar = np.array([whitening_model.ADE.GetAR(j) for j in range(par.ARorder + 1)])
    sigma = whitening_model.GetSigma()
    print(f"AR({par.ARorder}) fitted, sigma = {sigma:.4e}")

    # 2. Build the square-root filter. This is the only place a transform is
    #    used, and it happens once, not per block.
    whitening = ZeroPhaseWhitening(ar, int(par.resampling), 0, order=SQRT_ORDER)
    print(f"square-root order {SQRT_ORDER}, latency {whitening.latency} samples "
          f"({1e3 * whitening.latency / fs:.0f} ms), "
          f"predicted whitened sigma {whitening.sigma:.4e}")

    # 3. Stream. The forward pass is causal; the backward pass reads
    #    `whitening.latency` samples ahead, which is what `extra` reserves.
    ds = BandPassDownSampling(par)
    data, dataw = SV(), SV()
    extra = whitening.latency

    streaming = FrameIChannel(frame, channel, 1.0, gps)
    for _ in range(par.preWhite):
        streaming.GetData(data)
        whitening.Process(ds.Process(data), dataw)

    for _ in range(-(-extra // int(par.resampling)) or 1):
        streaming.GetData(data)
        whitening.Input(ds.Process(data))

    streaming.SetDataLength(SPAN_S)
    whitening.SetOutputSize(int(par.resampling * SPAN_S), extra)
    streaming.GetData(data)
    whitening.Process(ds.Process(data), dataw)
    whitened = SV_to_array(dataw)

    # 4. The three numbers that matter.
    frequencies, power = welch(whitened, fs=fs, nperseg=4096)
    band = (frequencies >= 20.0) & (frequencies <= 0.45 * fs)
    flatness = np.exp(np.mean(np.log(power[band]))) / np.mean(power[band])
    kurtosis = ((whitened - whitened.mean()) ** 4).mean() / np.var(whitened) ** 2

    print(f"\n{len(whitened)} whitened samples from {dataw.GetStart():.3f}")
    print(f"  std / sigma = {np.std(whitened) / sigma:.4f}   (1 means unit variance)")
    print(f"  kurtosis    = {kurtosis:.3f}   (3 means Gaussian)")
    print(f"  flatness    = {flatness:.4f}   (1 means white)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
