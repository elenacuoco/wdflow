"""Whiten a stream at zero phase, and check that it worked.

Run it on any frame file::

    python examples/zero_phase_whitening_example.py <frame-or-ffl> <channel> <gps>
        [--config configs/offline_default.toml]
        [--example examples/zero_phase_whitening_example.toml]

The run configuration (`--config`) is loaded with
`wdf.config.Parameters.load_config` for the detector of the channel prefix
("H1:..." -> "H1"); it gives the noise model (`ARorder`, `learn`,
`ARFitContext`), the band-pass and the decimation (`ResamplingFactor`,
`LowFrequencyCut`, ...), the warm-up (`preWhite`) and the order of the
square-root filter (`SqrtWhiteningOrder`; unset: the larger of
``[whitening.zero_phase] sqrt_order`` and `ARorder`, as the worker takes it).
The example's own values -- the span checked, the gap before the fit, the read
block and the checks' settings -- are in `--example`.

The example fits the noise model once, builds the square-root filter from it,
streams the data through the whitening in blocks, and reports the three numbers
that say whether the conditioning is sound: the standard deviation on the noise
scale, the kurtosis, and the spectral flatness.
"""
import argparse
import os

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import numpy as np
from scipy.signal import welch

from py4tsa.tsa import FrameIChannel, SeqView_double_t as SV
from wdf.config.Parameters import load_config
from wdf.config.defaults import defaults, parameter
from wdf.processes.BandPassDownSampling import (BandPassDownSampling, SV_to_array,
                                                discard_edges, read_conditioned)
from wdf.processes.Whitening import Whitening
from wdf.processes.zero_phase_whitening import ZeroPhaseWhitening

HERE = os.path.dirname(os.path.abspath(__file__))
#: The run configuration loaded when none is given.
DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), "configs", "offline_default.toml")
#: The example's own values.
DEFAULT_EXAMPLE = os.path.join(HERE, "zero_phase_whitening_example.toml")


def make_parameters(config, frame, channel, gps):
    """The run configuration for the channel's detector, with the input given.

    :param config: run configuration file (TOML).
    :param frame: frame file or frame-file list.
    :param channel: channel name; its prefix ("H1:...") names the detector.
    :param gps: start of the stretch to whiten, GPS seconds.
    :return: wdf.config.Parameters
    """
    ifo = channel.split(":", 1)[0] if ":" in channel else None
    par = load_config(config, ifo, file=frame, channel=channel, gps=gps, gpsStart=gps)
    probe = SV()
    FrameIChannel(frame, channel, 1.0, gps).GetData(probe)
    par.sampling = int(round(1.0 / probe.GetSampling()))
    par.resampling = par.sampling // int(par.ResamplingFactor)
    return par


def main(frame, channel, gps, config=DEFAULT_CONFIG, example=DEFAULT_EXAMPLE):
    with open(example, "rb") as fh:
        ex = tomllib.load(fh)
    par = make_parameters(config, frame, channel, gps)
    fs = float(par.resampling)
    print(f"{channel} at {par.sampling} Hz, analysed at {fs:.0f} Hz "
          f"(configuration {par.ConfigFile})")

    # 1. Fit the noise model once, on `learn` seconds ending `fit_gap_s` before
    #    the span, read with `ARFitContext` seconds of real data on each side
    #    that the band-pass settles over and that are then dropped.
    learn_s = float(par.learn)
    context = float(parameter(par, "ARFitContext", "worker"))
    first = gps - ex["fit_gap_s"] - learn_s - context
    raw = SV()
    FrameIChannel(frame, channel, learn_s + 2.0 * context, first).GetData(raw)
    whitening_model = Whitening(par.ARorder)
    whitening_model.ParametersEstimate(
        discard_edges(BandPassDownSampling(par, estimation=True).Process(raw), context))
    ar = np.array([whitening_model.ADE.GetAR(j) for j in range(par.ARorder + 1)])
    sigma = whitening_model.GetSigma()
    print(f"AR({par.ARorder}) fitted on {learn_s:g} s, sigma = {sigma:.4e}")

    # 2. Build the square-root filter. This is the only place a transform is
    #    used, and it happens once, not per block. Its order is the run's
    #    SqrtWhiteningOrder; unset, the larger of the default and the AR order.
    sqrt_order = parameter(par, "SqrtWhiteningOrder", "worker")
    sqrt_order = (max(defaults("whitening.zero_phase")["sqrt_order"], par.ARorder)
                  if sqrt_order is None else int(sqrt_order))
    block = int(round(fs * ex["block_s"]))
    whitening = ZeroPhaseWhitening(ar, block, 0, order=sqrt_order)
    print(f"square-root order {sqrt_order}, latency {whitening.latency} samples "
          f"({1e3 * whitening.latency / fs:.0f} ms), "
          f"predicted whitened sigma {whitening.sigma:.4e}")

    # 3. Stream. The band-pass emits a block once the data following it has
    #    arrived (`read_conditioned`); the whitening's forward pass is causal,
    #    and its backward pass reads `whitening.latency` samples ahead, which
    #    is what `extra` reserves.
    ds = BandPassDownSampling(par)
    data, dataw = SV(), SV()
    extra = whitening.latency

    streaming = FrameIChannel(frame, channel, ex["block_s"], gps)
    for _ in range(int(np.ceil(float(par.preWhite) / ex["block_s"]))):
        whitening.Process(read_conditioned(streaming, data, ds), dataw)

    whitening.SetOutputSize(int(round(fs * ex["span_s"])), extra)
    while whitening.DataNeeded() < 0:
        whitening.Input(read_conditioned(streaming, data, ds))
    whitening.Output(dataw)
    whitened = SV_to_array(dataw)

    # 4. The three numbers that matter.
    frequencies, power = welch(whitened, fs=fs, nperseg=ex["welch_nperseg"])
    band = ((frequencies >= ex["flatness_f_low"])
            & (frequencies <= ex["flatness_nyquist_fraction"] * 0.5 * fs))
    flatness = np.exp(np.mean(np.log(power[band]))) / np.mean(power[band])
    kurtosis = ((whitened - whitened.mean()) ** 4).mean() / np.var(whitened) ** 2

    print(f"\n{len(whitened)} whitened samples from {dataw.GetStart():.3f}")
    print(f"  std / sigma = {np.std(whitened) / sigma:.4f}   (1 means unit variance)")
    print(f"  kurtosis    = {kurtosis:.3f}   (3 means Gaussian)")
    print(f"  flatness    = {flatness:.4f}   (1 means white)")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Whiten a stream at zero phase and report its standard deviation "
                    "on the noise scale, kurtosis and spectral flatness.")
    parser.add_argument("frame", help="frame file or frame-file list (.ffl)")
    parser.add_argument("channel", help='channel name, e.g. "H1:GWOSC-4KHZ_R1_STRAIN"; '
                                        "its prefix selects the [detectors.<IFO>] table")
    parser.add_argument("gps", type=float, help="start of the stretch to whiten, GPS seconds")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="run configuration, loaded with load_config (default: %(default)s)")
    parser.add_argument("--example", default=DEFAULT_EXAMPLE,
                        help="the example's own values (default: %(default)s)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.frame, args.channel, args.gps, config=args.config, example=args.example)
