"""The standard set new search ideas are compared on.

Three detectors, H1, L1 and V1, in unit-variance white Gaussian noise at the
rate the search runs at, with low signal-to-noise compact binaries injected at
known times, and the same noise without them beside it. Built entirely by
:func:`wdf.mock.generate_dataset`; this module only fixes its configuration,
writes it where the set is, and validates what was written.

Why white. A search is a chain, and on real or coloured data a change to one
link is measured through all of them: a wavelet basis that looks better may be
one the whitening happened to suit. In white noise the conditioning has nothing
to do, so a difference between two configurations is a difference in the
search. The worker still conditions the stream --- it band-passes and fits an
autoregressive whitening filter to whatever it is given --- and on white noise
the fit is the identity to within its estimation error, which is the same for
every configuration compared.

Why these signals. The population is the one a burst search has to be judged
on where it is hardest, near threshold: network signal-to-noise ratio from 6
to 20, with most of it between 7 and 12, and chirp masses from 5 to 30 solar
masses in the detector frame, from tracks tens of seconds long to tracks a
fraction of a second long. Every source is placed on the sky and projected, so
each detector has its own antenna response and its own arrival time within the
light travel time of the others. Virgo receives each source at
:data:`V1_RELATIVE_SENSITIVITY` of the amplitude a LIGO detector with the same
antenna response would, the ratio of its sensitivity to theirs; a search
that treats the three detectors alike is tested on a network that is not.

The chirp is a compact binary and the search must not know it. The set holds
one morphology because the question it answers is recovery near threshold; a
configuration that wins on it by preferring rising frequency has learnt the
population and not the noise, and is not a better unmodelled search.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

# Virgo's amplitude sensitivity relative to the LIGO detectors, as measured
# and supplied for this benchmark.
V1_RELATIVE_SENSITIVITY = 0.32

BENCHMARK_CONFIG = {
    "duration": 21600.0,
    "start_gps": 1400000000.0,
    "sample_rate": 2048,
    "n_cbc": 300,
    "n_glitch": 0,
    "snr_range": (6.0, 20.0),
    "snr_core": (7.0, 12.0, 0.6),
    "seed": 20260927,
    "detectors": ("H1", "L1", "V1"),
    "edge_pad": 500.0,
    "low_frequency_cutoff": 5.0,
    "noise": "white",
    "relative_sensitivity": {"V1": V1_RELATIVE_SENSITIVITY},
    "cbc_population": {
        "chirp_mass": (5.0, 30.0),
        "mass_ratio": (1.0, 4.0),
        "spin": (-0.5, 0.5),
        "f_lower": 20.0,
        "approximant": "IMRPhenomD",
    },
    "minimum_injection_gap": 20.0,
    "channel_suffix": "MOCK-STRAIN",
    "frame_length": 1024.0,
    "write_background": True,
    "track_points": 64,
}

# The band the worker passes at the benchmark's rate, for the record of how
# much of each injection's amplitude lies outside it: its band-pass runs from
# `LowFrequencyCut` to 0.9 of the Nyquist frequency of the stream it searches
# (wdf.processes.BandPassDownSampling).
WORKER_BAND = (12.0, 0.9 * 0.5 * 2048)


def _provenance():
    """What produced the set: the library's commit and the waveform code."""
    import pycbc

    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(
            ["git", "-C", here, "rev-parse", "HEAD"], capture_output=True,
            text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", here, "status", "--porcelain", "--", "."],
            capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"wdflow_commit": commit, "wdf_mock_modified": dirty,
            "pycbc": pycbc.__version__, "numpy": np.__version__}


def worker_read_check(outdir, detectors, channel_suffix="MOCK-STRAIN",
                      times=None):
    """Read the frames through the worker's own reader and compare.

    The worker reads a stream with ``py4tsa``'s ``FrameIChannel`` from an FFL,
    which is not the reader the other checks use. A block of one second is
    read with it at each of `times`, in every detector and both kinds, and
    compared sample by sample with the same second read by gwpy.

    :type outdir: str
    :param outdir: where the set is.
    :param detectors: its detectors.
    :type channel_suffix: str
    :param channel_suffix: channel name after the detector prefix.
    :param times: GPS times to read at; three spread over the set when None.
    :return: float -- the largest absolute difference found.
    """
    from py4tsa.tsa import FrameIChannel
    from py4tsa.tsa import SeqView_double_t as SV

    from wdf.mock.validation import read_ffl

    worst = 0.0
    for ifo in detectors:
        channel = f"{ifo}:{channel_suffix}"
        for kind in ("FOREGROUND", "BACKGROUND"):
            ffl = os.path.join(outdir, f"{ifo}-MOCK-{kind}.ffl")
            if times is None:
                with open(ffl, encoding="utf-8") as handle:
                    lines = [line.split() for line in handle if line.strip()]
                first = float(lines[0][1])
                last = float(lines[-1][1]) + float(lines[-1][2])
                # Across a frame boundary as well as inside frames.
                times = (first + 1.0, first + float(lines[0][2]) - 0.5,
                         0.5 * (first + last))
            for gps in times:
                block = SV()
                FrameIChannel(ffl, channel, 1.0, float(gps)).GetData(block)
                worker = np.array([block.GetY(0, i)
                                   for i in range(block.GetSize())])
                reference, _, _ = read_ffl(ffl, channel, float(gps),
                                           float(gps) + 1.0)
                if worker.size != reference.size:
                    raise RuntimeError(
                        f"{ffl} at {gps}: the worker read {worker.size} "
                        f"samples, gwpy {reference.size}")
                worst = max(worst, float(np.abs(worker - reference).max()))
    return worst


def _band_loss(truth, detectors, relative_sensitivity, sample_rate, band):
    """Share of each injection's amplitude outside the band the worker passes."""
    from wdf.mock.dataset import optimal_snr
    from wdf.mock.noise import white_psd
    from wdf.mock.validation import true_template

    psd = white_psd(int(0.5 * sample_rate * 16) + 1, 1.0 / 16.0, sample_rate)
    losses = []
    for _, row in truth[truth["category"] == "cbc"].iterrows():
        for ifo in detectors:
            template, _ = true_template(row, ifo, sample_rate,
                                        relative_sensitivity)
            full = optimal_snr(template, sample_rate, 0.0, psd=psd)
            inside = optimal_snr(template, sample_rate, band[0], band[1],
                                 psd=psd)
            losses.append(1.0 - inside / full)
    return np.asarray(losses)


def _summarise(report, truth, losses):
    """The validation, reduced to what goes in a file and a README."""
    mf = report["matched_filter"]
    out = {"light_travel": report["light_travel"], "spectrum": {},
           "gaussianity": {}, "matched_filter": {}}
    for ifo, kinds in report["spectrum"].items():
        out["spectrum"][ifo] = kinds
        out["gaussianity"][ifo] = report["gaussianity"][ifo]
        rows = mf[mf["detector"] == ifo]
        residual = rows["recovered"] - rows["snr"]
        out["matched_filter"][ifo] = {
            "n": int(len(rows)),
            "injected_over_recorded_max_deviation": float(
                (rows["injected"] / rows["snr"] - 1.0).abs().max()),
            "injected_lag_samples_max": int(rows["injected_lag"].abs().max()),
            "recovered_minus_recorded_mean": float(residual.mean()),
            "recovered_minus_recorded_std": float(residual.std()),
            "recorded_snr_median": float(rows["snr"].median()),
        }
    cbc = truth[truth["category"] == "cbc"]
    out["population"] = {
        "n": int(len(cbc)),
        "network_snr_range": (float(cbc["network_snr"].min()),
                              float(cbc["network_snr"].max())),
        "share_network_snr_7_12": float(
            cbc["network_snr"].between(7.0, 12.0).mean()),
        "chirp_mass_range": (float(cbc["chirp_mass"].min()),
                             float(cbc["chirp_mass"].max())),
        "v1_over_ligo_rms_median": float(
            (cbc["snr_V1"] / np.hypot(cbc["snr_H1"], cbc["snr_L1"])
             * np.sqrt(2.0)).median()),
    }
    out["worker_band_loss"] = {
        "band": WORKER_BAND,
        "median": float(np.median(losses)),
        "max": float(losses.max()),
    }
    return out


README = """# WDF benchmark data: three detectors, white noise, low-SNR chirps

Written by `wdf.mock.benchmark.write_benchmark` (wdflow, branch
`feature/three-detector-white-mock`). Do not edit by hand: regenerate.

## Regenerate

    python -m wdf.mock.benchmark {outdir}

The set is a function of `benchmark_config.json` and nothing else: the seed
fixes the noise and the injections, so the same configuration writes the same
samples. `provenance.json` records the wdflow commit it was written from.

## What is here

- `<IFO>-MOCK-FOREGROUND.ffl`, `<IFO>-MOCK-FOREGROUND/`: noise plus injections,
  GWF frames of {frame_length:g} s, channel `<IFO>:{channel_suffix}`, {sample_rate} Hz.
- `<IFO>-MOCK-BACKGROUND.ffl`, `<IFO>-MOCK-BACKGROUND/`: the same noise
  realisation, no injections. Foreground minus background is the injection.
- `injections.parquet`: one row per injection. `gps` geocentric merger;
  `gps_<IFO>` merger in each detector; `gps_start_<IFO>`, `gps_end_<IFO>` the
  sample support written (zero padded by the generator); `snr_<IFO>` optimal
  SNR injected; `network_snr`; `mass1`, `mass2`, `chirp_mass`, `mass_ratio`
  (detector frame), spins, sky, polarisation, inclination; `track_f_low`,
  `track_f_high` the band the track spans.
- `tracks.parquet`: the time-frequency track, {track_points} points per injection:
  `injection_id`, `time` (s from merger), `frequency` (Hz). In a detector the
  track is at `gps_<IFO> + time`.
- `validation.json`: the checks below, measured on the frames as written.

## Configuration

{duration_h:g} h from GPS {start_gps:.0f}, detectors {detectors}, unit-variance white
noise, {n_cbc} compact binaries (IMRPhenomD from {f_lower:g} Hz), network SNR
{snr_low:g}-{snr_high:g} with a share {core_w:g} drawn in {core_low:g}-{core_high:g}, chirp mass
{mc_low:g}-{mc_high:g} Msun uniform in its logarithm, mass ratio {q_low:g}-{q_high:g}. V1 receives
{v1:g} of the amplitude a LIGO detector with the same antenna response would.
Injections are {gap:g} s apart at least, support to support; the first and last
{edge_pad:g} s are free of them.

## Reading it with the worker

`file` = the FFL, `channel` = `<IFO>:{channel_suffix}`, `sampling` = {sample_rate},
`ResamplingFactor` = 1. Any `window`/`overlap` (e.g. 512/32, 1024/768) and any
basis or thresholding rule: the data do not depend on them. The worker still
band-passes to [LowFrequencyCut, 0.9 Nyquist] and fits its AR whitening; on
white noise that fit is the identity within estimation error. A median
{loss_median:.1%} (max {loss_max:.1%}) of an injection's SNR lies outside
{band_low:g}-{band_high:g} Hz, the band the worker passes at this rate; `snr_<IFO>`
counts the full band.

## Validation (on the frames as written)

{validation}
"""


def _validation_text(summary):
    lines = []
    for ifo, kinds in summary["spectrum"].items():
        bg = kinds["BACKGROUND"]
        worst = max(abs(b["mean_ratio"] - 1.0) / b["mean_sigma"] for b in bg)
        scatter = max(abs(b["bin_scatter"] / b["bin_sigma"] - 1.0) for b in bg)
        gauss = summary["gaussianity"][ifo]["BACKGROUND"]
        kurt = max(abs(b["excess_kurtosis"]) / b["kurtosis_sigma"] for b in gauss)
        tail = max(abs(b["tail_3sigma"] - b["tail_3sigma_expected"])
                   / b["tail_3sigma_sigma"] for b in gauss)
        mf = summary["matched_filter"][ifo]
        lines.append(
            f"- {ifo}: background spectrum at 1/16 Hz, octave bands 8-1000 Hz: "
            f"band means within {worst:.1f} sigma of flat, bin scatter within "
            f"{scatter:.1%} of chi-square; excess kurtosis within {kurt:.1f} "
            f"sigma, 3-sigma tail within {tail:.1f} sigma. Matched filter with "
            f"the true template, {mf['n']} injections: injection alone returns "
            f"the recorded SNR to {mf['injected_over_recorded_max_deviation']:.1e}"
            f" at lag {mf['injected_lag_samples_max']}; in noise, recovered - "
            f"recorded = {mf['recovered_minus_recorded_mean']:+.2f} +/- "
            f"{mf['recovered_minus_recorded_std']:.2f} (expected 0 +/- 1).")
    for pair, (dt, bound) in summary["light_travel"].items():
        lines.append(f"- {pair}: largest |dt| {dt * 1e3:.2f} ms, light travel "
                     f"{bound * 1e3:.2f} ms.")
    pop = summary["population"]
    lines.append(
        f"- Population: {pop['n']} injections, network SNR "
        f"{pop['network_snr_range'][0]:.1f}-{pop['network_snr_range'][1]:.1f}, "
        f"{pop['share_network_snr_7_12']:.0%} in 7-12; chirp mass "
        f"{pop['chirp_mass_range'][0]:.1f}-{pop['chirp_mass_range'][1]:.1f}.")
    return "\n".join(lines)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_benchmark(outdir, config=None, validate=True):
    """Write the benchmark set, its configuration and its validation.

    :type outdir: str
    :param outdir: directory to write to; created if absent.
    :type config: dict | None
    :param config: keyword arguments for :func:`wdf.mock.generate_dataset`;
        :data:`BENCHMARK_CONFIG` when None. A different one writes a
        different set, recorded as such.
    :type validate: bool
    :param validate: run :func:`wdf.mock.validation.validate_white_set` and
        :func:`worker_read_check` on what was written, and record them.
    :return: pandas.DataFrame -- the truth table.
    """
    from wdf.mock.dataset import generate_dataset
    from wdf.mock.validation import validate_white_set

    config = dict(BENCHMARK_CONFIG if config is None else config)
    if config.get("noise") != "white":
        raise ValueError("the benchmark is a white-noise set; noise must be 'white'")
    outdir = os.path.abspath(os.fspath(outdir))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "benchmark_config.json"), "w",
              encoding="utf-8") as handle:
        json.dump(_jsonable(config), handle, indent=2)
    with open(os.path.join(outdir, "provenance.json"), "w",
              encoding="utf-8") as handle:
        json.dump(_provenance(), handle, indent=2)

    truth = generate_dataset(outdir, **config)
    if not validate:
        return truth

    detectors = tuple(config["detectors"])
    sensitivity = config.get("relative_sensitivity")
    report = validate_white_set(outdir, detectors,
                                config.get("channel_suffix", "MOCK-STRAIN"),
                                sensitivity)
    report["matched_filter"].to_parquet(
        os.path.join(outdir, "validation_matched_filter.parquet"), index=False)
    band = (WORKER_BAND[0], 0.9 * 0.5 * float(config["sample_rate"]))
    losses = _band_loss(truth, detectors, sensitivity,
                        float(config["sample_rate"]), band)
    summary = _summarise(report, truth, losses)
    summary["worker_band_loss"]["band"] = band
    summary["worker_read_max_difference"] = worker_read_check(
        outdir, detectors, config.get("channel_suffix", "MOCK-STRAIN"))
    with open(os.path.join(outdir, "validation.json"), "w",
              encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2)

    population = config["cbc_population"]
    with open(os.path.join(outdir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(README.format(
            outdir=outdir,
            frame_length=config["frame_length"],
            channel_suffix=config["channel_suffix"],
            sample_rate=config["sample_rate"],
            track_points=config["track_points"],
            duration_h=config["duration"] / 3600.0,
            start_gps=config["start_gps"],
            detectors=", ".join(detectors),
            n_cbc=config["n_cbc"],
            f_lower=population["f_lower"],
            snr_low=config["snr_range"][0], snr_high=config["snr_range"][1],
            core_low=config["snr_core"][0], core_high=config["snr_core"][1],
            core_w=config["snr_core"][2],
            mc_low=population["chirp_mass"][0],
            mc_high=population["chirp_mass"][1],
            q_low=population["mass_ratio"][0],
            q_high=population["mass_ratio"][1],
            v1=sensitivity["V1"], gap=config["minimum_injection_gap"],
            edge_pad=config["edge_pad"],
            loss_median=summary["worker_band_loss"]["median"],
            loss_max=summary["worker_band_loss"]["max"],
            band_low=band[0], band_high=band[1],
            validation=_validation_text(summary)))
    return truth


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", help="directory to write the set to")
    parser.add_argument("--no-validate", action="store_true",
                        help="write the set without validating it")
    arguments = parser.parse_args()
    write_benchmark(arguments.outdir, validate=not arguments.no_validate)
