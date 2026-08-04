"""Mock two-detector data sets with known injections, for validating a search.

Produces, per detector, a foreground frame (noise plus injections) and a
background frame (the same noise realisation without injections), plus a table
of every injection with its time, class, parameters and signal-to-noise ratio.
Comparing triggers found in the two frames attributes each one to a signal or
to the noise without needing time slides.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from wdf.mock import waveforms
from wdf.mock.noise import DEFAULT_PSD, analytic_psd, coloured_noise

GROUND_TRUTH_COLUMNS = [
    "injection_id", "category", "subclass", "gps", "gps_H1", "gps_L1",
    "snr_H1", "snr_L1", "network_snr", "duration",
    "mass1", "mass2", "spin1z", "spin2z", "inclination", "ra", "dec", "polarization",
    "f0", "q", "f_start", "f_end", "arch_period", "n_arches", "sigma_t",
]


def optimal_snr(strain, sample_rate=2048, low_frequency_cutoff=5.0,
                psd_name=DEFAULT_PSD):
    """Optimal matched-filter signal-to-noise ratio of a signal against a PSD.

    :type strain: numpy.ndarray
    :param strain: signal samples.
    :type sample_rate: int
    :param sample_rate: sampling rate, Hz.
    :type low_frequency_cutoff: float
    :param low_frequency_cutoff: lower limit of the SNR integral, Hz.
    :type psd_name: str
    :param psd_name: name of any analytic PSD provided by `pycbc.psd`.
    :return: float -- the optimal SNR for unit amplitude scaling.
    """
    from pycbc.filter import sigma
    from pycbc.types import TimeSeries

    n = int(2 ** np.ceil(np.log2(max(len(strain), sample_rate))))
    padded = np.zeros(n)
    padded[:len(strain)] = strain
    series = TimeSeries(padded, delta_t=1.0 / sample_rate)
    psd = analytic_psd(n // 2 + 1, 1.0 / (n / sample_rate),
                       low_frequency_cutoff, psd_name)
    return float(sigma(series, psd=psd, low_frequency_cutoff=low_frequency_cutoff))


def project_cbc(hp, hc, ra, dec, polarization, gps, detectors=("H1", "L1")):
    """Project plus and cross polarisations onto detectors.

    Applies each detector's antenna pattern and its light-travel-time delay
    from the geocentre, so the resulting arrival times and amplitude ratios are
    physically consistent across the network.

    :type hp: pycbc.types.TimeSeries
    :param hp: plus polarisation.
    :type hc: pycbc.types.TimeSeries
    :param hc: cross polarisation.
    :type ra: float
    :param ra: right ascension, radians.
    :type dec: float
    :param dec: declination, radians.
    :type polarization: float
    :param polarization: polarisation angle, radians.
    :type gps: float
    :param gps: geocentric GPS time of the merger.
    :type detectors: tuple
    :param detectors: detector names.
    :return: dict -- {detector: (strain, arrival_gps)}, arrival relative to the
        first sample of `strain`.
    """
    from pycbc.detector import Detector

    out = {}
    for name in detectors:
        det = Detector(name)
        fp, fc = det.antenna_pattern(ra, dec, polarization, gps)
        strain = fp * np.asarray(hp) + fc * np.asarray(hc)
        delay = det.time_delay_from_earth_center(ra, dec, gps)
        out[name] = (strain, gps + delay)
    return out


def draw_injections(n_cbc=250, n_glitch=750, duration=28800.0, start_gps=0.0,
                    edge_pad=500.0, snr_range=(4.0, 50.0), seed=0,
                    sample_rate=2048, detectors=("H1", "L1")):
    """Draw injection parameters and non-overlapping times.

    Compact-binary signals are coincident across all detectors; glitches are
    placed in a single, randomly chosen detector.

    :type n_cbc: int
    :param n_cbc: number of compact-binary injections.
    :type n_glitch: int
    :param n_glitch: number of glitch injections.
    :type duration: float
    :param duration: span to fill, seconds.
    :type start_gps: float
    :param start_gps: GPS time of the first sample of the data.
    :type edge_pad: float
    :param edge_pad: span left free of injections at each end, seconds. A search
        estimating its noise model from the start of the data needs that stretch
        to be signal-free.
    :type snr_range: tuple
    :param snr_range: (low, high) bounds of the target SNR distribution.
    :type seed: int
    :param seed: seed fixing the draw.
    :type sample_rate: int
    :param sample_rate: sampling rate, Hz.
    :type detectors: tuple
    :param detectors: detector names.
    :return: list -- one dict of parameters per injection, ordered in time.
    """
    rng = np.random.default_rng(seed)
    specs = [_draw_cbc(rng, snr_range) for _ in range(n_cbc)]
    specs += [_draw_glitch(rng, snr_range, detectors) for _ in range(n_glitch)]
    rng.shuffle(specs)

    usable = duration - 2 * edge_pad
    total_signal = sum(s["duration"] for s in specs)
    mean_gap = max((usable - total_signal) / max(len(specs), 1), 1.0)

    cursor = start_gps + edge_pad
    placed = []
    for i, spec in enumerate(specs):
        gap = rng.uniform(0.5 * mean_gap, 1.5 * mean_gap)
        cursor += gap
        if cursor + spec["duration"] > start_gps + duration - edge_pad:
            break
        spec["injection_id"] = i
        spec["gps"] = cursor + 0.5 * spec["duration"]
        placed.append(spec)
        cursor += spec["duration"]
    return placed


def generate_dataset(outdir, duration=28800.0, start_gps=1400000000.0,
                     sample_rate=2048, n_cbc=250, n_glitch=750,
                     snr_range=(4.0, 50.0), seed=0, detectors=("H1", "L1"),
                     edge_pad=500.0, low_frequency_cutoff=5.0, psd_name=DEFAULT_PSD,
                     channel_suffix="MOCK-STRAIN", write_background=True,
                     frame_length=1024.0):
    """Generate and write a mock data set.

    Writes, per detector, the foreground frames `<ifo>-MOCK-FOREGROUND/` with
    their `<ifo>-MOCK-FOREGROUND.ffl` index and (optionally) the injection-free
    `<ifo>-MOCK-BACKGROUND/` with `<ifo>-MOCK-BACKGROUND.ffl`, plus
    `injections.parquet` describing every
    injection.

    :type outdir: str
    :param outdir: directory to write into; created if missing.
    :type duration: float
    :param duration: length of the data, seconds.
    :type start_gps: float
    :param start_gps: GPS time of the first sample.
    :type sample_rate: int
    :param sample_rate: sampling rate, Hz.
    :type n_cbc: int
    :param n_cbc: number of compact-binary injections.
    :type n_glitch: int
    :param n_glitch: number of glitch injections.
    :type snr_range: tuple
    :param snr_range: (low, high) bounds of the target SNR distribution.
    :type seed: int
    :param seed: seed fixing noise and injections.
    :type detectors: tuple
    :param detectors: detector names.
    :type edge_pad: float
    :param edge_pad: span left free of injections at each end, seconds.
    :type low_frequency_cutoff: float
    :param low_frequency_cutoff: lower limit of noise generation, band limiting and the
        SNR integral, Hz. Generating below the band the search analyses leaves the
        edge of the generated spectrum outside it, where the search band-pass
        removes it.
    :type psd_name: str
    :param psd_name: name of any analytic PSD provided by `pycbc.psd`.
    :type channel_suffix: str
    :param channel_suffix: channel name after the `<ifo>:` prefix.
    :type write_background: bool
    :param write_background: also write the injection-free frames.
    :type frame_length: float
    :param frame_length: seconds of data per GWF frame file.
    :return: pandas.DataFrame -- the ground-truth table, also written to disk.
    """
    from gwpy.timeseries import TimeSeries as GwpyTimeSeries

    os.makedirs(outdir, exist_ok=True)
    injections = draw_injections(n_cbc=n_cbc, n_glitch=n_glitch, duration=duration,
                                 start_gps=start_gps, edge_pad=edge_pad,
                                 snr_range=snr_range, seed=seed,
                                 sample_rate=sample_rate, detectors=detectors)

    noise = {}
    for i, ifo in enumerate(detectors):
        series = coloured_noise(start_gps, start_gps + duration, seed=seed + 1000 * (i + 1),
                                sample_rate=sample_rate,
                                low_frequency_cutoff=low_frequency_cutoff,
                                psd_name=psd_name)
        noise[ifo] = np.asarray(series)[:int(duration * sample_rate)]

    if write_background:
        for ifo in detectors:
            _write_frames(GwpyTimeSeries, noise[ifo], start_gps, sample_rate, ifo,
                          channel_suffix, outdir, "MOCK-BACKGROUND", frame_length)

    strain = {ifo: noise[ifo].copy() for ifo in detectors}
    rows = []
    for spec in injections:
        rows.append(_inject_one(spec, strain, start_gps, sample_rate, detectors,
                                low_frequency_cutoff, psd_name))

    for ifo in detectors:
        _write_frames(GwpyTimeSeries, strain[ifo], start_gps, sample_rate, ifo,
                      channel_suffix, outdir, "MOCK-FOREGROUND", frame_length)

    table = pd.DataFrame(rows).reindex(columns=GROUND_TRUTH_COLUMNS)
    table.to_parquet(os.path.join(outdir, "injections.parquet"), index=False)
    return table


def _draw_cbc(rng, snr_range):
    """Parameters for one compact-binary injection."""
    subclass = rng.choice(waveforms.CBC_CLASSES, p=[0.5, 0.28, 0.22])
    if subclass == "bbh":
        mass1, mass2, f_lower = rng.uniform(15, 50), rng.uniform(10, 40), 20.0
    elif subclass == "bhns":
        mass1, mass2, f_lower = rng.uniform(6, 20), rng.uniform(1.2, 2.0), 25.0
    else:
        mass1, mass2, f_lower = rng.uniform(1.2, 2.0), rng.uniform(1.2, 2.0), 25.0
    mass1, mass2 = max(mass1, mass2), min(mass1, mass2)
    approximant = "IMRPhenomD" if subclass == "bbh" else "TaylorF2"
    return dict(
        category="cbc", subclass=str(subclass), approximant=approximant,
        mass1=float(mass1), mass2=float(mass2),
        spin1z=float(rng.uniform(-0.5, 0.5)) if subclass == "bbh" else 0.0,
        spin2z=float(rng.uniform(-0.5, 0.5)) if subclass == "bbh" else 0.0,
        inclination=float(np.arccos(rng.uniform(-1, 1))),
        ra=float(rng.uniform(0, 2 * np.pi)),
        dec=float(np.arcsin(rng.uniform(-1, 1))),
        polarization=float(rng.uniform(0, 2 * np.pi)),
        f_lower=f_lower, target_snr=float(rng.uniform(*snr_range)),
        duration=float(_cbc_duration(mass1, mass2, f_lower)),
    )


def _draw_glitch(rng, snr_range, detectors):
    """Parameters for one single-detector glitch injection."""
    subclass = str(rng.choice(waveforms.GLITCH_CLASSES))
    params = dict(category="glitch", subclass=subclass,
                  detector=str(rng.choice(list(detectors))),
                  target_snr=float(rng.uniform(*snr_range)))
    if subclass == "gaussian":
        params.update(sigma_t=float(rng.uniform(0.002, 0.02)))
        params["duration"] = 12 * params["sigma_t"]
    elif subclass == "sine_gaussian":
        params.update(f0=float(rng.uniform(60, 600)), q=float(rng.uniform(5, 30)))
        params["duration"] = 8 * params["q"] / (2 * np.pi * params["f0"])
    elif subclass == "blip":
        params.update(f0=float(rng.uniform(80, 500)), q=float(rng.uniform(2, 5)))
        params["duration"] = 36 * params["q"] / (2 * np.pi * params["f0"])
    elif subclass == "chirplike":
        params.update(f_start=float(rng.uniform(25, 80)), f_end=float(rng.uniform(150, 700)),
                      chirp_duration=float(rng.uniform(0.2, 1.5)))
        params["duration"] = params["chirp_duration"]
    else:
        params.update(f_peak=float(rng.uniform(20, 60)),
                      arch_period=float(rng.uniform(0.5, 2.0)),
                      n_arches=int(rng.integers(2, 6)))
        params["duration"] = params["arch_period"] * params["n_arches"]
    return params


def _cbc_duration(mass1, mass2, f_lower):
    """Newtonian chirp time from `f_lower` to coalescence, seconds."""
    from pycbc.waveform import get_waveform_filter_length_in_time
    try:
        return max(get_waveform_filter_length_in_time(
            "TaylorF2", mass1=mass1, mass2=mass2, f_lower=f_lower), 1.0) + 2.0
    except Exception:
        chirp = (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2
        return 2.18 * (1.21 / chirp) ** (5.0 / 3.0) * (100.0 / f_lower) ** (8.0 / 3.0) + 2.0


def band_limit(strain, sample_rate=2048, low_frequency_cutoff=5.0, order=8):
    """High-pass a signal to the band the noise occupies, with zero phase.

    Glitch generators produce shapes with power down to zero frequency, where a
    coloured-noise realisation defined above `low_frequency_cutoff` has none.
    Injecting them unfiltered puts signal where there is no noise. Filtering is
    zero-phase so the injection stays at the time the ground truth records.

    :type strain: numpy.ndarray
    :param strain: signal samples.
    :type sample_rate: int
    :param sample_rate: sampling rate, Hz.
    :type low_frequency_cutoff: float
    :param low_frequency_cutoff: high-pass corner, Hz.
    :type order: int
    :param order: Butterworth filter order.
    :return: numpy.ndarray -- the high-passed signal.
    """
    from scipy.signal import butter, sosfiltfilt

    sos = butter(order, low_frequency_cutoff, btype="highpass", fs=sample_rate, output="sos")
    padlen = min(3 * (2 * order + 1), len(strain) - 1)
    return sosfiltfilt(sos, strain, padlen=max(padlen, 0))


def _glitch_strain(spec, sample_rate):
    """Unit-amplitude samples of one glitch."""
    if spec["subclass"] == "gaussian":
        return waveforms.gaussian(spec["sigma_t"], sample_rate)
    if spec["subclass"] == "sine_gaussian":
        return waveforms.sine_gaussian(spec["f0"], spec["q"], sample_rate)
    if spec["subclass"] == "blip":
        return waveforms.blip(spec["f0"], spec["q"], sample_rate=sample_rate)
    if spec["subclass"] == "chirplike":
        return waveforms.chirplike(spec["f_start"], spec["f_end"],
                                   spec["chirp_duration"], sample_rate)
    return waveforms.scattered_light(spec["f_peak"], spec["arch_period"],
                                     spec["n_arches"], sample_rate)


def _inject_one(spec, strain, start_gps, sample_rate, detectors,
                low_frequency_cutoff, psd_name):
    """Add one injection to `strain` in place and return its ground-truth row."""
    row = {k: spec.get(k, np.nan) for k in GROUND_TRUTH_COLUMNS}
    row.update(injection_id=spec["injection_id"], category=spec["category"],
               subclass=spec["subclass"], gps=spec["gps"], duration=spec["duration"])

    if spec["category"] == "glitch":
        samples = band_limit(_glitch_strain(spec, sample_rate), sample_rate,
                             low_frequency_cutoff)
        snr0 = optimal_snr(samples, sample_rate, low_frequency_cutoff, psd_name)
        scale = spec["target_snr"] / snr0 if snr0 > 0 else 0.0
        ifo = spec["detector"]
        _add(strain[ifo], samples * scale, spec["gps"] - 0.5 * len(samples) / sample_rate,
             start_gps, sample_rate)
        row[f"snr_{ifo}"] = spec["target_snr"]
        row["network_snr"] = spec["target_snr"]
        return row

    hp, hc = waveforms.cbc_polarisations(
        spec["mass1"], spec["mass2"], spec["spin1z"], spec["spin2z"],
        inclination=spec["inclination"], f_lower=spec["f_lower"],
        sample_rate=sample_rate, approximant=spec["approximant"])
    projected = project_cbc(hp, hc, spec["ra"], spec["dec"], spec["polarization"],
                            spec["gps"], detectors)
    snrs = {ifo: optimal_snr(s, sample_rate, low_frequency_cutoff, psd_name)
            for ifo, (s, _) in projected.items()}
    network0 = np.sqrt(sum(v ** 2 for v in snrs.values()))
    scale = spec["target_snr"] / network0 if network0 > 0 else 0.0

    merger_offset = -float(hp.start_time)
    for ifo, (samples, arrival) in projected.items():
        _add(strain[ifo], samples * scale, arrival - merger_offset, start_gps, sample_rate)
        row[f"snr_{ifo}"] = snrs[ifo] * scale
        row[f"gps_{ifo}"] = arrival
    row["network_snr"] = spec["target_snr"]
    return row


def _add(target, samples, t_start, start_gps, sample_rate):
    """Add `samples` into `target` at GPS time `t_start`, clipped to bounds."""
    i0 = int(round((t_start - start_gps) * sample_rate))
    lo, hi = max(i0, 0), min(i0 + len(samples), len(target))
    if hi > lo:
        target[lo:hi] += samples[lo - i0:hi - i0]


def _write_frames(gwpy_timeseries, data, start_gps, sample_rate, ifo,
                  channel_suffix, outdir, tag, frame_length=1024.0):
    """Write one detector's samples as a series of GWF frame files and the FFL
    index that lists them.

    Frames are named `<ifo>-<tag>-<gps>-<length>.gwf` and written into
    `<outdir>/<ifo>-<tag>/`; the FFL is `<outdir>/<ifo>-<tag>.ffl`, one line per
    frame holding path, GPS start, length and two zero fields. A trailing span
    shorter than `frame_length` is written as a final, shorter frame.

    :type gwpy_timeseries: type
    :param gwpy_timeseries: the `gwpy.timeseries.TimeSeries` class.
    :type data: numpy.ndarray
    :param data: the detector's samples.
    :type start_gps: float
    :param start_gps: GPS time of the first sample.
    :type sample_rate: int
    :param sample_rate: samples per second.
    :type ifo: str
    :param ifo: detector prefix, e.g. `H1`.
    :type channel_suffix: str
    :param channel_suffix: channel name after the `<ifo>:` prefix.
    :type outdir: str
    :param outdir: directory to write the frame directory and FFL into.
    :type tag: str
    :param tag: frame type, e.g. `MOCK-FOREGROUND`.
    :type frame_length: float
    :param frame_length: seconds of data per frame file.
    :return: str -- path of the FFL file written.
    """
    framedir = os.path.join(outdir, f"{ifo}-{tag}")
    os.makedirs(framedir, exist_ok=True)
    per_frame = int(frame_length * sample_rate)
    lines = []
    for first in range(0, len(data), per_frame):
        chunk = data[first:first + per_frame]
        gps = start_gps + first / sample_rate
        length = len(chunk) / sample_rate
        path = os.path.join(framedir, f"{ifo}-{tag}-{int(gps)}-{int(length)}.gwf")
        series = gwpy_timeseries(chunk, t0=gps, sample_rate=sample_rate,
                                 channel=f"{ifo}:{channel_suffix}",
                                 name=f"{ifo}:{channel_suffix}")
        series.write(path, format="gwf")
        lines.append(f"{os.path.abspath(path)} {gps:.0f} {length:.0f} 0 0")

    ffl = os.path.join(outdir, f"{ifo}-{tag}.ffl")
    with open(ffl, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return ffl
