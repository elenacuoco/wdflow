"""Mock multi-detector data sets with known injections for validating WDF.

For each detector this module writes:

* foreground frames: one coloured-noise realisation plus injections;
* background frames: the same noise realisation without injections;
* an FFL index for each frame sequence;
* ``injections.parquet`` containing the ground truth.

Compact-binary ``gps`` values are geocentric merger times. Glitch ``gps``
values are the centre of the generated sample array. The truth table also
contains the actual time support in every detector, which is the preferred
quantity for matching long CBC signals to WDF triggers.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

from wdf.mock import waveforms
from wdf.mock.noise import DEFAULT_PSD, analytic_psd, coloured_noise


GROUND_TRUTH_COLUMNS = [
    "injection_id",
    "category",
    "subclass",
    "detector",
    "approximant",
    "gps",
    "gps_start",
    "gps_end",
    "gps_H1",
    "gps_L1",
    "gps_start_H1",
    "gps_end_H1",
    "gps_start_L1",
    "gps_end_L1",
    "target_snr",
    "snr_H1",
    "snr_L1",
    "network_snr",
    "duration",
    "support_before",
    "support_after",
    "mass1",
    "mass2",
    "spin1z",
    "spin2z",
    "inclination",
    "ra",
    "dec",
    "polarization",
    "f_lower",
    "f0",
    "q",
    "f_start",
    "f_end",
    "f_peak",
    "arch_period",
    "n_arches",
    "sigma_t",
]


def optimal_snr(
    strain,
    sample_rate=2048,
    low_frequency_cutoff=5.0,
    high_frequency_cutoff=None,
    psd_name=DEFAULT_PSD,
):
    """Return the optimal matched-filter SNR of a deterministic signal.

    The SNR is evaluated against the same analytic PSD used to generate the
    mock noise and only in the requested analysis band.
    """
    from pycbc.filter import sigma
    from pycbc.types import TimeSeries

    x = np.asarray(strain, dtype=float).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all() or not np.any(x):
        return 0.0

    sample_rate = float(sample_rate)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")

    nyquist = 0.5 * sample_rate
    f_low = float(low_frequency_cutoff)
    f_high = nyquist if high_frequency_cutoff is None else min(
        float(high_frequency_cutoff), nyquist
    )

    if not 0.0 <= f_low < f_high:
        raise ValueError(f"Invalid SNR band [{f_low}, {f_high}] Hz")

    minimum_length = max(x.size, int(np.ceil(sample_rate)))
    nfft = 1 << int(np.ceil(np.log2(minimum_length)))

    padded = np.zeros(nfft, dtype=float)
    padded[: x.size] = x

    delta_t = 1.0 / sample_rate
    delta_f = sample_rate / nfft
    series = TimeSeries(padded, delta_t=delta_t)
    psd = analytic_psd(
        nfft // 2 + 1,
        delta_f,
        f_low,
        psd_name,
    )

    value = sigma(
        series,
        psd=psd,
        low_frequency_cutoff=f_low,
        high_frequency_cutoff=f_high,
    )
    value = float(value)
    return value if np.isfinite(value) else 0.0


def project_cbc(hp, hc, ra, dec, polarization, gps, detectors=("H1", "L1")):
    """Project plus and cross polarisations onto the detector network.

    Returns ``{ifo: (strain, arrival_gps)}``, where ``arrival_gps`` is the
    detector merger time and ``strain`` retains the sample support of ``hp``.
    """
    from pycbc.detector import Detector

    hp_array = np.asarray(hp, dtype=float)
    hc_array = np.asarray(hc, dtype=float)
    if hp_array.shape != hc_array.shape:
        raise ValueError("hp and hc must have the same shape")

    out = {}
    for name in detectors:
        det = Detector(name)
        fp, fc = det.antenna_pattern(ra, dec, polarization, gps)
        strain = fp * hp_array + fc * hc_array
        delay = det.time_delay_from_earth_center(ra, dec, gps)
        out[name] = (strain, float(gps + delay))
    return out


def _draw_cbc(rng, snr_range):
    """Draw physical parameters for one compact-binary injection."""
    subclass = str(rng.choice(waveforms.CBC_CLASSES, p=[0.50, 0.28, 0.22]))

    if subclass == "bbh":
        mass1 = rng.uniform(15.0, 50.0)
        mass2 = rng.uniform(10.0, 40.0)
        f_lower = 20.0
        approximant = "IMRPhenomD"
    elif subclass == "bhns":
        mass1 = rng.uniform(6.0, 20.0)
        mass2 = rng.uniform(1.2, 2.0)
        f_lower = 25.0
        approximant = "TaylorF2"
    else:
        mass1 = rng.uniform(1.2, 2.0)
        mass2 = rng.uniform(1.2, 2.0)
        f_lower = 25.0
        approximant = "TaylorF2"

    mass1, mass2 = max(mass1, mass2), min(mass1, mass2)

    return {
        "category": "cbc",
        "subclass": subclass,
        "approximant": approximant,
        "mass1": float(mass1),
        "mass2": float(mass2),
        "spin1z": float(rng.uniform(-0.5, 0.5)) if subclass == "bbh" else 0.0,
        "spin2z": float(rng.uniform(-0.5, 0.5)) if subclass == "bbh" else 0.0,
        "inclination": float(np.arccos(rng.uniform(-1.0, 1.0))),
        "ra": float(rng.uniform(0.0, 2.0 * np.pi)),
        "dec": float(np.arcsin(rng.uniform(-1.0, 1.0))),
        "polarization": float(rng.uniform(0.0, 2.0 * np.pi)),
        "f_lower": float(f_lower),
        "target_snr": float(rng.uniform(*snr_range)),
    }


def _draw_glitch(rng, snr_range, detectors):
    """Draw parameters for one single-detector glitch injection."""
    subclass = str(rng.choice(waveforms.GLITCH_CLASSES))
    params = {
        "category": "glitch",
        "subclass": subclass,
        "detector": str(rng.choice(list(detectors))),
        "target_snr": float(rng.uniform(*snr_range)),
    }

    if subclass == "gaussian":
        params["sigma_t"] = float(rng.uniform(0.002, 0.020))
    elif subclass == "sine_gaussian":
        params["f0"] = float(rng.uniform(60.0, 600.0))
        params["q"] = float(rng.uniform(5.0, 30.0))
    elif subclass == "blip":
        params["f0"] = float(rng.uniform(80.0, 500.0))
        params["q"] = float(rng.uniform(2.0, 5.0))
    elif subclass == "chirplike":
        params["f_start"] = float(rng.uniform(25.0, 80.0))
        params["f_end"] = float(rng.uniform(150.0, 700.0))
        params["chirp_duration"] = float(rng.uniform(0.2, 1.5))
    elif subclass == "scattered_light":
        params["f_peak"] = float(rng.uniform(20.0, 60.0))
        params["arch_period"] = float(rng.uniform(0.5, 2.0))
        params["n_arches"] = int(rng.integers(2, 6))
    else:
        raise ValueError(f"Unsupported glitch subclass: {subclass}")

    return params


def _glitch_strain(spec, sample_rate):
    """Return unit-amplitude samples for one glitch specification."""
    if spec["subclass"] == "gaussian":
        return waveforms.gaussian(spec["sigma_t"], sample_rate)
    if spec["subclass"] == "sine_gaussian":
        return waveforms.sine_gaussian(spec["f0"], spec["q"], sample_rate)
    if spec["subclass"] == "blip":
        return waveforms.blip(spec["f0"], spec["q"], sample_rate=sample_rate)
    if spec["subclass"] == "chirplike":
        return waveforms.chirplike(
            spec["f_start"],
            spec["f_end"],
            spec["chirp_duration"],
            sample_rate,
        )
    if spec["subclass"] == "scattered_light":
        return waveforms.scattered_light(
            spec["f_peak"],
            spec["arch_period"],
            spec["n_arches"],
            sample_rate,
        )
    raise ValueError(f"Unsupported glitch subclass: {spec['subclass']}")


def _prepare_injection_support(spec, sample_rate):
    """Attach the actual generated time support to an injection specification."""
    prepared = dict(spec)

    if prepared["category"] == "cbc":
        hp, hc = waveforms.cbc_polarisations(
            prepared["mass1"],
            prepared["mass2"],
            prepared["spin1z"],
            prepared["spin2z"],
            inclination=prepared["inclination"],
            f_lower=prepared["f_lower"],
            sample_rate=sample_rate,
            approximant=prepared["approximant"],
        )
        if len(hp) != len(hc):
            raise RuntimeError("Generated hp and hc have different lengths")

        start_offset = float(hp.start_time)
        end_offset = start_offset + len(hp) / float(sample_rate)
        prepared["support_before"] = max(-start_offset, 0.0)
        prepared["support_after"] = max(end_offset, 0.0)
        prepared["duration"] = len(hp) / float(sample_rate)
    else:
        samples = _glitch_strain(prepared, sample_rate)
        duration = len(samples) / float(sample_rate)
        prepared["support_before"] = 0.5 * duration
        prepared["support_after"] = duration - prepared["support_before"]
        prepared["duration"] = duration

    return prepared


def draw_injections(
    n_cbc=250,
    n_glitch=750,
    duration=28800.0,
    start_gps=0.0,
    edge_pad=500.0,
    snr_range=(4.0, 50.0),
    seed=0,
    sample_rate=2048,
    detectors=("H1", "L1"),
    minimum_gap=1.0,
    detector_delay_pad=0.02,
    strict=True,
):
    """Draw and place non-overlapping CBC and glitch injections.

    CBC ``gps`` is the geocentric merger time. Glitch ``gps`` is the centre
    of its generated sample array. Placement uses the actual generated
    waveform support and reserves ``detector_delay_pad`` on both sides.
    """
    if n_cbc < 0 or n_glitch < 0:
        raise ValueError("Injection counts must be non-negative")
    if duration <= 2.0 * edge_pad:
        raise ValueError("duration must be larger than 2 * edge_pad")
    if minimum_gap < 0.0:
        raise ValueError("minimum_gap must be non-negative")

    rng = np.random.default_rng(seed)
    specs = [_draw_cbc(rng, snr_range) for _ in range(n_cbc)]
    specs.extend(_draw_glitch(rng, snr_range, detectors) for _ in range(n_glitch))
    specs = [_prepare_injection_support(spec, sample_rate) for spec in specs]
    rng.shuffle(specs)

    n_requested = len(specs)
    usable_start = float(start_gps) + float(edge_pad)
    usable_end = float(start_gps) + float(duration) - float(edge_pad)
    usable_span = usable_end - usable_start

    protected_lengths = np.asarray(
        [
            float(spec["support_before"])
            + float(spec["support_after"])
            + 2.0 * float(detector_delay_pad)
            for spec in specs
        ],
        dtype=float,
    )
    minimum_gap_total = float(minimum_gap) * max(n_requested - 1, 0)
    required = float(protected_lengths.sum()) + minimum_gap_total

    if required > usable_span:
        message = (
            f"Requested injections need {required:.3f} s but only "
            f"{usable_span:.3f} s are available after edge padding."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message)

    # Distribute all remaining space as non-negative random gaps, including
    # before the first and after the last injection. Normalising the weights
    # guarantees that random gap draws can never make placement overflow.
    leftover = max(usable_span - required, 0.0)
    weights = rng.exponential(scale=1.0, size=n_requested + 1)
    weights /= weights.sum() if weights.sum() > 0.0 else 1.0
    extra_gaps = leftover * weights

    cursor = usable_start + extra_gaps[0]
    placed = []

    for index, spec in enumerate(specs):
        support_before = float(spec["support_before"])
        support_after = float(spec["support_after"])

        protected_start = cursor
        gps = protected_start + float(detector_delay_pad) + support_before
        protected_end = gps + support_after + float(detector_delay_pad)

        if protected_end > usable_end + 1e-9:
            message = (
                f"Only {len(placed)} of {n_requested} requested injections fit."
            )
            if strict:
                raise RuntimeError(message)
            warnings.warn(message)
            break

        item = dict(spec)
        item["injection_id"] = len(placed)
        item["gps"] = float(gps)
        item["gps_start"] = float(gps - support_before)
        item["gps_end"] = float(gps + support_after)
        placed.append(item)

        if index < n_requested - 1:
            cursor = (
                protected_end
                + float(minimum_gap)
                + extra_gaps[index + 1]
            )

    if strict and len(placed) != n_requested:
        raise RuntimeError(
            f"Placed {len(placed)} of {n_requested} requested injections"
        )

    return placed


def band_limit(
    strain,
    sample_rate=2048,
    low_frequency_cutoff=5.0,
    high_frequency_cutoff=None,
    order=8,
):
    """Apply a zero-phase band limit to a generated glitch."""
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(strain, dtype=float).reshape(-1)
    if x.size < 3:
        return x.copy()

    sample_rate = float(sample_rate)
    nyquist = 0.5 * sample_rate
    f_low = max(float(low_frequency_cutoff), 0.0)
    f_high = nyquist if high_frequency_cutoff is None else min(
        float(high_frequency_cutoff), nyquist
    )

    limits_top = f_high < nyquist

    # scipy requires a strict inequality at Nyquist.
    if not limits_top:
        f_high = np.nextafter(nyquist, 0.0)

    if f_low <= 0.0 and not limits_top:
        return x.copy()
    if not 0.0 <= f_low < f_high:
        raise ValueError(f"Invalid filtering band [{f_low}, {f_high}] Hz")

    if f_low > 0.0 and limits_top:
        wn = (f_low, f_high)
        btype = "bandpass"
    elif f_low > 0.0:
        wn = f_low
        btype = "highpass"
    else:
        wn = f_high
        btype = "lowpass"

    sos = butter(order, wn, btype=btype, fs=sample_rate, output="sos")
    default_pad = 3 * (2 * len(sos) + 1)
    padlen = min(default_pad, x.size - 1)
    return sosfiltfilt(sos, x, padlen=max(padlen, 0))


def _add(
    target,
    samples,
    t_start,
    start_gps,
    sample_rate,
    allow_clipping=False,
):
    """Add samples to a target array and return the inserted time interval."""
    samples = np.asarray(samples, dtype=float).reshape(-1)
    i0 = int(np.rint((float(t_start) - float(start_gps)) * float(sample_rate)))
    i1 = i0 + samples.size

    lo = max(i0, 0)
    hi = min(i1, len(target))
    n_inserted = max(hi - lo, 0)
    clipped = n_inserted != samples.size

    if clipped and not allow_clipping:
        raise RuntimeError(
            "Injection would be clipped: "
            f"requested [{i0}, {i1}), available [0, {len(target)})"
        )

    if n_inserted > 0:
        source_lo = lo - i0
        source_hi = source_lo + n_inserted
        target[lo:hi] += samples[source_lo:source_hi]

    return {
        "sample_start": lo,
        "sample_end": hi,
        "gps_start": float(start_gps) + lo / float(sample_rate),
        "gps_end": float(start_gps) + hi / float(sample_rate),
        "n_inserted": n_inserted,
        "n_requested": samples.size,
        "clipped": clipped,
    }


def _inject_one(
    spec,
    strain,
    start_gps,
    sample_rate,
    detectors,
    low_frequency_cutoff,
    high_frequency_cutoff,
    psd_name,
):
    """Inject one signal in place and return its complete truth-table row."""
    row = {column: np.nan for column in GROUND_TRUTH_COLUMNS}
    for key, value in spec.items():
        if key in row:
            row[key] = value

    row.update(
        injection_id=int(spec["injection_id"]),
        category=spec["category"],
        subclass=spec["subclass"],
        gps=float(spec["gps"]),
        gps_start=float(spec["gps_start"]),
        gps_end=float(spec["gps_end"]),
        target_snr=float(spec["target_snr"]),
        duration=float(spec["duration"]),
        support_before=float(spec["support_before"]),
        support_after=float(spec["support_after"]),
    )

    for ifo in detectors:
        snr_key = f"snr_{ifo}"
        if snr_key in row:
            row[snr_key] = 0.0

    if spec["category"] == "glitch":
        samples = band_limit(
            _glitch_strain(spec, sample_rate),
            sample_rate=sample_rate,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
        )
        snr0 = optimal_snr(
            samples,
            sample_rate=sample_rate,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            psd_name=psd_name,
        )
        if snr0 <= 0.0:
            raise RuntimeError(f"Zero SNR for glitch {spec['injection_id']}")

        scaled = samples * (float(spec["target_snr"]) / snr0)
        ifo = spec["detector"]
        t_start = float(spec["gps"]) - 0.5 * len(scaled) / float(sample_rate)
        inserted = _add(
            strain[ifo],
            scaled,
            t_start,
            start_gps,
            sample_rate,
            allow_clipping=False,
        )
        achieved_snr = optimal_snr(
            scaled,
            sample_rate=sample_rate,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            psd_name=psd_name,
        )

        row["detector"] = ifo
        row[f"snr_{ifo}"] = achieved_snr
        row["network_snr"] = achieved_snr
        row[f"gps_{ifo}"] = float(spec["gps"])
        row[f"gps_start_{ifo}"] = inserted["gps_start"]
        row[f"gps_end_{ifo}"] = inserted["gps_end"]
        return row

    hp, hc = waveforms.cbc_polarisations(
        spec["mass1"],
        spec["mass2"],
        spec["spin1z"],
        spec["spin2z"],
        inclination=spec["inclination"],
        f_lower=spec["f_lower"],
        sample_rate=sample_rate,
        approximant=spec["approximant"],
    )
    projected = project_cbc(
        hp,
        hc,
        spec["ra"],
        spec["dec"],
        spec["polarization"],
        spec["gps"],
        detectors,
    )

    unscaled_snrs = {
        ifo: optimal_snr(
            samples,
            sample_rate=sample_rate,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            psd_name=psd_name,
        )
        for ifo, (samples, _) in projected.items()
    }
    network0 = float(np.sqrt(sum(value * value for value in unscaled_snrs.values())))
    if network0 <= 0.0:
        raise RuntimeError(f"Zero network SNR for CBC {spec['injection_id']}")

    scale = float(spec["target_snr"]) / network0
    waveform_start_offset = float(hp.start_time)
    achieved_network_squared = 0.0

    for ifo, (samples, arrival) in projected.items():
        scaled = np.asarray(samples, dtype=float) * scale
        inserted = _add(
            strain[ifo],
            scaled,
            float(arrival) + waveform_start_offset,
            start_gps,
            sample_rate,
            allow_clipping=False,
        )
        achieved_snr = optimal_snr(
            scaled,
            sample_rate=sample_rate,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            psd_name=psd_name,
        )

        row[f"snr_{ifo}"] = achieved_snr
        row[f"gps_{ifo}"] = float(arrival)
        row[f"gps_start_{ifo}"] = inserted["gps_start"]
        row[f"gps_end_{ifo}"] = inserted["gps_end"]
        achieved_network_squared += achieved_snr * achieved_snr

    row["network_snr"] = float(np.sqrt(achieved_network_squared))
    row["duration"] = len(hp) / float(sample_rate)
    return row


def _time_token(value):
    """Return a filesystem-safe token without discarding fractional GPS time."""
    value = float(value)
    if np.isclose(value, round(value), rtol=0.0, atol=1e-9):
        return str(int(round(value)))
    return f"{value:.9f}".rstrip("0").rstrip(".").replace(".", "p")


def _write_frames(
    gwpy_timeseries,
    data,
    start_gps,
    sample_rate,
    ifo,
    channel_suffix,
    outdir,
    tag,
    frame_length=1024.0,
):
    """Write consecutive GWF frames and an FFL index for one detector."""
    data = np.asarray(data, dtype=float).reshape(-1)
    sample_rate = float(sample_rate)
    frame_length = float(frame_length)

    if sample_rate <= 0.0 or frame_length <= 0.0:
        raise ValueError("sample_rate and frame_length must be positive")

    per_frame_float = frame_length * sample_rate
    per_frame = int(round(per_frame_float))
    if per_frame <= 0 or not np.isclose(
        per_frame, per_frame_float, rtol=0.0, atol=1e-8
    ):
        raise ValueError("frame_length * sample_rate must be a positive integer")

    framedir = os.path.join(outdir, f"{ifo}-{tag}")
    os.makedirs(framedir, exist_ok=True)
    lines = []

    for first in range(0, data.size, per_frame):
        chunk = data[first : first + per_frame]
        gps = float(start_gps) + first / sample_rate
        length = chunk.size / sample_rate
        path = os.path.join(
            framedir,
            f"{ifo}-{tag}-{_time_token(gps)}-{_time_token(length)}.gwf",
        )
        series = gwpy_timeseries(
            chunk,
            t0=gps,
            sample_rate=sample_rate,
            channel=f"{ifo}:{channel_suffix}",
            name=f"{ifo}:{channel_suffix}",
        )
        series.write(path, format="gwf")
        lines.append(f"{os.path.abspath(path)} {gps:.9f} {length:.9f} 0 0")

    ffl = os.path.join(outdir, f"{ifo}-{tag}.ffl")
    with open(ffl, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    return ffl


def generate_dataset(
    outdir,
    duration=28800.0,
    start_gps=1400000000.0,
    sample_rate=2048,
    n_cbc=250,
    n_glitch=750,
    snr_range=(4.0, 50.0),
    seed=0,
    detectors=("H1", "L1"),
    edge_pad=500.0,
    low_frequency_cutoff=5.0,
    high_frequency_cutoff=None,
    analysis_sample_rate=None,
    psd_name=DEFAULT_PSD,
    channel_suffix="MOCK-STRAIN",
    write_background=True,
    frame_length=1024.0,
    minimum_injection_gap=1.0,
    strict=True,
):
    """Generate and write a complete mock foreground/background data set.

    ``sample_rate`` is the frame generation rate. ``analysis_sample_rate`` is
    the rate after WDF resampling and is used only to cap the SNR/filter band.
    """
    from gwpy.timeseries import TimeSeries as GwpyTimeSeries

    outdir = os.fspath(outdir)
    duration = float(duration)
    start_gps = float(start_gps)
    sample_rate = int(sample_rate)
    if sample_rate <= 0 or duration <= 0.0:
        raise ValueError("sample_rate and duration must be positive")

    if analysis_sample_rate is None:
        analysis_sample_rate = sample_rate
    analysis_sample_rate = float(analysis_sample_rate)
    if analysis_sample_rate <= 0.0:
        raise ValueError("analysis_sample_rate must be positive")

    native_nyquist = 0.5 * sample_rate
    analysis_nyquist = 0.5 * analysis_sample_rate
    if high_frequency_cutoff is None:
        f_high = min(native_nyquist, analysis_nyquist)
    else:
        f_high = min(float(high_frequency_cutoff), native_nyquist, analysis_nyquist)

    f_low = float(low_frequency_cutoff)
    if not 0.0 <= f_low < f_high:
        raise ValueError(f"Invalid analysis band [{f_low}, {f_high}] Hz")

    nsamples_float = duration * sample_rate
    nsamples = int(round(nsamples_float))
    if not np.isclose(nsamples, nsamples_float, rtol=0.0, atol=1e-8):
        raise ValueError("duration * sample_rate must be an integer")

    os.makedirs(outdir, exist_ok=True)

    injections = draw_injections(
        n_cbc=n_cbc,
        n_glitch=n_glitch,
        duration=duration,
        start_gps=start_gps,
        edge_pad=edge_pad,
        snr_range=snr_range,
        seed=seed,
        sample_rate=sample_rate,
        detectors=detectors,
        minimum_gap=minimum_injection_gap,
        strict=strict,
    )

    requested = int(n_cbc) + int(n_glitch)
    if strict and len(injections) != requested:
        raise RuntimeError(
            f"Generated {len(injections)} of {requested} requested injections"
        )

    noise = {}
    for index, ifo in enumerate(detectors):
        series = coloured_noise(
            start_gps,
            start_gps + duration,
            seed=seed + 1000 * (index + 1),
            sample_rate=sample_rate,
            low_frequency_cutoff=f_low,
            psd_name=psd_name,
        )
        values = np.asarray(series, dtype=float)
        if values.size < nsamples:
            raise RuntimeError(
                f"Noise for {ifo} has {values.size} samples; expected {nsamples}"
            )
        noise[ifo] = values[:nsamples].copy()

    if write_background:
        for ifo in detectors:
            _write_frames(
                GwpyTimeSeries,
                noise[ifo],
                start_gps,
                sample_rate,
                ifo,
                channel_suffix,
                outdir,
                "MOCK-BACKGROUND",
                frame_length,
            )

    foreground = {ifo: noise[ifo].copy() for ifo in detectors}
    rows = [
        _inject_one(
            spec,
            foreground,
            start_gps,
            sample_rate,
            detectors,
            f_low,
            f_high,
            psd_name,
        )
        for spec in injections
    ]

    for ifo in detectors:
        _write_frames(
            GwpyTimeSeries,
            foreground[ifo],
            start_gps,
            sample_rate,
            ifo,
            channel_suffix,
            outdir,
            "MOCK-FOREGROUND",
            frame_length,
        )

    table = pd.DataFrame(rows).reindex(columns=GROUND_TRUTH_COLUMNS)
    table.to_parquet(os.path.join(outdir, "injections.parquet"), index=False)
    return table
 