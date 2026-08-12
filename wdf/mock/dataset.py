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

A compact binary is placed on the sky and projected, not copied. Each is drawn
isotropically -- uniform in right ascension and in the sine of declination, with
a uniform polarisation and an inclination uniform in its cosine -- and
``project_cbc`` forms each detector's strain from that detector's own antenna
response, with the arrival time delayed by the geometry. The two detectors
therefore see different amplitudes, and arrival times differing by up to the
light travel time between them, for one source. Only the overall normalisation
is imposed: the network signal-to-noise ratio is scaled to the requested value,
and how it divides between the detectors follows from where the source is. A
coincidence test is then asked the question it will be asked on real data,
rather than being shown one detector's data copied into the other.

Glitches are single-detector by construction and are not projected: each exists
in the detector it was placed in. In coincidence they can therefore only measure
the accidental floor, which is what makes them useful there --- an efficiency at
that floor is not evidence of recovery.
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
    # Which of the simulation's axes the observer sits on, for a waveform read
    # from a catalogue. Empty for everything generated in closed form.
    "direction",
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
    psd=None,
):
    """Return the optimal matched-filter SNR of a deterministic signal.

    The SNR is evaluated in the requested analysis band against ``psd`` when one
    is supplied, and otherwise against the analytic model named by ``psd_name``.
    Data whose noise was not generated from that model -- a real detector
    segment, say -- must supply its own measured spectrum, or the recorded SNR
    describes a different detector than the one the signal was injected into.

    :type psd: pycbc.types.FrequencySeries or None
    :param psd: measured noise spectrum, resampled onto the required frequency
        grid if its resolution differs.
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
    if psd is None:
        psd = analytic_psd(nfft // 2 + 1, delta_f, f_low, psd_name)
    else:
        from pycbc.psd import interpolate
        psd = interpolate(psd, delta_f)
        psd = psd[: nfft // 2 + 1]

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

    Each detector sees `F+ hp + Fx hc`, with its own antenna response evaluated
    for the source's position and polarisation at that time, and its own arrival
    time delayed from the geocentre by the geometry. This is what makes the two
    detectors' data a network view of one source rather than two copies: the
    amplitude ratio and the arrival-time difference are then properties of where
    the source is, and are exactly what the coincidence stage has to survive.

    :type hp: array-like
    :param hp: plus polarisation, sampled uniformly in time.
    :type hc: array-like
    :param hc: cross polarisation, on the same samples as `hp`.
    :type ra: float
    :param ra: right ascension of the source, radians.
    :type dec: float
    :param dec: declination of the source, radians.
    :type polarization: float
    :param polarization: polarisation angle, radians.
    :type gps: float
    :param gps: geocentric time the response is evaluated at, seconds. The
        response depends on it through the Earth's orientation.
    :type detectors: iterable of str
    :param detectors: the detectors to project onto.
    :return: dict -- `{ifo: (strain, arrival_gps)}`, where `strain` keeps the
        sample support of `hp` and `arrival_gps` is that detector's merger time.
    :raises ValueError: if `hp` and `hc` do not have the same shape.
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


# How a drawn compact binary divides between the classes, in the order of
# `waveforms.CBC_CLASSES`. It is a property of the set being generated and not
# of the search: an efficiency averaged over classes says as much about this
# mixture as about the pipeline, so a set states the mixture it was drawn with.
DEFAULT_CBC_MIX = (0.50, 0.28, 0.22)


def _draw_cbc(rng, snr_range, cbc_mix=DEFAULT_CBC_MIX):
    """Draw physical parameters for one compact-binary injection.

    :param rng: the generator the draw comes from.
    :param snr_range: the network signal-to-noise ratio to scale to.
    :param cbc_mix: probability of each class, in the order of
        `waveforms.CBC_CLASSES`. Normalised internally, so weights need not
        sum to one.
    :return: dict -- the injection's physical parameters.
    :raises ValueError: if the mixture does not give one non-negative weight
        per class.
    """
    mix = np.asarray(cbc_mix, dtype=float)
    if mix.size != len(waveforms.CBC_CLASSES) or mix.min() < 0.0 or mix.sum() <= 0:
        raise ValueError(
            "cbc_mix must give one non-negative weight per class "
            f"{tuple(waveforms.CBC_CLASSES)}, got {tuple(cbc_mix)}")
    subclass = str(rng.choice(waveforms.CBC_CLASSES, p=mix / mix.sum()))

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


def _draw_ccsn(rng, snr_range, catalogue):
    """Draw one core-collapse supernova injection from a waveform catalogue.

    The model and the observer direction are drawn uniformly over what the
    catalogue holds, since neither is known in advance for a real source and the
    catalogue is not a population: it is a set of simulations, and weighting it
    by anything would be asserting a progenitor distribution the set does not
    carry. Directions are drawn per model, so a model the catalogue holds along
    two axes is never asked for a third.

    :param rng: the generator the draw comes from.
    :param snr_range: the network signal-to-noise ratio to scale to.
    :type catalogue: dict
    :param catalogue: ``{model: {direction: path}}``, as
        :func:`wdf.mock.waveforms.ccsn_catalogue` returns.
    :return: dict -- the injection's parameters. There is no inclination: the
        orientation that matters is already fixed by the direction drawn.
    :raises ValueError: if the catalogue is empty.
    """
    if not catalogue:
        raise ValueError("a core-collapse injection needs a waveform catalogue")

    model = str(rng.choice(sorted(catalogue)))
    direction = str(rng.choice(sorted(catalogue[model])))

    return {
        "category": "ccsn",
        "subclass": model,
        "direction": direction,
        "waveform_path": catalogue[model][direction],
        "ra": float(rng.uniform(0.0, 2.0 * np.pi)),
        "dec": float(np.arcsin(rng.uniform(-1.0, 1.0))),
        "polarization": float(rng.uniform(0.0, 2.0 * np.pi)),
        "target_snr": float(rng.uniform(*snr_range)),
    }


def _polarisations(spec, sample_rate):
    """The two polarisations of one astrophysical injection, and where they start.

    Compact binaries are generated from their parameters and supernovae read
    from a catalogue, but both reach the detectors the same way --- through the
    antenna response and the time of flight --- so everything downstream of this
    takes the same three values and does not ask which it is holding.

    :param spec: the injection's drawn parameters.
    :type sample_rate: int
    :param sample_rate: rate to generate at, Hz.
    :return: tuple -- ``(hp, hc, start_offset)``, the polarisations and the time
        of their first sample relative to the injection's reference time, which
        is the merger for a compact binary and core bounce for a supernova.
    :raises RuntimeError: if the two polarisations differ in length.
    """
    if spec["category"] == "ccsn":
        hp, hc, start_offset = waveforms.ccsn_polarisations(
            spec["waveform_path"], sample_rate)
    else:
        series_p, series_c = waveforms.cbc_polarisations(
            spec["mass1"],
            spec["mass2"],
            spec["spin1z"],
            spec["spin2z"],
            inclination=spec["inclination"],
            f_lower=spec["f_lower"],
            sample_rate=sample_rate,
            approximant=spec["approximant"],
        )
        hp = np.asarray(series_p, dtype=float)
        hc = np.asarray(series_c, dtype=float)
        start_offset = float(series_p.start_time)

    if len(hp) != len(hc):
        raise RuntimeError("Generated hp and hc have different lengths")
    return hp, hc, start_offset


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

    if prepared["category"] in ("cbc", "ccsn"):
        hp, hc, start_offset = _polarisations(prepared, sample_rate)

        end_offset = start_offset + len(hp) / float(sample_rate)
        prepared["support_before"] = max(-start_offset, 0.0)
        prepared["support_after"] = max(end_offset, 0.0)
        prepared["duration"] = len(hp) / float(sample_rate)

        if prepared["category"] == "ccsn":
            # The three inner products that decide how the network amplitude
            # divides between the detectors. They are kept here because this is
            # where the waveform is already in hand: recomputing them inside the
            # sky redraws would re-read and resample the catalogue file once per
            # attempt, for a quantity that does not depend on the sky at all.
            prepared["hp_energy"] = float(hp @ hp)
            prepared["hc_energy"] = float(hc @ hc)
            prepared["cross_energy"] = float(hp @ hc)
    else:
        samples = _glitch_strain(prepared, sample_rate)
        duration = len(samples) / float(sample_rate)
        prepared["support_before"] = 0.5 * duration
        prepared["support_after"] = duration - prepared["support_before"]
        prepared["duration"] = duration

    return prepared


def _detector_share(spec, gps, detectors):
    """Each detector's share of the network amplitude, from geometry alone.

    For one waveform seen through equal spectra the signal-to-noise ratio a
    detector receives is proportional to the norm of what reaches it. For a
    circular binary the two polarisations are fixed by the inclination and that
    norm reduces to

        a = sqrt( (F+ (1 + cos^2 i) / 2)^2 + (Fx cos i)^2 ),

    while for a waveform read from a catalogue the polarisations are whatever
    the simulation produced and the norm is taken on them directly,

        a^2 = F+^2 <h+,h+> + Fx^2 <hx,hx> + 2 F+ Fx <h+,hx>,

    which is the same quantity without the circular-binary assumption. Either
    way the shares a_i / sqrt(sum a_i^2) are exact when the detectors' spectra
    are equal --- the simulated set's case --- and an approximation where they
    differ.

    :param spec: the injection's drawn parameters.
    :type gps: float
    :param gps: geocentric time the antenna response is taken at.
    :param detectors: the detector names.
    :return: numpy.ndarray -- one share per detector, unit norm.
    """
    from pycbc.detector import Detector

    # A waveform read from a simulation has two polarisations that are neither
    # in phase nor scaled copies of one another, so the amplitude a detector
    # receives is the norm of `F+ h+ + Fx hx` itself, cross term included, and
    # not a function of an inclination. The expression above describes a
    # circular binary and would misstate the split for anything else.
    from_catalogue = spec["category"] == "ccsn"
    cosine = 0.0 if from_catalogue else np.cos(float(spec["inclination"]))

    amplitude = []
    for ifo in detectors:
        f_plus, f_cross = Detector(ifo).antenna_pattern(
            float(spec["ra"]), float(spec["dec"]),
            float(spec["polarization"]), float(gps))
        if from_catalogue:
            energy = (f_plus * f_plus * float(spec["hp_energy"])
                      + f_cross * f_cross * float(spec["hc_energy"])
                      + 2.0 * f_plus * f_cross * float(spec["cross_energy"]))
            amplitude.append(np.sqrt(max(energy, 0.0)))
        else:
            amplitude.append(np.hypot(f_plus * (1.0 + cosine * cosine) / 2.0,
                                      f_cross * cosine))
    amplitude = np.asarray(amplitude, dtype=float)
    norm = float(np.linalg.norm(amplitude))
    if norm <= 0.0:
        return np.zeros(len(amplitude))
    return amplitude / norm


def _enforce_detector_floor(rng, spec, gps, detectors, snr_range, floor,
                            attempts=500):
    """Redraw sky and orientation until every detector receives `floor`.

    The population this produces is the one every detector sees: sources whose
    projection nulls a detector are redrawn, so no injection is carried by one
    detector alone. The network target is raised where even the best split of
    the drawn one cannot give each detector its floor, and stays inside
    `snr_range`.

    :param rng: the generator the redraws come from.
    :param spec: the injection's parameters, modified in place.
    :type gps: float
    :param gps: geocentric time of the merger.
    :param detectors: the detector names.
    :param snr_range: the allowed network signal-to-noise ratio.
    :type floor: float
    :param floor: least signal-to-noise ratio any detector may receive.
    :raises ValueError: if the range cannot hold the floor at all, or no
        acceptable geometry is found.
    """
    low, high = float(snr_range[0]), float(snr_range[1])
    if high < floor * np.sqrt(float(len(detectors))):
        raise ValueError(
            f"a per-detector floor of {floor:g} needs a network "
            f"signal-to-noise ratio of at least "
            f"{floor * np.sqrt(len(detectors)):.1f}, above the range's top "
            f"{high:g}")
    for _ in range(int(attempts)):
        share = _detector_share(spec, gps, detectors)
        least = float(share.min())
        if least > 0.0 and floor / least <= high:
            target = float(spec["target_snr"])
            needed = floor / least
            if target < needed:
                # The drawn loudness cannot give this geometry its floor, so
                # the target is redrawn from the part of the range that can.
                spec["target_snr"] = float(rng.uniform(max(low, needed), high))
            return
        spec["inclination"] = float(np.arccos(rng.uniform(-1.0, 1.0)))
        spec["ra"] = float(rng.uniform(0.0, 2.0 * np.pi))
        spec["dec"] = float(np.arcsin(rng.uniform(-1.0, 1.0)))
        spec["polarization"] = float(rng.uniform(0.0, 2.0 * np.pi))
    raise ValueError(
        f"no sky and orientation gave every detector {floor:g} within "
        f"{attempts} redraws")


def _edge_pads(edge_pad):
    """Resolve an edge padding to the pair of spans kept free at the two ends.

    The two ends of a stretch do not reserve the same amount: the start has to
    clear whatever a search fits its noise model on, the end whatever the
    conditioning chain reads ahead of what it emits. A scalar is the pair whose
    ends are equal.

    :param edge_pad: a span in seconds, or a ``(start, end)`` pair of them.
    :return: tuple[float, float] -- the spans kept free at start and at end.
    :raises ValueError: if either span is negative, or a sequence is given that
        is not a pair.
    """
    if np.isscalar(edge_pad):
        pads = (float(edge_pad), float(edge_pad))
    else:
        values = tuple(edge_pad)
        if len(values) != 2:
            raise ValueError(
                f"edge_pad takes a span or a (start, end) pair, got "
                f"{len(values)} values")
        pads = (float(values[0]), float(values[1]))
    if pads[0] < 0.0 or pads[1] < 0.0:
        raise ValueError("edge_pad spans must be non-negative")
    return pads


def draw_injections(
    n_cbc=250,
    n_glitch=750,
    n_ccsn=0,
    ccsn_catalogue=None,
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
    cbc_mix=DEFAULT_CBC_MIX,
    min_detector_snr=None,
):
    """Draw and place non-overlapping CBC and glitch injections.

    CBC ``gps`` is the geocentric merger time. Glitch ``gps`` is the centre
    of its generated sample array. Placement uses the actual generated
    waveform support and reserves ``detector_delay_pad`` on both sides.

    ``min_detector_snr``, when given, redraws each compact binary's sky and
    orientation until every detector receives at least that signal-to-noise
    ratio, raising the network target inside ``snr_range`` where the geometry
    needs it. The population is then the one every detector sees; a source
    projected onto one detector alone cannot occur in it, and the aggregate
    efficiencies are no longer diluted by signals nothing could recover.

    ``edge_pad`` is a span in seconds kept free at each end, or a
    ``(start, end)`` pair of them where the two ends reserve different amounts.
    An injection placed inside either is placed where a search does not look,
    and is then indistinguishable from one it looked for and missed.

    ``n_ccsn`` draws core-collapse supernova waveforms from ``ccsn_catalogue``,
    the index :func:`wdf.mock.waveforms.ccsn_catalogue` returns. They reach the
    detectors as compact binaries do, through the antenna response and the time
    of flight, and differ in carrying no closed-form parameters and no
    inclination.
    """
    if n_cbc < 0 or n_glitch < 0 or n_ccsn < 0:
        raise ValueError("Injection counts must be non-negative")
    if n_ccsn and not ccsn_catalogue:
        raise ValueError("n_ccsn needs a ccsn_catalogue to draw from")
    pad_start, pad_end = _edge_pads(edge_pad)
    if duration <= pad_start + pad_end:
        raise ValueError("duration must be larger than the edge padding")
    if minimum_gap < 0.0:
        raise ValueError("minimum_gap must be non-negative")

    rng = np.random.default_rng(seed)
    specs = [_draw_cbc(rng, snr_range, cbc_mix) for _ in range(n_cbc)]
    specs.extend(_draw_ccsn(rng, snr_range, ccsn_catalogue) for _ in range(n_ccsn))
    specs.extend(_draw_glitch(rng, snr_range, detectors) for _ in range(n_glitch))
    specs = [_prepare_injection_support(spec, sample_rate) for spec in specs]
    rng.shuffle(specs)

    n_requested = len(specs)
    usable_start = float(start_gps) + pad_start
    usable_end = float(start_gps) + float(duration) - pad_end
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
        if item["category"] in ("cbc", "ccsn") and min_detector_snr is not None:
            _enforce_detector_floor(rng, item, float(gps), detectors,
                                    snr_range, float(min_detector_snr))
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
    psd=None,
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
            psd=psd if psd is None else psd.get(spec["detector"], None),
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

    hp, hc, waveform_start_offset = _polarisations(spec, sample_rate)
    projected = project_cbc(
        hp,
        hc,
        spec["ra"],
        spec["dec"],
        spec["polarization"],
        spec["gps"],
        detectors,
    )

    # Against the spectrum of the detector the signal is being projected onto,
    # measured when one is given: the amplitude of an injection is only
    # meaningful against the noise it has to be seen through, and on recorded
    # data that is neither the design curve nor the same in the two detectors.
    unscaled_snrs = {
        ifo: optimal_snr(
            samples,
            sample_rate=sample_rate,
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            psd_name=psd_name,
            psd=None if psd is None else psd.get(ifo),
        )
        for ifo, (samples, _) in projected.items()
    }
    network0 = float(np.sqrt(sum(value * value for value in unscaled_snrs.values())))
    if network0 <= 0.0:
        raise RuntimeError(
            f"Zero network SNR for {spec['category']} {spec['injection_id']}")

    scale = float(spec["target_snr"]) / network0
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
            psd=None if psd is None else psd.get(ifo),
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
    n_ccsn=0,
    ccsn_catalogue=None,
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
    cbc_mix=DEFAULT_CBC_MIX,
    min_detector_snr=None,
):
    """Generate and write a complete mock foreground/background data set.

    ``sample_rate`` is the frame generation rate. ``analysis_sample_rate`` is
    the rate after WDF resampling and is used only to cap the SNR/filter band.
    ``edge_pad`` is a span in seconds kept free at each end, or a
    ``(start, end)`` pair of them; see :func:`draw_injections`. ``n_ccsn`` draws
    that many core-collapse supernova waveforms from ``ccsn_catalogue``.
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
        n_ccsn=n_ccsn,
        ccsn_catalogue=ccsn_catalogue,
        duration=duration,
        start_gps=start_gps,
        edge_pad=edge_pad,
        snr_range=snr_range,
        seed=seed,
        sample_rate=sample_rate,
        detectors=detectors,
        minimum_gap=minimum_injection_gap,
        strict=strict,
        cbc_mix=cbc_mix,
        min_detector_snr=min_detector_snr,
    )

    requested = int(n_cbc) + int(n_glitch) + int(n_ccsn)
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

    # The background frames are already written, and nothing reads `noise`
    # again, so the foreground takes the arrays over rather than copying them.
    # The copy was a second full-length series per detector held for no reader:
    # at a few days of livetime that is the difference between fitting in
    # memory and swapping. What is produced is unchanged --- the same samples
    # from the same seeds --- since injection adds into these arrays in place
    # either way.
    foreground = {ifo: noise.pop(ifo) for ifo in detectors}
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
 