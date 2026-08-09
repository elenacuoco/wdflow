"""Geometry of WDF's own forward wavelet coefficients -- where each index sits
in (time, frequency), with no assumption about the shape of whatever
transient produced them.

`pytsa.WaveletTransform` / GSL's `gsl_wavelet_transform` is a standard
pyramidal (Mallat) DWT in GSL's documented packed coefficient layout
(confirmed against `p4TSA/src/WaveletTransform.cpp` and the GSL DWT manual):
`wt[0]` is the single coarsest scaling coefficient; level `j = 0..J-1`
(`J = log2(Ncoeff)`) occupies indices `[2^j, 2^(j+1)-1]` (`2^j` coefficients),
each tiling a block of `Ncoeff/2^j` samples, doubling in frequency band each
level up to Nyquist at the finest level. No function anywhere in wdf/p4TSA
exposes this index -> (time, frequency) mapping -- this is a derived reading
of that documented, transform-level layout (fixed for a given Ncoeff/fs, the
same for any input), not something WDF itself computes. Band edges are
therefore approximate (filter-length overlap/leakage in a 9-tap B-spline
wavelet, e.g. "BsplineC309").

Avoids hand-picked reductions of the coefficient VALUES (WDF is meant to find
transients of arbitrary shape and estimate their parameters coherently, and
an arbitrary heuristic -- "the peak is the biggest coefficient", "keep the
top 10%" -- bakes in assumptions about what a signal looks like that may not
hold for a different transient shape). `wavelet_energy_snr` is the one
exception: it thresholds coefficients with the Donoho-Johnstone *universal*
threshold (Donoho & Johnstone 1994), a general, transient-shape-agnostic
statistical rule (depends only on the noise scale and coefficient count, not
on any assumed signal shape), then sums the surviving coefficients' energy.
The intended longer-term consumer of the raw `wt*` values themselves is a
downstream learned model (e.g. a normalizing flow) trained to estimate
parameters directly from them -- `wavelet_energy_snr` is a principled interim
statistic, not a replacement for that.

Kept free of wdf/pytsa imports, like the rest of wdfLib besides io.py --
operates on plain numpy arrays / already-materialized `wt*` DataFrame
columns, so it works offline on saved trigger CSVs without a wdf/pytsa
install.
"""
from __future__ import annotations

import numpy as np


def coeff_levels(n_coeff: int) -> tuple[np.ndarray, np.ndarray]:
    """(level, position) per coefficient index for a length-`n_coeff`
    (power of 2) packed-DWT array. `level[0] == -1` for the single coarsest
    scaling coefficient (index 0); for `k >= 1`, `level[k] == j` where index
    `k` falls in octave level `j`'s block `[2^j, 2^(j+1)-1]`, and
    `position[k]` is `k`'s position within that level (`0..2^j-1`).
    """
    k = np.arange(n_coeff)
    level = np.full(n_coeff, -1, dtype=int)
    position = np.zeros(n_coeff, dtype=int)
    if n_coeff > 1:
        level[1:] = np.floor(np.log2(k[1:])).astype(int)
        position[1:] = k[1:] - (1 << level[1:])
    return level, position


def coeff_freq_bands(n_coeff: int, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """(f_lo, f_hi) in Hz per coefficient index -- doubles each octave level,
    spanning [0, fs/2] (Nyquist) exactly and contiguously across all indices.
    """
    J = int(np.log2(n_coeff))
    level, _ = coeff_levels(n_coeff)
    f_lo = np.where(level < 0, 0.0, fs / 2.0 ** (J - level + 1))
    f_hi = np.where(level < 0, fs / 2.0 ** (J + 1), fs / 2.0 ** (J - level))
    return f_lo, f_hi


def coeff_time_bounds(n_coeff: int, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """(t_lo, t_hi) in seconds (relative to the analysis window's start) per
    coefficient index -- each level's coefficients tile the full window
    [0, n_coeff/fs) exactly and contiguously, with block width halving each
    level up.
    """
    level, position = coeff_levels(n_coeff)
    lvl = np.where(level < 0, 0, level)
    block = n_coeff / (2.0 ** lvl)
    t_lo = np.where(level < 0, 0.0, position * block / fs)
    t_hi = np.where(level < 0, n_coeff / fs, (position + 1) * block / fs)
    return t_lo, t_hi


def wavelet_coeff_tiles(wt: np.ndarray, fs: float) -> list[tuple[float, float, float, float, float]]:
    """Per-coefficient `(t_lo, t_hi, f_lo, f_hi, |wt|)` tiles -- see module
    docstring for the approximation this represents.
    """
    n = len(wt)
    t_lo, t_hi = coeff_time_bounds(n, fs)
    f_lo, f_hi = coeff_freq_bands(n, fs)
    return list(zip(t_lo.tolist(), t_hi.tolist(), f_lo.tolist(), f_hi.tolist(), np.abs(wt).tolist()))


def donoho_johnstone_threshold(sigma: float, n_coeff: int) -> float:
    """The universal wavelet-denoising threshold (Donoho & Johnstone, 1994):
    `sigma * sqrt(2 * ln(n_coeff))`. General (depends only on the noise scale
    and the number of coefficients, not on any assumption about the signal's
    shape) -- the same principle WDF's own C++ thresholding already defaults
    to (`WaveletThreshold.dohonojohnston`), reproduced here directly on `wt*`
    so it's inspectable/reusable from Python instead of only living inside
    opaque C++.
    """
    return sigma * np.sqrt(2.0 * np.log(n_coeff))


def wavelet_energy_snr(wt: np.ndarray, sigma: float) -> dict:
    """Integrated wavelet-domain energy and the SNR it implies on a noise scale.

    Sums the energy of every coefficient, giving an integrated statistic over
    the window rather than a single peak sample. The `sqrt(energy)/sigma`
    normalization keeps the result on an amplitude ("how many noise sigma")
    scale rather than a squared-power one. `wt` is expected in the form
    `WaveletThreshold` emits: coefficients that did not pass thresholding are
    exactly 0, and under soft thresholding the survivors are shrunk by the
    threshold, so the zeros already carry the significance decision.

    :type wt: numpy.ndarray
    :param wt: thresholded wavelet coefficients of one analysis window.
    :type sigma: float
    :param sigma: noise scale the SNR is expressed in.
    :return: dict -- `energy` (sum of `|wt|^2`), `snr` (`sqrt(energy)/sigma`,
        `nan` when `sigma <= 0`), `n_nonzero` (count of non-zero coefficients).
    """
    energy = float(np.sum(np.asarray(wt) ** 2))
    return dict(
        energy=energy,
        snr=float(np.sqrt(energy) / sigma) if sigma > 0 else float("nan"),
        n_nonzero=int(np.count_nonzero(wt)),
    )


def plot_wavelet_tiles(ax, wt: np.ndarray, fs: float, t0: float = 0.0, top_frac: float = 0.10, **rect_kwargs) -> int:
    """Draw the top `top_frac` (by `|wt|`) coefficient tiles as rectangle
    patches on `ax` (e.g. overlaid on a Q-scan Axes whose x-data is absolute
    GPS: pass `t0=`the trigger window's start GPS so tiles land in that
    frame; for a standalone plot in window-relative seconds, leave `t0=0.0`).

    `top_frac` is purely a plotting-legibility cutoff (512 tiles otherwise
    bury whatever's underneath) -- it does not feed into any returned value
    or downstream computation, so it carries none of the "which frequency is
    THE peak" assumption that a summary statistic would.

    Returns the number of tiles drawn.
    """
    from matplotlib.patches import Rectangle

    tiles = wavelet_coeff_tiles(wt, fs)
    mags = np.array([t[4] for t in tiles])
    if mags.max() <= 0:
        return 0

    # A zero coefficient is not a faint tile, it is a tile thresholding removed,
    # and drawing it says the search found something there. It also breaks the
    # percentile: at the measured density of about half a per cent, the 90th
    # percentile of |wt| is exactly zero, so every tile clears the cut and the
    # overlay becomes a solid block covering whatever it was drawn over.
    surviving = mags > 0
    thresh = np.percentile(mags[surviving], 100 * (1 - top_frac))
    vmax = mags.max()

    style = dict(fill=False, edgecolor="white", lw=1.1)
    style.update(rect_kwargs)

    n_drawn = 0
    for t_lo, t_hi, f_lo, f_hi, mag in tiles:
        if mag <= 0 or mag < thresh:
            continue
        f_lo = max(f_lo, 0.1)  # avoid a zero/negative lower edge under a log frequency axis
        alpha = 0.25 + 0.65 * min(mag / vmax, 1.0)
        ax.add_patch(Rectangle((t0 + t_lo, f_lo), t_hi - t_lo, f_hi - f_lo, alpha=alpha, **style))
        n_drawn += 1
    return n_drawn


def tile_frequency(f_lo: float, f_hi: float) -> float:
    """Representative frequency of a wavelet tile spanning `[f_lo, f_hi]`.

    The dyadic tiling is uniform in log frequency, so the geometric centre is
    the centre of the band; the coarsest tile starts at 0 Hz and has none, and
    is represented by half its upper edge.

    :type f_lo: float
    :param f_lo: lower band edge, Hz.
    :type f_hi: float
    :param f_hi: upper band edge, Hz.
    :return: float -- the tile's representative frequency, Hz.
    """
    if f_lo <= 0.0:
        return 0.5 * float(f_hi)
    return float(np.sqrt(f_lo * f_hi))


def dominant_tile(wt: np.ndarray, fs: float, sigma: float | None = None) -> dict:
    """The time-frequency tile carrying the largest wavelet coefficient.

    The transform is orthogonal and the data are whitened, so a coefficient
    divided by the noise scale is the signal-to-noise ratio of that one tile;
    the largest coefficient is therefore the tile where the local
    signal-to-noise ratio peaks, and its band is the frequency the transient
    deposits most of its amplitude in.

    :type wt: numpy.ndarray
    :param wt: wavelet coefficients of one analysis window.
    :type fs: float
    :param fs: sampling frequency the coefficients were computed at, Hz.
    :type sigma: float or None
    :param sigma: noise scale the tile signal-to-noise ratio is expressed in.
    :return: dict -- `freq`, `f_lo`, `f_hi`, `time`, `t_lo`, `t_hi`,
        `magnitude` and `snr` of that tile; empty when `wt` carries no energy.
    """
    wt = np.asarray(wt, dtype=float).reshape(-1)
    if wt.size == 0:
        return {}

    magnitudes = np.abs(wt)
    k = int(np.argmax(magnitudes))
    if magnitudes[k] <= 0.0:
        return {}

    t_lo, t_hi = coeff_time_bounds(wt.size, fs)
    f_lo, f_hi = coeff_freq_bands(wt.size, fs)

    magnitude = float(magnitudes[k])
    return {
        "freq": tile_frequency(f_lo[k], f_hi[k]),
        "f_lo": float(f_lo[k]),
        "f_hi": float(f_hi[k]),
        "time": 0.5 * float(t_lo[k] + t_hi[k]),
        "t_lo": float(t_lo[k]),
        "t_hi": float(t_hi[k]),
        "magnitude": magnitude,
        "snr": float(magnitude / sigma) if (sigma is not None and sigma > 0) else float("nan"),
    }


def wavegram_ridge(wt: np.ndarray, fs: float, sigma: float | None = None,
                   n_time_bins: int | None = None) -> dict:
    """The time-frequency track of the loudest tile in each time bin.

    Bins the coefficient tiles by the centre of their time support and keeps,
    per bin, the tile with the largest coefficient. The result is the ridge of
    the wavegram: the frequency the transient occupies as a function of time,
    read off WDF's own coefficients rather than off a spectrogram of the
    reconstruction. Bins no tile falls in are left `nan` rather than
    interpolated, so a gap in the track stays visible as a gap.

    :type wt: numpy.ndarray
    :param wt: wavelet coefficients of one analysis window.
    :type fs: float
    :param fs: sampling frequency the coefficients were computed at, Hz.
    :type sigma: float or None
    :param sigma: noise scale the ridge signal-to-noise ratio is expressed in.
    :type n_time_bins: int or None
    :param n_time_bins: number of time bins; defaults to the number of tiles
        of the finest scale, which is the transform's own time resolution.
    :return: dict -- `times`, `freqs`, `magnitudes` and `snr` arrays over the
        bins; empty when `wt` carries no energy.
    """
    wt = np.asarray(wt, dtype=float).reshape(-1)
    if wt.size == 0:
        return {}

    magnitudes = np.abs(wt)
    if magnitudes.max() <= 0.0:
        return {}

    t_lo, t_hi = coeff_time_bounds(wt.size, fs)
    f_lo, f_hi = coeff_freq_bands(wt.size, fs)
    t_mid = 0.5 * (t_lo + t_hi)
    f_mid = np.array([tile_frequency(a, b) for a, b in zip(f_lo, f_hi)])

    if n_time_bins is None:
        n_time_bins = wt.size // 2
    n_time_bins = max(int(n_time_bins), 1)

    span = wt.size / fs
    edges = np.linspace(0.0, span, n_time_bins + 1)
    bin_index = np.clip(np.digitize(t_mid, edges) - 1, 0, n_time_bins - 1)

    freqs = np.full(n_time_bins, np.nan)
    mags = np.full(n_time_bins, np.nan)
    for i in range(n_time_bins):
        members = np.flatnonzero(bin_index == i)
        if members.size == 0:
            continue
        k = members[int(np.argmax(magnitudes[members]))]
        if magnitudes[k] <= 0.0:
            continue
        freqs[i] = f_mid[k]
        mags[i] = magnitudes[k]

    snr = mags / sigma if (sigma is not None and sigma > 0) else np.full(n_time_bins, np.nan)

    return {
        "times": 0.5 * (edges[:-1] + edges[1:]),
        "freqs": freqs,
        "magnitudes": mags,
        "snr": snr,
    }


def peak_frequency(wt: np.ndarray, fs: float) -> float:
    """Frequency at which the transient's local amplitude peaks.

    The loudest single coefficient locates the peak in time and frequency, but
    reading its band off directly quantises the answer onto the dyadic ladder:
    a tile is an octave wide, so the estimate can only take one value per
    octave. It also biases high-frequency narrowband signals low, because at
    fine scales a tile is a couple of samples long, the signal spreads over
    many of them, and each individual coefficient is smaller than one at a
    coarser scale that captured more of it -- so the single largest coefficient
    sits below the true carrier.

    The estimate is the energy-weighted geometric mean of the tile frequencies,
    each tile weighted also by how much of its time support the loudest tile
    covers. Nothing is admitted or excluded by a test, so the estimate moves
    continuously as the coefficients do rather than jumping when a tile crosses
    a boundary, and no level is privileged. The loudest tile dominates, the
    tiles sharing its time contribute in proportion to how much they share, and
    tiles elsewhere in the window fall away.

    Weighting rather than selecting is what lets the answer leave the dyadic
    ladder. A tile is an octave wide, so a rule that picks tiles can only ever
    return one of a handful of frequencies; a rule that weights them returns a
    value between, and moves it as the energy shifts. The mean is geometric
    because the tiling is uniform in log frequency.

    What it cannot do: when every surviving coefficient sits at one octave
    level, there is nothing to interpolate between and the honest answer is that
    level's own frequency. The ladder is then the transform's real resolution,
    not an artefact of this function.

    ``freqPeak`` remains a poorer estimator of a carrier than ``freqMean``,
    which is a spectral moment and the right instrument when the signal *has*
    one carrier. It earns its place on transients that sweep, where
    ``freqMean`` answers a different question.

    :type wt: numpy.ndarray
    :param wt: wavelet coefficients of one analysis window.
    :type fs: float
    :param fs: sampling frequency the coefficients were computed at, Hz.
    :return: float -- the peak frequency in Hz, `nan` when `wt` carries no
        energy.
    """
    wt = np.asarray(wt, dtype=float).reshape(-1)
    if wt.size == 0:
        return float("nan")

    magnitude = np.abs(wt)
    k = int(np.argmax(magnitude))
    if magnitude[k] <= 0.0:
        return float("nan")

    t_lo, t_hi = coeff_time_bounds(wt.size, fs)
    f_lo, f_hi = coeff_freq_bands(wt.size, fs)
    f_mid = np.array([tile_frequency(a, b) for a, b in zip(f_lo, f_hi)])

    shared = np.maximum(0.0, np.minimum(t_hi, t_hi[k]) - np.maximum(t_lo, t_lo[k]))
    weight = magnitude ** 2 * shared / np.maximum(t_hi - t_lo, np.finfo(float).tiny)
    total = float(weight.sum())
    if total <= 0.0:
        return float(f_mid[k])

    frequency = np.maximum(f_mid, np.finfo(float).tiny)
    return float(np.exp(float(weight @ np.log(frequency)) / total))
