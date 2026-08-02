"""Geometry of WDF's own forward wavelet coefficients (the `wt0..wtN` trigger
CSV columns, written at `fullPrint>=1`) -- where each coefficient index sits
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
    """Energy-based, SNR-like statistic computed directly from the raw
    wavelet coefficients `wt*`, thresholded with `donoho_johnstone_threshold`.

    - `energy`: sum of surviving (above-threshold) coefficients' `|wt|^2` --
      an integrated energy over the window, not a single peak sample.
    - `snr`: `sqrt(energy) / sigma` -- an amplitude-scale statistic (same
      "sqrt of energy" normalization coherent WaveBurst's own `rho`/`eta_c`
      detection statistic uses, `eta_c ~ sqrt(E_c)`), so it stays roughly on
      a "how many noise-sigma" scale instead of a squared-power scale.
    - `n_above_threshold`: how many of `len(wt)` coefficients survived.
    - `threshold`: the threshold value used.
    """
    threshold = donoho_johnstone_threshold(sigma, len(wt))
    mask = np.abs(wt) >= threshold
    energy = float(np.sum(np.asarray(wt)[mask] ** 2))
    return dict(
        energy=energy,
        snr=float(np.sqrt(energy) / sigma) if sigma > 0 else float("nan"),
        n_above_threshold=int(mask.sum()),
        threshold=float(threshold),
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
    thresh, vmax = np.percentile(mags, 100 * (1 - top_frac)), mags.max()

    style = dict(fill=False, edgecolor="white", lw=1.1)
    style.update(rect_kwargs)

    n_drawn = 0
    for t_lo, t_hi, f_lo, f_hi, mag in tiles:
        if mag < thresh:
            continue
        f_lo = max(f_lo, 0.1)  # avoid a zero/negative lower edge under a log frequency axis
        alpha = 0.25 + 0.65 * min(mag / vmax, 1.0)
        ax.add_patch(Rectangle((t0 + t_lo, f_lo), t_hi - t_lo, f_hi - f_lo, alpha=alpha, **style))
        n_drawn += 1
    return n_drawn
