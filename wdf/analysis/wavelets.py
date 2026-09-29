"""Geometry of WDF's own forward wavelet coefficients -- where each index sits
in (time, frequency), with no assumption about the shape of whatever
transient produced them.

A coefficient's place in the plane is a property of the basis that produced
it, which the trigger records by name (`wave`), and of the window length and
the rate. Two layouts exist:

- **The pyramid** (a mother's name alone, e.g. ``"DaubC8"``):
  `py4tsa.WaveletTransform` / GSL's `gsl_wavelet_transform` is a standard
  pyramidal (Mallat) DWT in GSL's documented packed coefficient layout
  (`p4TSA/src/WaveletTransform.cpp` and the GSL DWT manual): `wt[0]` is the
  single coarsest scaling coefficient; level `j = 0..J-1` (`J = log2(Ncoeff)`)
  occupies indices `[2^j, 2^(j+1)-1]` (`2^j` coefficients), each tiling a
  block of `Ncoeff/2^j` samples, doubling in frequency band each level up to
  Nyquist at the finest level.
- **A uniform wavelet-packet level** (the mother's name followed by ``P`` and
  the depth, e.g. ``"Coif1P6"``): every band is split `D` times, so the level
  holds `2^D` bands of equal width `fs / 2^(D+1)`, written by p4TSA in
  frequency order, band `b` occupying indices `[b L, (b+1) L)` with
  `L = Ncoeff / 2^D`, and coefficient `k` of a band tiling the samples
  `[k 2^D, (k+1) 2^D)`.

Either way the index alone places the coefficient once the layout is known,
and the layout is fixed for a given basis, `Ncoeff` and `fs`, the same for any
input: this is a reading of the transform's layout, not something computed
from the data. Band edges are nominal: a filter of finite length leaks across
them.

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

Kept free of wdf/py4tsa imports, like the rest of wdfLib besides io.py --
operates on plain numpy arrays / already-materialized `wt*` DataFrame
columns, so it works offline on saved trigger CSVs without a wdf/py4tsa
install.
"""
from __future__ import annotations

import numpy as np


def packet_depth(wave) -> int:
    """The wavelet-packet depth a basis name declares, 0 for the pyramid.

    A name is a mother (``"Haar"``, ``"DaubC8"``, ...) for its pyramidal
    transform, or a mother followed by ``P`` and a depth for that uniform
    level of its packet tree, as p4TSA's `WDF2Classify.SetBases` reads it.

    Every function of this module that takes a `wave` reads it through here,
    so an integer depth may stand for a name wherever only the layout matters.

    :type wave: str, int or None
    :param wave: the basis name a trigger records, or a packet depth; None is
        the pyramid.
    :return: int -- the depth, 0 for the pyramid.
    :raises ValueError: if the name declares a depth of 0, or the depth given
        is negative.
    """
    if wave is None:
        return 0
    if isinstance(wave, (int, np.integer)):
        if wave < 0:
            raise ValueError(f"a packet depth cannot be negative, got {wave}")
        return int(wave)
    name = str(wave)
    at = name.rfind("P")
    if at <= 0 or at == len(name) - 1 or not name[at + 1:].isdigit():
        return 0
    depth = int(name[at + 1:])
    if depth == 0:
        raise ValueError(f"basis {name!r} declares a packet depth of 0")
    return depth


def mother_wavelet(wave: str) -> str:
    """The mother a basis name is built on: the name without its packet depth.

    :type wave: str
    :param wave: the basis name a trigger records.
    :return: str -- the mother's name, as `py4tsa.WaveletTransform` calls it.
    """
    name = str(wave)
    return name[:name.rfind("P")] if packet_depth(name) else name


def packet_depths(waves) -> np.ndarray:
    """`packet_depth` over a column of basis names, one parse per distinct name.

    :param waves: basis names, e.g. a trigger table's `wave` column; None
        stands for a table that records none, which is the pyramid.
    :return: numpy.ndarray -- one integer depth per entry.
    """
    if waves is None:
        return np.zeros(0, dtype=int)
    names = np.asarray(waves, dtype=object)
    distinct, inverse = np.unique(names.astype(str), return_inverse=True)
    return np.array([packet_depth(n) for n in distinct], dtype=int)[inverse]


def _packet_check(n_coeff: int, depth: int) -> None:
    if (1 << depth) > n_coeff:
        raise ValueError(f"a packet depth of {depth} exceeds log2 of a window of {n_coeff}")


def coeff_levels(n_coeff: int, wave=None) -> tuple[np.ndarray, np.ndarray]:
    """(level, position) per coefficient index.

    For the pyramid, `level[0] == -1` for the single coarsest scaling
    coefficient (index 0); for `k >= 1`, `level[k] == j` where index `k` falls
    in octave level `j`'s block `[2^j, 2^(j+1)-1]`, and `position[k]` is `k`'s
    position within that level (`0..2^j-1`).

    For a packet level the octave is not a property of the index but of where
    its band lies, and `level` is the pyramid octave holding the band's
    representative frequency (`tile_frequency`), on the same numbering, so
    that an octave-by-time grid places tiles of either layout on one set of
    rows; `position` is the coefficient's position within its band.

    :type n_coeff: int
    :param n_coeff: window length, a power of 2.
    :type wave: str, int or None
    :param wave: the basis name, or its packet depth; None is the pyramid.
    :return: tuple -- `(level, position)`, integer arrays of length `n_coeff`.
    """
    depth = packet_depth(wave)
    k = np.arange(n_coeff)
    if depth == 0:
        level = np.full(n_coeff, -1, dtype=int)
        position = np.zeros(n_coeff, dtype=int)
        if n_coeff > 1:
            level[1:] = np.floor(np.log2(k[1:])).astype(int)
            position[1:] = k[1:] - (1 << level[1:])
        return level, position
    _packet_check(n_coeff, depth)
    f_lo, f_hi = coeff_freq_bands(n_coeff, 1.0, wave)
    J = int(np.log2(n_coeff))
    pyramid_lo, _ = coeff_freq_bands(n_coeff, 1.0)
    edges = np.unique(pyramid_lo)
    row = np.searchsorted(edges, tile_frequencies(f_lo, f_hi), side="right") - 1
    level = np.clip(row, 0, J) - 1
    return level.astype(int), k % (n_coeff >> depth)


def coeff_freq_bands(n_coeff: int, fs: float, wave=None) -> tuple[np.ndarray, np.ndarray]:
    """(f_lo, f_hi) in Hz per coefficient index, spanning [0, fs/2] (Nyquist)
    exactly and contiguously within each band of the layout.

    For the pyramid the band doubles each octave level; for a packet level of
    depth `D` every band is `fs / 2^(D+1)` wide, in frequency order.

    :type n_coeff: int
    :param n_coeff: window length, a power of 2.
    :type fs: float
    :param fs: sampling frequency, Hz.
    :type wave: str, int or None
    :param wave: the basis name, or its packet depth; None is the pyramid.
    :return: tuple -- `(f_lo, f_hi)`, arrays of length `n_coeff`, Hz.
    """
    depth = packet_depth(wave)
    if depth:
        _packet_check(n_coeff, depth)
        band = np.arange(n_coeff) // (n_coeff >> depth)
        width = fs / 2.0 ** (depth + 1)
        return band * width, (band + 1) * width
    J = int(np.log2(n_coeff))
    level, _ = coeff_levels(n_coeff)
    f_lo = np.where(level < 0, 0.0, fs / 2.0 ** (J - level + 1))
    f_hi = np.where(level < 0, fs / 2.0 ** (J + 1), fs / 2.0 ** (J - level))
    return f_lo, f_hi


def coeff_time_bounds(n_coeff: int, fs: float, wave=None) -> tuple[np.ndarray, np.ndarray]:
    """(t_lo, t_hi) in seconds (relative to the analysis window's start) per
    coefficient index -- each band's coefficients tile the full window
    [0, n_coeff/fs) exactly and contiguously.

    For the pyramid the block width halves each level up; for a packet level
    of depth `D` every coefficient covers `2^D` samples.

    :type n_coeff: int
    :param n_coeff: window length, a power of 2.
    :type fs: float
    :param fs: sampling frequency, Hz.
    :type wave: str, int or None
    :param wave: the basis name, or its packet depth; None is the pyramid.
    :return: tuple -- `(t_lo, t_hi)`, arrays of length `n_coeff`, seconds.
    """
    depth = packet_depth(wave)
    if depth:
        _packet_check(n_coeff, depth)
        position = np.arange(n_coeff) % (n_coeff >> depth)
        step = 2.0 ** depth / fs
        return position * step, (position + 1) * step
    level, position = coeff_levels(n_coeff)
    lvl = np.where(level < 0, 0, level)
    block = n_coeff / (2.0 ** lvl)
    t_lo = np.where(level < 0, 0.0, position * block / fs)
    t_hi = np.where(level < 0, n_coeff / fs, (position + 1) * block / fs)
    return t_lo, t_hi


def ladder_rows(f_lo, f_hi, bands) -> np.ndarray:
    """The row of a band ladder each tile is drawn in, -1 outside it.

    A tile whose band is a row of the ladder is drawn in that row: every
    pyramid tile, including the band from zero of each window length, which
    is a row of its own. A tile whose band is not a row, a packet tile
    narrower than an octave, is drawn in the row holding its representative
    frequency (`tile_frequency`).

    :param f_lo: lower band edge of each tile, Hz.
    :param f_hi: upper band edge of each tile, Hz.
    :param bands: the ladder, `(n_bands, 2)` of `(f_lo, f_hi)` ordered by
        lower edge, as `wdf.analysis.detector_graph.band_grid` returns it.
    :return: numpy.ndarray -- one row per tile, -1 where the band is not a
        row and its representative frequency lies outside every row.
    """
    f_lo = np.asarray(f_lo, dtype=float)
    f_hi = np.asarray(f_hi, dtype=float)
    bands = np.asarray(bands, dtype=float).reshape(-1, 2)
    if not len(bands):
        return np.full(f_lo.shape, -1, dtype=int)
    row_of = {(round(lo, 9), round(hi, 9)): row for row, (lo, hi) in enumerate(bands)}
    exact = np.array([row_of.get((round(lo, 9), round(hi, 9)), -1)
                      for lo, hi in zip(f_lo.ravel().tolist(), f_hi.ravel().tolist())],
                     dtype=int).reshape(f_lo.shape)
    centre = tile_frequencies(f_lo, f_hi)
    row = np.searchsorted(bands[:, 0], centre, side="right") - 1
    clipped = np.clip(row, 0, len(bands) - 1)
    inside = (row >= 0) & (centre < bands[clipped, 1])
    return np.where(exact >= 0, exact, np.where(inside, row, -1)).astype(int)


def wavelet_coeff_tiles(wt: np.ndarray, fs: float,
                        wave=None) -> list[tuple[float, float, float, float, float]]:
    """Per-coefficient `(t_lo, t_hi, f_lo, f_hi, |wt|)` tiles -- see module
    docstring for the approximation this represents; `wave` is the basis
    name, None for the pyramid.
    """
    n = len(wt)
    t_lo, t_hi = coeff_time_bounds(n, fs, wave)
    f_lo, f_hi = coeff_freq_bands(n, fs, wave)
    return list(zip(t_lo.tolist(), t_hi.tolist(), f_lo.tolist(), f_hi.tolist(), np.abs(wt).tolist()))


def donoho_johnstone_threshold(sigma: float, n_coeff: int) -> float:
    """The universal wavelet-denoising threshold (Donoho & Johnstone, 1994):
    `sigma * sqrt(2 * ln(n_coeff))`. General (depends only on the noise scale
    and the number of coefficients, not on any assumption about the signal's
    shape) -- the rule WDF's C++ engine applies under
    `WaveletThreshold.dohonojohnston`, reproduced here directly on `wt*`
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


def plot_wavelet_tiles(ax, wt: np.ndarray, fs: float, t0: float = 0.0, top_frac: float = 0.10,
                       wave=None, **rect_kwargs) -> int:
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

    tiles = wavelet_coeff_tiles(wt, fs, wave)
    mags = np.array([t[4] for t in tiles])
    if mags.max() <= 0:
        return 0

    # A zero coefficient is not a faint tile, it is a tile thresholding removed,
    # and drawing it says the search found something there. It also breaks the
    # percentile: thresholding leaves so few coefficients that most percentiles
    # of |wt| are exactly zero, every tile clears the cut, and the overlay
    # becomes a solid block covering whatever it was drawn over.
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


def tile_frequencies(f_lo, f_hi) -> np.ndarray:
    """`tile_frequency` over arrays of band edges.

    :param f_lo: lower band edges, Hz.
    :param f_hi: upper band edges, Hz.
    :return: numpy.ndarray -- the representative frequency of each tile, Hz.
    """
    f_lo = np.asarray(f_lo, dtype=float)
    f_hi = np.asarray(f_hi, dtype=float)
    return np.where(f_lo <= 0.0, 0.5 * f_hi, np.sqrt(np.maximum(f_lo, 0.0) * f_hi))


def wavegram_ridge(wt: np.ndarray, fs: float, sigma: float | None = None,
                   n_time_bins: int | None = None, wave=None) -> dict:
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
    :type wave: str or None
    :param wave: the basis name, None for the pyramid.
    :return: dict -- `times`, `freqs`, `magnitudes` and `snr` arrays over the
        bins; empty when `wt` carries no energy.
    """
    wt = np.asarray(wt, dtype=float).reshape(-1)
    if wt.size == 0:
        return {}

    magnitudes = np.abs(wt)
    if magnitudes.max() <= 0.0:
        return {}

    t_lo, t_hi = coeff_time_bounds(wt.size, fs, wave)
    f_lo, f_hi = coeff_freq_bands(wt.size, fs, wave)
    t_mid = 0.5 * (t_lo + t_hi)
    f_mid = tile_frequencies(f_lo, f_hi)

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
