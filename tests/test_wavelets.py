import numpy as np
import pytest

from wdf.analysis.wavelets import (
    coeff_freq_bands,
    coeff_levels,
    coeff_time_bounds,
    donoho_johnstone_threshold,
    plot_wavelet_tiles,
    wavelet_coeff_tiles,
    wavelet_energy_snr,
)

NCOEFF = 512
FS = 2048.0


def test_coeff_levels_partition_is_exhaustive():
    level, position = coeff_levels(NCOEFF)
    assert level[0] == -1
    J = int(np.log2(NCOEFF))
    for j in range(J):
        idx = np.where(level == j)[0]
        assert len(idx) == 2 ** j
        assert idx.min() == 2 ** j and idx.max() == 2 ** (j + 1) - 1
        assert (position[idx] == np.arange(2 ** j)).all()


def test_freq_bands_span_zero_to_nyquist_contiguously():
    f_lo, f_hi = coeff_freq_bands(NCOEFF, FS)
    level, _ = coeff_levels(NCOEFF)
    J = int(np.log2(NCOEFF))
    assert f_lo[0] == 0.0
    assert f_hi[level == J - 1][0] == FS / 2
    # band edges double each level and are contiguous level-to-level
    for j in range(J - 1):
        this_hi = f_hi[level == j][0]
        next_lo = f_lo[level == j + 1][0]
        assert this_hi == pytest.approx(next_lo)
    assert f_hi[0] == pytest.approx(f_lo[level == 0][0])


def test_time_bounds_tile_window_without_gaps_or_overlap():
    t_lo, t_hi = coeff_time_bounds(NCOEFF, FS)
    level, position = coeff_levels(NCOEFF)
    window_s = NCOEFF / FS
    assert t_lo[0] == 0.0 and t_hi[0] == pytest.approx(window_s)
    for j in range(int(np.log2(NCOEFF))):
        idx = np.where(level == j)[0]
        order = idx[np.argsort(position[idx])]
        assert t_lo[order[0]] == pytest.approx(0.0)
        assert t_hi[order[-1]] == pytest.approx(window_s)
        for a, b in zip(order[:-1], order[1:]):
            assert t_hi[a] == pytest.approx(t_lo[b])


def test_wavelet_coeff_tiles_shape_and_ordering():
    wt = np.random.default_rng(0).normal(size=NCOEFF)
    tiles = wavelet_coeff_tiles(wt, FS)
    assert len(tiles) == NCOEFF
    for (t_lo, t_hi, f_lo, f_hi, mag) in tiles:
        assert t_hi >= t_lo and f_hi >= f_lo and mag >= 0


def test_donoho_johnstone_threshold_matches_closed_form():
    sigma, n = 2.5, 512
    assert donoho_johnstone_threshold(sigma, n) == pytest.approx(sigma * np.sqrt(2 * np.log(n)))
    # scales linearly with sigma, monotonically (slowly) with n
    assert donoho_johnstone_threshold(2 * sigma, n) == pytest.approx(2 * donoho_johnstone_threshold(sigma, n))
    assert donoho_johnstone_threshold(sigma, 4 * n) > donoho_johnstone_threshold(sigma, n)


def test_wavelet_energy_snr_counts_every_surviving_coefficient():
    """`wt` reaches Python already thresholded by p4TSA's WaveletThreshold
    (sub-threshold coefficients are exactly 0), so the energy sum must take
    every non-zero coefficient -- including small ones, which in the default
    soft mode are legitimate survivors shrunk by the threshold.
    """
    sigma = 1.0
    thresh = donoho_johnstone_threshold(sigma, NCOEFF)
    wt = np.zeros(NCOEFF)          # C++ zeroed everything sub-threshold
    wt[10] = thresh * 3            # a loud survivor
    wt[20] = thresh * 1e-3         # a soft-shrunk survivor: tiny but real

    result = wavelet_energy_snr(wt, sigma)
    assert result["n_nonzero"] == 2
    expected = (thresh * 3) ** 2 + (thresh * 1e-3) ** 2
    assert result["energy"] == pytest.approx(expected, rel=1e-6)
    assert result["snr"] == pytest.approx(np.sqrt(result["energy"]) / sigma, rel=1e-6)


def test_wavelet_energy_snr_does_not_reapply_the_threshold():
    """Regression guard: re-applying Donoho-Johnstone here discarded 69-90% of
    each real trigger's energy (measured on GW250114) and left most triggers at
    snr == 0, because soft thresholding upstream shrinks survivors below it.
    """
    sigma = 1.0
    thresh = donoho_johnstone_threshold(sigma, NCOEFF)
    wt = np.zeros(NCOEFF)
    wt[[5, 6, 7]] = thresh * 0.1   # all survivors, all below a second DJ cut

    result = wavelet_energy_snr(wt, sigma)
    assert result["n_nonzero"] == 3
    assert result["energy"] > 0
    assert result["snr"] > 0


def test_wavelet_energy_snr_energy_is_additive_across_coefficients():
    sigma = 1.0
    thresh = donoho_johnstone_threshold(sigma, NCOEFF)
    wt = np.zeros(NCOEFF)
    wt[[10, 20, 30]] = thresh * np.array([2.0, 3.0, 4.0])

    result = wavelet_energy_snr(wt, sigma)
    assert result["n_nonzero"] == 3
    expected_energy = sum((thresh * m) ** 2 for m in (2.0, 3.0, 4.0))
    assert result["energy"] == pytest.approx(expected_energy, rel=1e-6)


def test_plot_wavelet_tiles_draws_top_fraction():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wt = np.random.default_rng(3).normal(size=NCOEFF)
    fig, ax = plt.subplots()
    n_drawn = plot_wavelet_tiles(ax, wt, FS, t0=1000.0, top_frac=0.1)
    assert n_drawn == pytest.approx(NCOEFF * 0.1, abs=NCOEFF * 0.02)
    assert len(ax.patches) == n_drawn
    plt.close(fig)


def test_dominant_tile_is_the_band_of_the_largest_coefficient():
    """freqPeak is read off the tile where the local signal-to-noise ratio peaks."""
    from wdf.analysis.wavelets import coeff_freq_bands, dominant_tile

    fs, n = 2048.0, 512
    wt = np.zeros(n)
    k = 100
    wt[k] = 5.0
    wt[7] = 1.0

    f_lo, f_hi = coeff_freq_bands(n, fs)
    tile = dominant_tile(wt, fs, sigma=2.0)

    assert tile["f_lo"] == f_lo[k]
    assert tile["f_hi"] == f_hi[k]
    assert f_lo[k] <= tile["freq"] <= f_hi[k]
    assert tile["snr"] == 2.5


def test_dominant_tile_ignores_a_window_with_no_energy():
    from wdf.analysis.wavelets import dominant_tile

    assert dominant_tile(np.zeros(64), 2048.0) == {}


def test_wavegram_ridge_follows_a_rising_track():
    """One frequency per time bin, taken from the loudest tile in that bin."""
    from wdf.analysis.wavelets import coeff_levels, wavegram_ridge

    fs, n = 2048.0, 512
    level, position = coeff_levels(n)
    wt = np.zeros(n)
    # Two coefficients of different octaves, the later one an octave higher.
    early = np.flatnonzero((level == 6) & (position == 4))[0]
    late = np.flatnonzero((level == 7) & (position == 100))[0]
    wt[early] = 3.0
    wt[late] = 4.0

    ridge = wavegram_ridge(wt, fs, sigma=1.0, n_time_bins=8)

    present = np.isfinite(ridge["freqs"])
    assert present.any()
    assert np.nanmax(ridge["snr"]) == 4.0
    assert ridge["times"][np.nanargmax(ridge["magnitudes"])] > 0.0


def test_peak_frequency_is_not_quantised_onto_the_octave_ladder():
    """Reading the loudest tile's band directly can only return one value per
    octave; weighting the tiles around it in time makes the estimate continuous."""
    from wdf.analysis.wavelets import dominant_tile, peak_frequency

    fs, n = 2048.0, 512
    seen_tile, seen_weighted = set(), set()
    rng = np.random.default_rng(0)
    for _ in range(40):
        wt = np.zeros(n)
        # a loud coarse-scale coefficient with finer ones inside its span,
        # which is the configuration the weighting exists to correct
        k = int(rng.integers(8, 32))
        wt[k] = 5.0
        for finer in rng.integers(64, n, size=4):
            wt[int(finer)] = rng.uniform(0.5, 3.0)
        seen_tile.add(round(dominant_tile(wt, fs)["freq"], 3))
        seen_weighted.add(round(peak_frequency(wt, fs), 3))

    assert len(seen_weighted) > len(seen_tile)


def test_peak_frequency_stays_inside_the_band_of_the_tiles_it_averages():
    from wdf.analysis.wavelets import coeff_freq_bands, peak_frequency

    fs, n = 2048.0, 512
    wt = np.zeros(n)
    wt[300] = 4.0
    f_lo, f_hi = coeff_freq_bands(n, fs)
    value = peak_frequency(wt, fs)
    assert 0.0 < value <= fs / 2


def test_peak_frequency_of_an_empty_window_is_nan():
    from wdf.analysis.wavelets import peak_frequency

    assert np.isnan(peak_frequency(np.zeros(64), 2048.0))
    assert np.isnan(peak_frequency(np.array([]), 2048.0))


def test_the_tile_overlay_skips_the_coefficients_thresholding_removed():
    """A zero coefficient is an absent tile, not a faint one. Drawing it would
    claim the search found something there -- and because the coefficients are
    about half a per cent dense, the percentile cut is itself zero, so every
    tile would clear it and the overlay would cover the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wdf.analysis.wavelets import plot_wavelet_tiles

    n = 256
    wt = np.zeros(n)
    wt[[10, 40, 77]] = [3.0, 5.0, 1.0]

    fig, ax = plt.subplots()
    drawn = plot_wavelet_tiles(ax, wt, 2048.0, top_frac=1.0)
    plt.close(fig)

    assert drawn == 3


def test_the_overlay_still_honours_the_top_fraction():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wdf.analysis.wavelets import plot_wavelet_tiles

    wt = np.zeros(256)
    wt[:20] = np.arange(1, 21, dtype=float)

    fig, ax = plt.subplots()
    drawn = plot_wavelet_tiles(ax, wt, 2048.0, top_frac=0.5)
    plt.close(fig)

    assert 8 <= drawn <= 12       # about half of the twenty survivors
