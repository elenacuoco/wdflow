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


PACKET = "Coif1P6"
PACKET_N = 1024


def test_a_packet_level_is_named_by_its_mother_and_its_depth():
    from wdf.analysis.wavelets import mother_wavelet, packet_depth, packet_depths

    assert packet_depth(PACKET) == 6 and mother_wavelet(PACKET) == "Coif1"
    assert packet_depth("Coif1") == 0 and mother_wavelet("Coif1") == "Coif1"
    assert packet_depth(None) == 0
    assert packet_depth(6) == 6 and packet_depth(np.int64(3)) == 3
    # A P with nothing before it is no mother and declares no depth.
    assert packet_depth("P6") == 0 and mother_wavelet("P6") == "P6"
    assert packet_depths(["DaubC8", PACKET, "Haar", PACKET]).tolist() == [0, 6, 0, 6]
    assert packet_depths(None).size == 0


def test_a_packet_depth_of_zero_or_below_is_refused():
    from wdf.analysis.wavelets import packet_depth

    with pytest.raises(ValueError, match="depth of 0"):
        packet_depth("HaarP0")
    with pytest.raises(ValueError, match="negative"):
        packet_depth(-1)


def test_a_packet_level_splits_the_band_into_equal_bands_in_frequency_order():
    f_lo, f_hi = coeff_freq_bands(PACKET_N, FS, PACKET)
    width = FS / 2 ** 7
    band = np.arange(PACKET_N) // 16
    assert np.array_equal(f_lo, band * width)
    assert np.array_equal(f_hi, (band + 1) * width)
    assert f_lo[0] == 0.0 and f_hi[-1] == FS / 2


def test_a_packet_coefficient_covers_two_to_the_depth_samples():
    t_lo, t_hi = coeff_time_bounds(PACKET_N, FS, PACKET)
    position = np.arange(PACKET_N) % 16
    assert np.array_equal(t_lo, position * 64 / FS)
    assert np.array_equal(t_hi, (position + 1) * 64 / FS)
    assert t_hi.max() == pytest.approx(PACKET_N / FS)


def test_a_packet_band_takes_the_octave_that_holds_it():
    level, position = coeff_levels(PACKET_N, PACKET)
    pyramid_level, _ = coeff_levels(PACKET_N)
    # [0, 16) Hz is drawn by half its upper edge, in [8, 16); [16, 32) is an
    # octave itself; [256, 272) lies in [256, 512); [1008, 1024) in the last.
    assert level[[0, 16, 256, 1023]].tolist() == [3, 4, 8, 9]
    assert level[256] == pyramid_level[256]
    assert np.array_equal(position, np.arange(PACKET_N) % 16)


def test_a_packet_depth_beyond_the_window_is_refused():
    with pytest.raises(ValueError, match="exceeds"):
        coeff_freq_bands(64, FS, "Coif1P7")


def test_the_geometry_follows_each_basis_of_a_mixed_table():
    """A number stands for a name wherever only the layout matters."""
    assert np.array_equal(coeff_freq_bands(PACKET_N, FS, 6)[0],
                          coeff_freq_bands(PACKET_N, FS, PACKET)[0])
    assert np.array_equal(coeff_freq_bands(PACKET_N, FS, "DaubC8")[0],
                          coeff_freq_bands(PACKET_N, FS)[0])
