import numpy as np
import pytest

from pytsa.tsa import WaveletTransform
from wdf.analysis.coefficients import from_dense
from wdf.analysis.metaparameters import META_FEATURES, meta_features
from wdf.analysis.wavelets import (
    coeff_freq_bands,
    coeff_time_bounds,
    donoho_johnstone_threshold,
)
from wdf.structures.array2SeqView import array2SeqView

FS = 2048.0
WINDOWS = [128, 512]
WAVE = "DaubC12"
SIGMA = 1.0


def thresholded(samples, sigma=SIGMA, wave=WAVE):
    """One window's coefficients as the search reports them: thresholded."""
    n = len(samples)
    view = array2SeqView(0.0, FS, n)
    view = view.Fill(0.0, np.asarray(samples, dtype=float).copy())
    WaveletTransform(n, getattr(WaveletTransform, wave)).Forward(view)
    coefficients = np.array([view.GetY(0, i) for i in range(n)])
    coefficients[np.abs(coefficients) <= donoho_johnstone_threshold(sigma, n)] = 0.0
    return from_dense(coefficients)


def sine_gaussian(n, fs, t0, f0, tau, amplitude=50.0):
    t = np.arange(n) / fs
    return amplitude * np.exp(-((t - t0) / tau) ** 2) * np.sin(2 * np.pi * f0 * (t - t0))


def tile_of(index, n_coeff, fs=FS):
    t_lo, t_hi = coeff_time_bounds(n_coeff, fs)
    f_lo, f_hi = coeff_freq_bands(n_coeff, fs)
    return t_lo[index], t_hi[index], f_lo[index], f_hi[index]


def test_no_surviving_coefficient_leaves_every_parameter_undefined():
    features = meta_features([], [], 512, FS, SIGMA)
    assert sorted(features) == sorted(META_FEATURES)
    assert all(np.isnan(v) for v in features.values())


@pytest.mark.parametrize("n_coeff", WINDOWS)
def test_the_band_brackets_the_frequency_the_burst_was_made_at(n_coeff):
    f0 = FS / 16.0
    index, value = thresholded(
        sine_gaussian(n_coeff, FS, 0.5 * n_coeff / FS, f0, 0.1 * n_coeff / FS))
    features = meta_features(index, value, n_coeff, FS, SIGMA)

    assert features["freqMin"] <= f0 <= features["freqMax"]
    assert features["freqMin"] <= features["freqMean"] <= features["freqMax"]


@pytest.mark.parametrize("n_coeff", WINDOWS)
def test_the_extent_brackets_the_time_the_burst_was_placed_at(n_coeff):
    t0 = 0.5 * n_coeff / FS
    index, value = thresholded(
        sine_gaussian(n_coeff, FS, t0, FS / 16.0, 0.05 * n_coeff / FS))
    features = meta_features(index, value, n_coeff, FS, SIGMA)

    assert features["gpsStart"] <= t0 <= features["gpsStart"] + features["duration"]
    assert features["gpsStart"] <= features["gpsCentroid"]
    assert features["gpsCentroid"] <= features["gpsStart"] + features["duration"]
    assert features["duration"] > 0.0


@pytest.mark.parametrize("n_coeff", WINDOWS)
def test_a_sharper_burst_is_wider_in_frequency(n_coeff):
    """A short transient occupies more band than a long one at the same carrier."""
    span = n_coeff / FS
    long_burst = meta_features(
        *thresholded(sine_gaussian(n_coeff, FS, 0.5 * span, FS / 16.0, 0.15 * span)),
        n_coeff, FS, SIGMA)
    short_burst = meta_features(
        *thresholded(sine_gaussian(n_coeff, FS, 0.5 * span, FS / 16.0, 0.02 * span)),
        n_coeff, FS, SIGMA)

    assert (short_burst["freqMax"] - short_burst["freqMin"]
            >= long_burst["freqMax"] - long_burst["freqMin"])


@pytest.mark.parametrize("n_coeff", WINDOWS)
def test_a_longer_burst_lasts_longer(n_coeff):
    span = n_coeff / FS
    short_burst = meta_features(
        *thresholded(sine_gaussian(n_coeff, FS, 0.5 * span, FS / 16.0, 0.02 * span)),
        n_coeff, FS, SIGMA)
    long_burst = meta_features(
        *thresholded(sine_gaussian(n_coeff, FS, 0.5 * span, FS / 16.0, 0.15 * span)),
        n_coeff, FS, SIGMA)

    assert long_burst["duration"] >= short_burst["duration"]
    assert long_burst["tSpread"] >= short_burst["tSpread"]


def test_the_centroid_sits_between_two_equal_tiles_where_the_peak_cannot():
    """Why the coincidence time is the centroid and not the loudest tile.

    Two coefficients of the same magnitude at the same scale: no tile dominates,
    the peak has to pick one of them, and a noise fluctuation that reverses the
    ordering moves it by the whole separation. The centroid sits between them
    and does not move at all.
    """
    n_coeff = 512
    level = 5
    first, second = (1 << level) + 4, (1 << level) + 12
    t_lo_first, t_hi_first, _, _ = tile_of(first, n_coeff)
    t_lo_second, t_hi_second, _, _ = tile_of(second, n_coeff)
    middle = 0.25 * (t_lo_first + t_hi_first + t_lo_second + t_hi_second)

    separation = abs(0.5 * (t_lo_second + t_hi_second) - 0.5 * (t_lo_first + t_hi_first))

    index = np.array([first, second])
    ahead = meta_features(index, np.array([1.0, 0.99]), n_coeff, FS, SIGMA)
    behind = meta_features(index, np.array([0.99, 1.0]), n_coeff, FS, SIGMA)

    # A one per cent change of amplitude moves the peak by the whole separation
    # between the two tiles, and the centroid by two orders of magnitude less.
    assert abs(ahead["gpsPeak"] - behind["gpsPeak"]) == pytest.approx(separation)
    assert abs(ahead["gpsCentroid"] - behind["gpsCentroid"]) < 0.02 * separation
    assert ahead["gpsCentroid"] == pytest.approx(middle, abs=0.02 * separation)


def test_the_spread_grows_with_the_extent_of_the_energy():
    n_coeff = 512
    level = 5
    near = np.array([(1 << level) + 7, (1 << level) + 8])
    far = np.array([(1 << level) + 1, (1 << level) + 14])
    value = np.array([1.0, 1.0])

    assert (meta_features(far, value, n_coeff, FS, SIGMA)["tSpread"]
            > meta_features(near, value, n_coeff, FS, SIGMA)["tSpread"])


def test_a_single_tile_is_centred_on_itself_and_spread_over_its_own_width():
    n_coeff = 512
    index = np.array([70])
    t_lo, t_hi, _, _ = tile_of(index[0], n_coeff)
    features = meta_features(index, np.array([3.0]), n_coeff, FS, SIGMA)

    # The energy is somewhere in the tile, not at its centre: uniform over a
    # width w has standard deviation w/sqrt(12).
    assert features["tSpread"] == pytest.approx((t_hi - t_lo) / np.sqrt(12.0))
    assert features["gpsCentroid"] == pytest.approx(0.5 * (t_lo + t_hi))
    assert features["gpsPeak"] == pytest.approx(features["gpsCentroid"])
    assert features["gpsStart"] == pytest.approx(t_lo)
    assert features["duration"] == pytest.approx(t_hi - t_lo)


def test_the_peak_amplitude_is_on_the_noise_scale():
    n_coeff = 512
    index, value = np.array([70, 71]), np.array([3.0, -5.0])
    assert meta_features(index, value, n_coeff, FS, SIGMA)["snrPeak"] == pytest.approx(5.0)
    assert meta_features(index, value, n_coeff, FS, 2.0)["snrPeak"] == pytest.approx(2.5)
    assert meta_features(index, 2 * value, n_coeff, FS, SIGMA)["snrPeak"] == pytest.approx(10.0)


def test_the_times_shift_with_the_window_and_the_frequencies_do_not():
    n_coeff = 512
    index, value = np.array([70, 200]), np.array([3.0, -5.0])
    at_zero = meta_features(index, value, n_coeff, FS, SIGMA)
    later = meta_features(index, value, n_coeff, FS, SIGMA, gps=1.4e9)

    for name in ("gpsStart", "gpsCentroid", "gpsPeak"):
        assert later[name] - at_zero[name] == pytest.approx(1.4e9)
    for name in ("duration", "tSpread", "freqMin", "freqMean", "freqMax", "snrPeak"):
        assert later[name] == pytest.approx(at_zero[name])




def test_the_spread_includes_the_tile_s_own_width():
    """A tile is an interval, not a point: a single surviving coefficient still
    has a spread, set by the width of the tile carrying it."""
    import numpy as np
    from wdf.analysis.metaparameters import meta_features
    from wdf.analysis.wavelets import coeff_time_bounds

    n_coeff, fs = 512, 2048.0
    index = np.array([300])
    features = meta_features(index, np.array([5.0]), n_coeff, fs, sigma=1.0)

    t_lo, t_hi = coeff_time_bounds(n_coeff, fs)
    width = float(t_hi[300] - t_lo[300])
    assert features["tSpread"] == pytest.approx(width / np.sqrt(12.0), rel=1e-9)


def test_a_wider_tile_gives_a_wider_spread_at_the_same_place():
    import numpy as np
    from wdf.analysis.metaparameters import meta_features

    # Level 1 tiles the same instant with a wide tile; level 8 with a narrow one.
    wide = meta_features(np.array([2]), np.array([5.0]), 512, 2048.0, sigma=1.0)
    narrow = meta_features(np.array([300]), np.array([5.0]), 512, 2048.0, sigma=1.0)
    assert wide["tSpread"] > narrow["tSpread"]


def test_one_marginal_tile_stretches_the_support_but_not_the_quantiles():
    """A coefficient carrying almost no energy moves freqMin, freqMax and the
    span arbitrarily far. The quantiles follow the energy instead, which is what
    the band-overlap and timing tests should be reading."""
    import numpy as np
    from wdf.analysis.metaparameters import meta_features

    loud_and_marginal = meta_features(
        np.array([300, 3]), np.array([10.0, 0.05]), 512, 2048.0, sigma=1.0)
    loud_alone = meta_features(np.array([300]), np.array([10.0]), 512, 2048.0, sigma=1.0)

    assert loud_and_marginal["freqMin"] < 0.1 * loud_alone["freqMin"]
    assert loud_and_marginal["duration"] > 100 * loud_and_marginal["duration90"]

    assert loud_and_marginal["freqQ05"] > loud_alone["freqMin"] * 0.5
    assert loud_and_marginal["freqQ95"] <= loud_alone["freqMax"]


def test_the_quantiles_bracket_the_energy_weighted_frequency():
    import numpy as np
    from wdf.analysis.metaparameters import meta_features

    features = meta_features(np.array([70, 140, 300]), np.array([4.0, 6.0, 5.0]),
                             512, 2048.0, sigma=1.0)
    assert features["freqQ05"] <= features["freqMean"] <= features["freqQ95"]
    assert features["freqMin"] <= features["freqQ05"]
    assert features["freqQ95"] <= features["freqMax"]
    assert 0.0 <= features["duration90"] <= features["duration"]


def test_a_single_tile_has_quantiles_inside_its_own_extent():
    import numpy as np
    from wdf.analysis.metaparameters import meta_features
    from wdf.analysis.wavelets import coeff_freq_bands

    features = meta_features(np.array([70]), np.array([3.0]), 512, 2048.0, sigma=1.0)
    f_lo, f_hi = coeff_freq_bands(512, 2048.0)
    assert f_lo[70] <= features["freqQ05"] <= features["freqQ95"] <= f_hi[70]
