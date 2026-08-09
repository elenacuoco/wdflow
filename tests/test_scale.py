import numpy as np
import pandas as pd
import pytest

from wdf.analysis.scale import (
    ScaleCalibration,
    normalised_energy,
    pixel_cloud,
    scale_maximum,
)

FS = 2048.0


def _triggers(scale, n, rng, amplitude=1.0, gps0=1000.0, n_nonzero=4):
    """`n` triggers at one window length, on that window length's own grid."""
    stride = scale / FS
    rows = []
    for i in range(n):
        index = np.sort(rng.choice(np.arange(1, scale), size=n_nonzero,
                                   replace=False))
        rows.append(dict(
            gps=gps0 + i * stride, sigma=1.0, n_coeff=int(scale), fs=FS,
            wt_index=index.astype(np.uint16),
            wt_value=(amplitude * rng.normal(size=n_nonzero)).astype(np.float32),
            ifo="H1",
        ))
    return pd.DataFrame(rows)


def _cloud(scales, n, seed=0, amplitude=1.0):
    rng = np.random.default_rng(seed)
    return pixel_cloud(pd.concat(
        [_triggers(scale, n, rng, amplitude) for scale in scales],
        ignore_index=True))


def test_the_cloud_carries_the_window_length_as_its_own_coordinate():
    cloud = _cloud([256, 512, 1024], 20)
    assert set(cloud["scale"].unique()) == {256, 512, 1024}
    assert (cloud["t_hi"] > cloud["t_lo"]).all()
    assert (cloud["f_hi"] > cloud["f_lo"]).all()


def test_a_longer_window_reaches_below_a_shorter_one():
    """The ladder extends downward rather than subdividing."""
    cloud = _cloud([256, 1024], 60)
    short = cloud[cloud.scale == 256]
    long = cloud[cloud.scale == 1024]
    assert long["f_lo"].min() < short["f_lo"].min()
    shared = set(short["f_lo"]) & set(long["f_lo"])
    assert shared, "window lengths must share the bands both of them reach"


def test_the_significance_is_calibrated_per_window_length_and_band():
    """The same energy at two window lengths need not be the same significance,
    and the calibration is what makes the two comparable."""
    background = _cloud([256, 1024], 400, seed=1)
    calibration = ScaleCalibration.fit(background)
    assert len(calibration.tables) > 1

    significance = calibration.significance(background)
    finite = np.isfinite(significance)
    assert finite.any()
    assert (significance[finite] >= 0.0).all()


def test_a_louder_tile_is_never_less_significant_within_its_band():
    background = _cloud([512], 500, seed=2)
    calibration = ScaleCalibration.fit(background)
    probe = background.copy()
    significance = calibration.significance(probe)
    energy = normalised_energy(probe)
    for band, group in pd.DataFrame(
            dict(band=probe["f_lo"], e=energy, s=significance)).dropna().groupby("band"):
        order = np.argsort(group["e"].to_numpy())
        ranked = group["s"].to_numpy()[order]
        assert np.all(np.diff(ranked) >= -1e-12)


def test_a_band_the_background_never_produced_gets_no_significance():
    background = _cloud([512], 200, seed=3)
    calibration = ScaleCalibration.fit(background)
    foreign = background.head(3).copy()
    foreign["f_lo"] = -1.0
    assert np.isnan(calibration.significance(foreign)).all()


def test_fitting_on_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing to measure"):
        ScaleCalibration.fit(pd.DataFrame(columns=["scale", "f_lo", "energy", "sigma"]))


def test_the_cross_scale_maximum_is_at_least_the_tile_s_own_significance():
    background = _cloud([256, 512, 1024], 300, seed=4)
    calibration = ScaleCalibration.fit(background)
    scored = scale_maximum(background, calibration)
    own = scored["significance"].to_numpy()
    finite = np.isfinite(own)
    assert (scored["s_max"].to_numpy()[finite] >= own[finite] - 1e-12).all()


def test_the_scale_signature_counts_the_window_lengths_that_saw_a_tile():
    background = _cloud([256, 512, 1024], 300, seed=5)
    scored = scale_maximum(background, ScaleCalibration.fit(background))
    assert scored["n_scales"].min() >= 1
    assert scored["n_scales"].max() <= 3
    assert (scored["s_ratio"] <= 1.0 + 1e-12).all()


def test_one_window_length_leaves_the_statistic_untouched():
    """With a single window length there is no maximum to take, and the
    cross-scale statistic must not differ from the single-scale one."""
    background = _cloud([512], 400, seed=6)
    calibration = ScaleCalibration.fit(background)
    scored = scale_maximum(background, calibration)
    own = scored["significance"].to_numpy()
    finite = np.isfinite(own)
    assert np.allclose(scored["s_max"].to_numpy()[finite], own[finite])
    assert (scored["n_scales"] == 1).all()


def test_an_empty_cloud_survives_every_stage():
    empty = pixel_cloud(pd.DataFrame(columns=["gps", "n_coeff", "fs", "sigma"]))
    assert empty.empty
    scored = scale_maximum(empty, ScaleCalibration(tables={"x": np.zeros(1)}))
    assert scored.empty
