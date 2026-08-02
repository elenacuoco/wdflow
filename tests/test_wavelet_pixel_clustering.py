import numpy as np
import pandas as pd
import pytest

from wdf.analysis.clustering import WaveletPixelClusterer, collect_significant_pixels
from wdf.analysis.wavelets import coeff_freq_bands, coeff_time_bounds, donoho_johnstone_threshold

NCOEFF = 16
FS = 16.0  # -> window = NCOEFF/FS = 1.0s, J = log2(16) = 4 levels
SIGMA = 1.0  # DJ threshold at N=16 -> sigma*sqrt(2*ln16) ~= 2.35: well below the "loud"
             # magnitude (10.0) and well above the "noise" magnitude (1e-6) used below.


def _trigger_row(gps, wt_index, magnitude, ifo="H1", ncoeff=NCOEFF):
    wt = np.full(ncoeff, 1e-6)  # near-zero "noise" everywhere else
    wt[wt_index] = magnitude
    row = {f"wt{i}": wt[i] for i in range(ncoeff)}
    row["gps"] = gps
    row["ifo"] = ifo
    return row


def test_collect_significant_pixels_keeps_only_above_dj_threshold():
    df = pd.DataFrame([_trigger_row(0.0, wt_index=8, magnitude=10.0)])
    pixels = collect_significant_pixels(df, FS, sigma=SIGMA)
    thresh = donoho_johnstone_threshold(SIGMA, NCOEFF)
    assert thresh < 10.0 and thresh > 1e-6  # sanity on the test's own setup
    # only the one loud coefficient should survive; the near-zero "noise" ones shouldn't
    assert len(pixels) == 1
    assert pixels["energy"].iloc[0] == pytest.approx(100.0)


def test_overlapping_pixels_from_different_triggers_cluster_together():
    # index 8 is level-3 (finest), t=[0,0.125), f=[4,8) at gps=0 -- see wdf.analysis.wavelets
    df = pd.DataFrame([
        _trigger_row(0.00, wt_index=8, magnitude=10.0),
        _trigger_row(0.05, wt_index=8, magnitude=10.0),  # tile t=[0.05,0.175) overlaps the first
    ])
    clusterer = WaveletPixelClusterer(time_tol_s=0.0, sigma=SIGMA)
    pixels = clusterer.fit(df, FS)
    assert len(pixels) == 2
    assert pixels["cluster_id"].nunique() == 1


def test_distant_pixels_do_not_cluster():
    df = pd.DataFrame([
        _trigger_row(0.0, wt_index=8, magnitude=10.0),
        _trigger_row(10.0, wt_index=8, magnitude=10.0),  # far away in time
    ])
    clusterer = WaveletPixelClusterer(time_tol_s=0.05, sigma=SIGMA)
    pixels = clusterer.fit(df, FS)
    assert pixels["cluster_id"].nunique() == 2


def test_different_frequency_bands_do_not_cluster_even_if_close_in_time():
    # index 8 -> level 3 band [4,8); index 1 -> level 0 band [0.5,1) -- disjoint
    df = pd.DataFrame([
        _trigger_row(0.0, wt_index=8, magnitude=10.0),
        _trigger_row(0.0, wt_index=1, magnitude=10.0),
    ])
    clusterer = WaveletPixelClusterer(time_tol_s=1.0, sigma=SIGMA)
    pixels = clusterer.fit(df, FS)
    assert pixels["cluster_id"].nunique() == 2


def test_time_tol_s_bridges_a_gap():
    # index 8 at gps=0 -> t=[0,0.125); index 8 at gps=0.2 -> t=[0.2,0.325). Gap = 0.075s.
    df = pd.DataFrame([
        _trigger_row(0.0, wt_index=8, magnitude=10.0),
        _trigger_row(0.2, wt_index=8, magnitude=10.0),
    ])
    too_tight = WaveletPixelClusterer(time_tol_s=0.01, sigma=SIGMA)
    assert too_tight.fit(df, FS)["cluster_id"].nunique() == 2

    loose_enough = WaveletPixelClusterer(time_tol_s=0.10, sigma=SIGMA)
    assert loose_enough.fit(df, FS)["cluster_id"].nunique() == 1


def test_clustered_events_aggregates_energy_and_span():
    df = pd.DataFrame([
        _trigger_row(0.00, wt_index=8, magnitude=3.0),
        _trigger_row(0.05, wt_index=8, magnitude=4.0),
    ])
    clusterer = WaveletPixelClusterer(time_tol_s=0.0, sigma=SIGMA)
    pixels = clusterer.fit(df, FS)
    events = clusterer.clustered_events(pixels)

    assert len(events) == 1
    row = events.iloc[0]
    assert row["n_pixels"] == 2
    assert row["n_triggers"] == 2
    assert row["total_energy"] == pytest.approx(3.0 ** 2 + 4.0 ** 2)
    assert row["gpsStart"] == pytest.approx(0.00)
    assert row["gpsEnd"] == pytest.approx(0.05 + 0.125)
    assert row["ifos"] == ["H1"]


def test_no_wt_columns_gives_empty_pixels():
    df = pd.DataFrame([{"gps": 0.0, "ifo": "H1", "snrPeak": 1.0}])
    pixels = collect_significant_pixels(df, FS, sigma=SIGMA)
    assert pixels.empty


def test_tile_geometry_sanity_used_in_this_test_file():
    # confirms the hand-derived t/f bounds for wt_index=8 used above are correct,
    # so the tests above aren't silently relying on wrong assumptions.
    t_lo, t_hi = coeff_time_bounds(NCOEFF, FS)
    f_lo, f_hi = coeff_freq_bands(NCOEFF, FS)
    assert t_lo[8] == pytest.approx(0.0) and t_hi[8] == pytest.approx(0.125)
    assert f_lo[8] == pytest.approx(4.0) and f_hi[8] == pytest.approx(8.0)
    assert f_lo[1] == pytest.approx(0.5) and f_hi[1] == pytest.approx(1.0)
