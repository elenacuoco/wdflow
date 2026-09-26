"""When two tiles may belong to the same transient, and on whose geometry.

The admission may rest on the analysis --- the stride the search advanced by
and the dyadic ladder it tiled frequency with --- and never on the shape of the
source. These tests state that, and state that the default configuration is the
rule that was there before.
"""
import numpy as np
import pandas as pd
import pytest

from wdf.analysis.pixel_graph import (PixelGraphConfig, build_pixel_graph,
                                      cloud_strides, cluster_events)

STRIDE = (512 - 32) / 2048.0


def cloud(**overrides):
    """Two detections of one transient, a stride apart, in adjacent bands."""
    frame = pd.DataFrame(dict(
        trigger_index=[0, 1],
        stride=[STRIDE] * 2,
        ifo=["H1"] * 2,
        scale=[512.0] * 2,
        fs=[2048.0] * 2,
        t_lo=[0.0, STRIDE],
        t_hi=[0.03125, STRIDE + 0.03125],
        f_lo=[64.0, 128.0],
        f_hi=[128.0, 256.0],
        energy=[9.0, 4.0],
        sigma=[1.0, 1.0]))
    return frame.assign(**overrides)


def legacy_edges(pixels, config):
    """The rule as it stood, written out, to compare the new one against."""
    graph = build_pixel_graph(pixels, config=config)
    nodes = graph.nodes
    t_lo = nodes["t_lo"].to_numpy(dtype=float)
    t_hi = nodes["t_hi"].to_numpy(dtype=float)
    f_lo = nodes["f_lo"].to_numpy(dtype=float)
    f_hi = nodes["f_hi"].to_numpy(dtype=float)
    width = t_hi - t_lo
    out = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            gap = max(t_lo[j] - t_hi[i], t_lo[i] - t_hi[j])
            allowed = config.time_tolerance * 0.5 * (width[i] + width[j])
            band = (f_lo[i] <= f_hi[j]) and (f_lo[j] <= f_hi[i])
            if gap <= allowed and band:
                out.append((i, j))
    return sorted(out)


def messy_cloud(n=200, seed=3):
    """A cloud of several window lengths, bands and overlaps."""
    rng = np.random.default_rng(seed)
    scale = rng.choice([256.0, 512.0], size=n)
    level = rng.integers(0, 8, size=n)
    f_lo = np.where(level == 0, 0.0, 1024.0 / 2.0 ** (8 - level))
    f_hi = 1024.0 / 2.0 ** (7 - level)
    t_lo = np.round(rng.uniform(0.0, 3.0, size=n), 4)
    return pd.DataFrame(dict(
        trigger_index=rng.integers(0, 12, size=n), ifo="H1", scale=scale,
        stride=STRIDE,
        fs=2048.0, t_lo=t_lo, t_hi=t_lo + 512.0 / scale * 0.03,
        f_lo=f_lo, f_hi=f_hi, energy=rng.uniform(1.0, 50.0, size=n),
        sigma=1.0))


@pytest.mark.parametrize("tolerance", [0.0, 1.0, 4.0])
def test_the_defaults_give_the_edges_the_old_rule_gave(tolerance):
    pixels = messy_cloud()
    config = PixelGraphConfig(time_tolerance=tolerance)
    graph = build_pixel_graph(pixels, config=config)
    found = sorted(map(tuple, np.sort(graph.edges, axis=1).tolist()))
    assert found == legacy_edges(pixels, config)
    assert config.stride_tolerance == 0.0
    assert config.band_tolerance == 0


def test_the_defaults_give_the_events_the_old_rule_gave():
    pixels = messy_cloud()
    plain = cluster_events(build_pixel_graph(pixels))
    explicit = cluster_events(build_pixel_graph(
        pixels, config=PixelGraphConfig(stride_tolerance=0.0,
                                        band_tolerance=0)))
    pd.testing.assert_frame_equal(plain, explicit)


def test_a_width_based_tolerance_cannot_reach_across_a_stride():
    """The measured fact the stride allowance exists for.

    The tiles are 4 samples wide, 0.002 s at 2048 Hz, while the stride is
    0.234 s. No multiple of the width the rule allows closes that.
    """
    pixels = cloud(f_lo=[64.0, 64.0], f_hi=[128.0, 128.0],
                   t_hi=[4 / 2048.0, STRIDE + 4 / 2048.0])
    for tolerance in (1.0, 2.0, 4.0, 8.0, 16.0):
        graph = build_pixel_graph(
            pixels, config=PixelGraphConfig(time_tolerance=tolerance))
        assert len(graph.edges) == 0


def test_a_stride_allowance_joins_two_tiles_one_stride_apart():
    pixels = cloud(f_lo=[64.0, 64.0], f_hi=[128.0, 128.0],
                   t_hi=[4 / 2048.0, STRIDE + 4 / 2048.0])
    graph = build_pixel_graph(
        pixels, config=PixelGraphConfig(stride_tolerance=1.0))
    assert len(graph.edges) == 1
    assert len(cluster_events(graph)) == 1


def test_the_stride_is_read_from_the_cloud_and_not_given_by_hand():
    """`pixel_cloud` carries the run's declared stride onto every tile."""
    from wdf.analysis.scale import pixel_cloud

    triggers = pd.DataFrame(dict(
        gps=[0.0, STRIDE], ifo="H1", n_coeff=512, fs=2048.0, sigma=1.0,
        stride=STRIDE, wt_index=[[1], [1]], wt_value=[[3.0], [2.0]]))
    pixels = pixel_cloud(triggers)
    assert cloud_strides(pixels) == {512.0: pytest.approx(STRIDE)}
    assert cloud_strides(cloud()) == {512.0: pytest.approx(STRIDE)}


def test_an_unknown_stride_leaves_the_width_rule_alone():
    """A cloud that declares no stride grants no allowance in strides."""
    pixels = cloud(f_lo=[64.0, 64.0], f_hi=[128.0, 128.0],
                   t_hi=[4 / 2048.0, STRIDE + 4 / 2048.0])
    silent = pixels.drop(columns=["stride"])
    assert cloud_strides(silent) == {}
    graph = build_pixel_graph(
        silent, config=PixelGraphConfig(stride_tolerance=1.0))
    assert len(graph.edges) == 0


def test_a_band_allowance_of_one_crosses_one_empty_step_and_zero_does_not():
    # 64-128 and 256-512, with the empty band 128-256 between them.
    pixels = cloud(t_lo=[0.0, 0.0], t_hi=[0.03125] * 2,
                   f_lo=[64.0, 256.0], f_hi=[128.0, 512.0])
    # The ladder has to hold the empty step for it to be countable.
    pixels = pd.concat([pixels, pixels.iloc[[0]].assign(
        trigger_index=2, t_lo=2.0, t_hi=2.03125, f_lo=128.0, f_hi=256.0)],
        ignore_index=True)
    closed = build_pixel_graph(pixels, config=PixelGraphConfig(band_tolerance=0))
    open_one = build_pixel_graph(pixels, config=PixelGraphConfig(band_tolerance=1))
    assert len(closed.edges) == 0
    assert len(open_one.edges) == 1


def test_the_band_allowance_is_symmetric_in_direction():
    """A jump upward and the same jump downward are admitted alike."""
    def edges_of(f_lo, f_hi):
        pixels = cloud(t_lo=[0.0, STRIDE], t_hi=[0.03125, STRIDE + 0.03125],
                       f_lo=f_lo, f_hi=f_hi)
        config = PixelGraphConfig(stride_tolerance=1.0, band_tolerance=1)
        return len(build_pixel_graph(pixels, config=config).edges)

    assert edges_of([64.0, 256.0], [128.0, 512.0]) == 1
    assert edges_of([256.0, 64.0], [512.0, 128.0]) == 1


def test_a_burst_that_jumps_and_comes_back_is_admitted_as_a_sweep_is():
    """No monotonicity in the admission: think of a supernova."""
    def n_events(bands):
        f_lo = np.array(bands, dtype=float)
        pixels = pd.DataFrame(dict(
            trigger_index=np.arange(len(f_lo)), ifo="H1", scale=512.0,
            fs=2048.0, stride=STRIDE, t_lo=np.arange(len(f_lo)) * STRIDE,
            t_hi=np.arange(len(f_lo)) * STRIDE + 0.03125,
            f_lo=f_lo, f_hi=f_lo * 2.0, energy=9.0, sigma=1.0))
        config = PixelGraphConfig(stride_tolerance=1.0, band_tolerance=1)
        return len(cluster_events(build_pixel_graph(pixels, config=config)))

    assert n_events([64.0, 128.0, 256.0, 512.0]) == 1
    assert n_events([64.0, 256.0, 512.0, 128.0]) == 1
    assert n_events([512.0, 256.0, 128.0, 64.0]) == 1


def test_the_edge_features_keep_their_meaning_and_their_number():
    from wdf.analysis.pixel_graph import PIXEL_EDGE_FEATURES

    pixels = cloud(f_lo=[64.0, 64.0], f_hi=[128.0, 128.0])
    graph = build_pixel_graph(
        pixels, config=PixelGraphConfig(stride_tolerance=1.0,
                                        band_tolerance=1))
    assert graph.edge_features.shape == (1, len(PIXEL_EDGE_FEATURES))
    table = graph.edge_table()
    assert list(table.columns) == ["node_i", "node_j"] + PIXEL_EDGE_FEATURES
    assert table.time_gap[0] == pytest.approx(STRIDE - 0.03125)
    assert table.frequency_overlap[0] == pytest.approx(1.0)
