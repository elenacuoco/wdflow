"""The paper path's optional screens: the lag of the two reconstructions."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("py4tsa")

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
sys.path.insert(0, FIXTURES)

import make_paper_path_snapshot as snap  # noqa: E402

from wdf.analysis.network_driver import run_network  # noqa: E402
from wdf.analysis.pair_screen import (PairScreenSettings, ScreenedFinder, apply_penalty,  # noqa: E402
                                      lag_factor, lag_windows)
from wdf.analysis.detectors import light_travel_time  # noqa: E402
from wdf.analysis.robust_events import CoincidenceConfig  # noqa: E402
from wdf.config.Parameters import Parameters  # noqa: E402

TIME_COLUMNS = ("gps", "gpsStart", "gpsPeak", "gpsCentroid")


def triggers(offset=0.0):
    """The snapshot's triggers (the chirp 5 ms later in L1), L1 moved by
    `offset` seconds."""
    out = {ifo: pd.read_parquet(os.path.join(FIXTURES, f"paper_path_triggers_{ifo}.parquet"))
           for ifo in snap.DELAY}
    for column in TIME_COLUMNS:
        out["L1"][column] = out["L1"][column] + offset
    return out


def on_chirp(table):
    return table[(table["gps_candidate"] - snap.T0 - snap.CHIRP_AT).between(-1, snap.CHIRP_S + 1)]


def test_by_default_nothing_is_screened():
    plain = run_network(Parameters(), triggers=triggers()).candidates
    assert "network_lag_s" not in plain


def test_a_coincident_pair_is_kept_at_its_lag():
    cand = run_network(Parameters(NetworkLagCut="cut"), triggers=triggers()).candidates
    row = on_chirp(cand).iloc[0]
    planted = snap.DELAY["H1"] - snap.DELAY["L1"]
    assert np.sign(row["network_lag_s"]) == np.sign(planted)
    assert abs(row["network_lag_s"] - planted) <= max(4e-3, 2 * row["network_lag_width_s"])
    assert abs(row["network_lag_s"]) <= row["network_lag_window_s"]


def test_a_pair_half_a_second_apart_is_cut_or_penalised():
    par = Parameters(NetworkLagCut="cut")
    assert on_chirp(run_network(par, triggers=triggers(0.5)).candidates).empty
    plain = on_chirp(run_network(Parameters(), triggers=triggers(0.5)).candidates)
    assert len(plain)
    pen = Parameters(NetworkLagCut="penalty")
    cand = on_chirp(run_network(pen, triggers=triggers(0.5)).candidates)
    row = cand.iloc[0]
    expected = lag_factor(row["network_lag_s"], row["network_lag_window_s"], row["network_lag_window_s"])
    # the scale is the pair's own window (NetworkLagScale "window"), ~25 ms for H1-L1
    assert row["network_lag_factor"] == pytest.approx(float(expected)) and expected < 0.05
    assert row["network_rank_statistic"] == pytest.approx(
        row["network_morphology"] * row["network_lag_factor"])
    near = on_chirp(run_network(pen, triggers=triggers()).candidates).iloc[0]
    assert near["network_lag_factor"] == 1.0


def test_the_lag_factor_is_symmetric_and_monotone():
    lags = np.array([-0.3, -0.05, -0.01, 0.0, 0.01, 0.05, 0.3, np.nan])
    f = lag_factor(lags, 0.02, 0.01)
    np.testing.assert_allclose(f[:3], f[[6, 5, 4]])
    assert f[3] == f[2] == 1.0 and f[1] < 1.0 and f[0] < f[1] and f[-1] == 0.0


def test_the_screen_runs_on_time_slides():
    par = Parameters(NetworkLagCut="penalty", NetworkSlides=5, NetworkSlideStep=4.0)
    result = run_network(par, triggers=triggers())
    assert "far_hz" in result.candidates
    if len(result.background):
        assert "network_lag_factor" in result.background
        assert result.background["network_lag_s"].notna().any()


def test_an_unknown_criterion_is_refused():
    with pytest.raises(ValueError, match="NetworkLagCut"):
        run_network(Parameters(NetworkLagCut="gate"), triggers=triggers())


# ================================================================ glitch decision
CHIRP = (snap.T0 + snap.CHIRP_AT, snap.T0 + snap.CHIRP_AT + snap.CHIRP_S)


def census(*spans):
    return pd.DataFrame(dict(extent_first=[a for a, _ in spans], extent_last=[b for _, b in spans],
                             peak=[30.0] * len(spans)))


def test_a_coincident_loud_transient_is_not_a_glitch():
    """Both detectors' census hold the chirp (5 ms apart): no glitch."""
    both = {"H1": census(CHIRP), "L1": census((CHIRP[0] + 0.005, CHIRP[1] + 0.005))}
    row = on_chirp(run_network(Parameters(), triggers=triggers(), census=both).candidates).iloc[0]
    assert not row["glitch_single_ifo"] and not row["glitch_i"] and not row["glitch_j"]


def test_a_single_detector_transient_is_flagged_and_vetoed_or_penalised():
    """Only H1's census holds the transient; L1 sees noise there: the H1
    event is a single-detector glitch."""
    one = {"H1": census(CHIRP), "L1": census((CHIRP[0] - 30.0, CHIRP[0] - 29.9))}
    row = on_chirp(run_network(Parameters(), triggers=triggers(), census=one).candidates).iloc[0]
    assert row["glitch_single_ifo"] and row["glitch_i"] and not row["glitch_j"]
    vetoed = run_network(Parameters(NetworkGlitchVeto="veto"), triggers=triggers(), census=one)
    assert on_chirp(vetoed.candidates).empty
    pen = run_network(Parameters(NetworkGlitchVeto="penalty"), triggers=triggers(), census=one)
    row = on_chirp(pen.candidates).iloc[0]
    assert row["network_rank_statistic"] == pytest.approx(0.1 * row["network_morphology"])


def test_the_glitch_decision_needs_the_census_and_runs_in_slides():
    with pytest.raises(ValueError, match="census"):
        run_network(Parameters(NetworkGlitchVeto="veto"), triggers=triggers())
    both = {"H1": census(CHIRP), "L1": census((CHIRP[0] + 0.005, CHIRP[1] + 0.005))}
    par = Parameters(NetworkGlitchVeto="penalty", NetworkSlides=5, NetworkSlideStep=4.0)
    result = run_network(par, triggers=triggers(), census=both)
    if len(result.background):
        assert "glitch_single_ifo" in result.background


PAIRS = (("H1", "L1"), ("H1", "V1"), ("L1", "V1"))


def test_the_coincidence_window_is_each_pairs_own_tolerance():
    config = CoincidenceConfig()
    a, b = zip(*PAIRS)
    got = lag_windows(a, b, [0.0] * 3, [0.0] * 3, PairScreenSettings())
    for (p, q), w in zip(PAIRS, got):
        travel = light_travel_time(p, q)
        floor = config.timing_sigma * np.hypot(config.timing_jitter_s, config.timing_jitter_s)
        assert w == pytest.approx(min(travel + floor, config.maximum_tolerance_travel_times * travel))
        assert w == pytest.approx(float(config.timing_tolerance(0.0, 0.0, (p, q))))
    assert got[1] > 2.5 * got[0] / 1.5 and got[0] < got[2] < got[1]


def test_the_coincidence_window_is_capped_and_reads_the_spreads():
    config = CoincidenceConfig()
    a, b = zip(*PAIRS)
    wide = lag_windows(a, b, [0.5] * 3, [0.5] * 3, PairScreenSettings())
    for (p, q), w in zip(PAIRS, wide):
        assert w == pytest.approx(config.maximum_tolerance_travel_times * light_travel_time(p, q))
    narrow = lag_windows(["H1"], ["L1"], [0.004], [np.nan], PairScreenSettings())
    travel = light_travel_time("H1", "L1")
    assert narrow[0] == pytest.approx(min(travel + 3 * np.hypot(0.004, 0.0035), 2.5 * travel))


def test_travel_plus_tolerance_is_light_travel_plus_the_tolerance():
    settings = PairScreenSettings(lag_window="travel_plus_tolerance", lag_tolerance=0.0)
    a, b = zip(*PAIRS)
    got = lag_windows(a, b, [0.0] * 3, [0.0] * 3, settings)
    assert got == pytest.approx([light_travel_time(p, q) for p, q in PAIRS])


def test_the_window_choice_is_read_and_checked():
    par = Parameters(NetworkLagCut="cut")
    assert PairScreenSettings.from_parameters(par).lag_window == "coincidence"
    assert PairScreenSettings.from_parameters(Parameters(NetworkLagScale="window")).lag_scale == "window"
    assert PairScreenSettings.from_parameters(Parameters(NetworkLagScale=0.02)).lag_scale == 0.02
    with pytest.raises(ValueError):
        PairScreenSettings.from_parameters(Parameters(NetworkLagWindow="sky"))


def test_a_pair_reports_its_coincidence_window():
    cand = run_network(Parameters(NetworkLagCut="cut"), triggers=triggers()).candidates
    row = on_chirp(cand).iloc[0]
    travel = light_travel_time("H1", "L1")
    assert travel < row["network_lag_window_s"] <= 2.5 * travel + 1e-12
    assert str(row["network_lag_window_rule"]).startswith("coincidence")


# ================================================================ penalty space
def test_a_penalty_multiplies_a_statistic_and_lowers_a_logit():
    factor = np.array([1.0, 0.5, 0.1, 0.0])
    np.testing.assert_allclose(apply_penalty([2.0, 2.0, 2.0, 2.0], factor, "multiply"), [2.0, 1.0, 0.2, 0.0])
    logit = np.array([-2.0, -2.0, 1.5, 1.5])
    out = apply_penalty(logit, factor, "log-odds")
    assert out[0] == -2.0 and np.isneginf(out[3])
    assert np.all(out[1:3] < logit[1:3])
    np.testing.assert_allclose(out[1:3], logit[1:3] + np.log(factor[1:3]))
    with pytest.raises(ValueError):
        apply_penalty(logit, factor, "add")


def test_the_penalty_space_follows_the_score():
    auto = PairScreenSettings()
    assert auto.space(signed=False) == "multiply" and auto.space(signed=True) == "log-odds"
    with pytest.raises(ValueError, match="signed logit"):
        PairScreenSettings(penalty_space="multiply").space(signed=True)
    assert PairScreenSettings(penalty_space="log-odds").space(signed=False) == "log-odds"
    with pytest.raises(ValueError, match="NetworkPenaltySpace"):
        PairScreenSettings.from_parameters(Parameters(NetworkPenaltySpace="product"))
    par = Parameters()
    assert PairScreenSettings.from_parameters(par).penalty_space == "auto"
    assert par.NetworkPenaltySpace == "auto"


class _Pairs:
    """A finder that returns fixed pairs, each with a GNN logit."""

    config = None

    def __init__(self, table):
        self.table = table

    def find(self, events_by_ifo):
        return self.table.copy()


def _glitch_screen(logits, signed, space="auto"):
    """Two events per detector; H1's first overlaps a census transient that
    L1 lacks, so every pair holding it is a single-detector glitch."""
    events = {ifo: pd.DataFrame(dict(cluster_id=[0, 1], gpsPeak=[100.0 + d, 200.0 + d],
                                     gpsStart=[99.9 + d, 199.9 + d], duration=[0.2, 0.2]))
              for ifo, d in (("H1", 0.0), ("L1", 0.005))}
    table = pd.DataFrame(dict(node_i=[0, 1], node_j=[2, 3], gnn_logit=logits,
                              network_morphology=np.abs(logits)))
    census = {"H1": pd.DataFrame(dict(first=[99.95], last=[100.05])),
              "L1": pd.DataFrame(dict(first=[400.0], last=[400.1]))}
    settings = PairScreenSettings(glitch_veto="penalty", glitch_factor=0.1, penalty_space=space)
    column = "gnn_logit" if signed else "network_morphology"
    finder = ScreenedFinder(_Pairs(table), events, {}, 2048.0, settings, column, census=census,
                            signed=signed)
    return finder.find(events)


def test_a_glitch_penalty_lowers_a_negative_gnn_logit():
    out = _glitch_screen([-2.0, -1.0], signed=True)
    assert list(out["glitch_single_ifo"]) == [True, False]
    assert out["network_rank_statistic"].iloc[0] == pytest.approx(-2.0 + np.log(0.1))
    assert out["network_rank_statistic"].iloc[0] < out["network_rank_statistic"].iloc[1] == -1.0


def test_a_glitch_penalty_on_a_statistic_is_unchanged():
    out = _glitch_screen([-2.0, -1.0], signed=False)
    assert out["network_rank_statistic"].tolist() == pytest.approx([0.2, 1.0])
    with pytest.raises(ValueError, match="signed logit"):
        _glitch_screen([-2.0, -1.0], signed=True, space="multiply")
