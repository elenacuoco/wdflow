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
from wdf.analysis.pair_screen import lag_factor  # noqa: E402
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
    expected = lag_factor(row["network_lag_s"], row["network_lag_window_s"], 0.01)
    assert row["network_lag_factor"] == pytest.approx(float(expected)) and expected < 0.01
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
