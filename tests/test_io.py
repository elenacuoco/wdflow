"""Loading trigger files."""
import numpy as np
import pandas as pd

from wdf.analysis.io import TRIGGER_COLUMNS, triggers_from_csvs


def _trigger_frame(n=20, n_coeff=64):
    frame = pd.DataFrame({c: np.zeros(n) for c in TRIGGER_COLUMNS if c != "wave"})
    frame["wave"] = "DaubC12"
    rng = np.random.default_rng(0)
    for i in range(n_coeff):
        frame[f"wt{i}"] = rng.standard_normal(n)
        frame[f"rw{i}"] = rng.standard_normal(n)
    return frame


def test_coefficient_columns_are_loaded_in_single_precision(tmp_path):
    """One column per coefficient dominates a trigger frame -- 512 of them at a
    typical window length -- and they carry seven significant digits at most,
    so double precision doubles the frame for nothing."""
    path = tmp_path / "triggers.parquet"
    _trigger_frame().to_parquet(path)

    loaded = triggers_from_csvs([str(path)], "H1")

    assert loaded["wt0"].dtype == np.float32
    assert loaded["rw0"].dtype == np.float32
    assert loaded["gps"].dtype == np.float64     # the metadata keeps its precision
    assert (loaded["ifo"] == "H1").all()


def test_a_frame_without_coefficients_loads_unchanged(tmp_path):
    """`fullPrint = 0` writes no wt* columns at all."""
    frame = pd.DataFrame({c: np.zeros(5) for c in TRIGGER_COLUMNS if c != "wave"})
    frame["wave"] = "Haar"
    path = tmp_path / "triggers.parquet"
    frame.to_parquet(path)

    loaded = triggers_from_csvs([str(path)], "L1")
    assert len(loaded) == 5
    assert not [c for c in loaded.columns if c[:2] == "wt" and c[2:].isdigit()]


def test_missing_columns_are_named(tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"gps": [1.0]}).to_parquet(path)

    try:
        triggers_from_csvs([str(path)], "H1")
    except ValueError as error:
        assert "missing expected columns" in str(error)
    else:
        raise AssertionError("a frame missing the trigger columns should be refused")
