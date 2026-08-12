"""Loading trigger files."""
import numpy as np
import pandas as pd
import pytest

from wdf.analysis.coefficients import TriggerWriter
from wdf.analysis.io import (
    TRIGGER_COLUMNS,
    add_wavelet_energy_diagnostics,
    analysed_span,
    covered_livetime_days,
    triggers_from_files,
)
from wdf.analysis.metaparameters import meta_features

N_COEFF = 64
FS = 2048.0


OVERLAP = N_COEFF // 4


def write_triggers(path, n=20, seed=0, n_coeff=N_COEFF):
    """A trigger file as the pipeline writes it: named for its window length,
    with the run configuration beside it. The stride the search advanced by is
    in that configuration and nowhere else."""
    import json

    path = path.with_name(path.name.replace(".parquet", f"-Win{n_coeff}-.parquet"))
    json.dump(dict(window=n_coeff, overlap=OVERLAP, sampling=2.0 * FS,
                   ResamplingFactor=2),
              open(path.parent / f"parametersUsed-Win{n_coeff}.json", "w"))

    rng = np.random.default_rng(seed)
    with TriggerWriter(str(path)) as writer:
        for k in range(n):
            index = np.sort(rng.choice(n_coeff, size=3, replace=False))
            value = rng.normal(size=3)
            gps = 1000.0 + k
            writer.append(dict(
                meta_features(index, value, n_coeff, FS, 1.0, gps=gps),
                gps=gps, EnWDF=float(np.linalg.norm(value)), sigma=1.0,
                wave="DaubC12", n_coeff=n_coeff, fs=FS,
                wt_index=[int(i) for i in index],
                wt_value=[float(v) for v in value]))
    return path


def test_a_detectors_files_load_into_one_frame(tmp_path):
    first = write_triggers(tmp_path / "a.parquet", n=7, seed=0)
    second = write_triggers(tmp_path / "b.parquet", n=5, seed=1)

    loaded = triggers_from_files([str(first), str(second)], "H1")

    assert len(loaded) == 12
    assert set(TRIGGER_COLUMNS) <= set(loaded.columns)
    assert (loaded["ifo"] == "H1").all()
    assert loaded["gps"].dtype == np.float64


def test_no_paths_is_refused():
    with pytest.raises(ValueError, match="no trigger file paths"):
        triggers_from_files([], "H1")


def test_missing_columns_are_named(tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"gps": [1.0], "wt_index": [[0]], "wt_value": [[1.0]]}).to_parquet(path)

    with pytest.raises(ValueError, match="missing expected columns"):
        triggers_from_files([str(path)], "H1")


def test_the_statistic_can_be_recomputed_from_the_coefficients(tmp_path):
    """EnWDF is the coefficient norm on the noise scale, so the two agree."""
    path = write_triggers(tmp_path / "t.parquet", n=10, seed=2)
    loaded = add_wavelet_energy_diagnostics(triggers_from_files([str(path)], "H1"))

    assert loaded["EnWDF_from_coeff"].to_numpy() == pytest.approx(
        loaded["EnWDF"].to_numpy(), rel=1e-6)
    assert loaded["EnWDF_residual"].abs().max() < 1e-6
    assert (loaded["nActiveCoeff"] == 3).all()


def test_analysed_span_is_what_every_detector_searched():
    """The livetime that divides a coincidence rate is the stretch both
    detectors covered: a coincidence cannot be formed where one was not looking.
    """
    a = pd.DataFrame({"gpsStart": [100.0, 500.0, 900.0]})
    b = pd.DataFrame({"gpsStart": [150.0, 800.0]})
    assert analysed_span([a, b]) == (150.0, 800.0)


def test_analysed_span_is_empty_when_the_detectors_never_overlap():
    a = pd.DataFrame({"gpsStart": [0.0, 10.0]})
    b = pd.DataFrame({"gpsStart": [100.0, 110.0]})
    first, last = analysed_span([a, b])
    assert last == first
    assert covered_livetime_days([(first, last)]) == 0.0


def test_a_detector_that_wrote_nothing_is_an_error_not_a_livetime():
    """Treating it as covered would put a livetime under a rate nothing
    measured, which is the failure this whole quantity exists to prevent.
    """
    a = pd.DataFrame({"gpsStart": [0.0, 10.0]})
    with pytest.raises(ValueError):
        analysed_span([a, pd.DataFrame({"gpsStart": []})])
    with pytest.raises(ValueError):
        analysed_span([])


def test_covered_livetime_sums_the_stretches():
    assert covered_livetime_days([(0.0, 86400.0), (0.0, 43200.0)]) == 1.5
    with pytest.raises(ValueError):
        covered_livetime_days([(10.0, 0.0)])
