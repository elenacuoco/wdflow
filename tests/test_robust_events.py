from types import SimpleNamespace

import numpy as np
import pandas as pd

from wdf.analysis.robust_events import (
    FARConfig,
    IndexedCoincidenceFinder,
    TimeSlideFAR,
    cluster_detector_triggers,
)


def _events(times, strengths, ifo):
    return pd.DataFrame(
        {
            "cluster_id": np.arange(len(times)),
            "gpsPeak": times,
            "freqMin": 50.0,
            "freqMean": 100.0,
            "freqMax": 200.0,
            "EnWDF": strengths,
            "ifo": ifo,
        }
    )


def test_indexed_coincidence_is_one_to_one_and_exposes_enwdf_statistics():
    finder = IndexedCoincidenceFinder(timing_jitter_s=0.05)
    left = _events([10.00, 10.04], [12.0, 8.0], "H1")
    right = _events([10.02], [7.0], "L1")

    result = finder.find({"H1": left, "L1": right})

    assert len(result) == 1
    assert result.iloc[0].network_enwdf == np.hypot(12.0, 7.0)
    assert result.iloc[0].network_min_enwdf == 7.0
    assert result.iloc[0].dt_s == result.iloc[0].delta_t


def test_far_uses_total_slide_livetime_and_never_reports_exact_zero():
    finder = IndexedCoincidenceFinder(timing_jitter_s=0.05)
    estimator = TimeSlideFAR(finder, FARConfig(n_slides=10))
    candidate = pd.DataFrame(
        {"network_min_enwdf": [20.0], "network_enwdf": [25.0]}
    )
    background = pd.DataFrame({"network_min_enwdf": [1.0, 2.0]})
    background.attrs["total_livetime_s"] = 1000.0

    ranked = estimator.rank_candidates(candidate, background, 100.0)

    assert ranked.iloc[0].n_background_ge == 0
    assert ranked.iloc[0].far_hz == 1.0 / 1000.0
    assert 0.0 < ranked.iloc[0].fap < 1.0


def test_cluster_catalog_keeps_true_enwdf_and_member_indices():
    triggers = pd.DataFrame(
        {
            "gps": [100.0, 100.1],
            "gpsPeak": [100.05, 100.15],
            "duration": [0.1, 0.1],
            "freqMin": [50.0, 60.0],
            "freqMean": [100.0, 110.0],
            "freqMax": [150.0, 160.0],
            "freqPeak": [90.0, 100.0],
            "EnWDF": [4.0, 7.0],
            "snrMean": [1.0, 2.0],
            "snrPeak": [2.0, 3.0],
            "sigma": [1.0, 1.0],
            "ifo": ["H1", "H1"],
        }
    )
    parameters = SimpleNamespace(window=256, overlap=128, resampling=1024)

    _, catalog = cluster_detector_triggers(triggers, parameters)

    assert len(catalog) == 1
    assert catalog.iloc[0].EnWDF == 7.0
    assert catalog.iloc[0].EnWDF == 7.0
    assert catalog.iloc[0].member_indices == (0, 1)
