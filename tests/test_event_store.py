"""The file that separates the detector stage from the network stage."""
import numpy as np
import pandas as pd
import pytest

from wdf.analysis.event_store import load_events, save_events, stored_manifest


def a_store(tmp_path, n_triggers=5):
    events = pd.DataFrame(dict(cluster_id=[0, 1], EnWDF=[3.0, 4.0],
                               gpsStart=[10.0, 20.0]))
    triggers = pd.DataFrame(dict(gps=np.arange(n_triggers, dtype=float),
                                 EnWDF=np.arange(n_triggers, dtype=float),
                                 wt_index=[[1, 2]] * n_triggers))
    labels = np.array([0] * (n_triggers - 1) + [1])
    save_events(str(tmp_path), "H1", "foreground", events, triggers, labels,
                provenance=dict(dataset="a stretch", livetime_days=1.5))
    return events, triggers, labels


def test_what_is_written_is_what_comes_back(tmp_path):
    events, triggers, labels = a_store(tmp_path)
    back_events, back_triggers, back_labels = load_events(
        str(tmp_path), "H1", "foreground")

    pd.testing.assert_frame_equal(back_events, events.reset_index(drop=True))
    pd.testing.assert_frame_equal(back_triggers, triggers.reset_index(drop=True))
    np.testing.assert_array_equal(back_labels, labels)
    # The label places a trigger in an event, so it does not survive as a
    # column: it comes back beside the triggers, as the graph reads it.
    assert "cluster_id" not in back_triggers


def test_the_order_is_the_one_the_network_stage_indexes_by(tmp_path):
    """The prepared node arrays are indexed by position, so a store that
    reordered the events would hand the second stage a different catalogue."""
    events, _, _ = a_store(tmp_path)
    back, _, _ = load_events(str(tmp_path), "H1", "foreground")
    np.testing.assert_array_equal(back.cluster_id.to_numpy(),
                                  events.cluster_id.to_numpy())


def test_a_label_per_trigger_or_nothing(tmp_path):
    events = pd.DataFrame(dict(cluster_id=[0]))
    triggers = pd.DataFrame(dict(gps=[1.0, 2.0]))
    with pytest.raises(ValueError, match="one of each"):
        save_events(str(tmp_path), "H1", "foreground", events, triggers, [0])


def test_the_store_says_what_produced_it(tmp_path):
    a_store(tmp_path)
    manifest = stored_manifest(str(tmp_path))
    assert manifest["H1-foreground"]["dataset"] == "a stretch"
    assert manifest["H1-foreground"]["events"] == 2
    assert manifest["H1-foreground"]["triggers"] == 5


def test_a_missing_store_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no store at"):
        load_events(str(tmp_path), "L1", "background")
    assert stored_manifest(str(tmp_path)) == {}
