"""Writing a detector's events to disk, and reading them back whole.

The detector stage and the network stage are separated by a file. The first
reads frames, finds triggers and assembles them into events; the second reads
those events and compares detectors. Nothing about the second changes what the
first produced, so a change to the coincidence, to the ranking or to the
learned stage costs the network stage's time and not the search's.

What has to travel is more than the event table. The network stage reads each
event's wavegram as its node feature and its reconstruction as the arrival time
it is timed on, and both are functions of the coefficients the event kept. The
store therefore holds two tables per detector and frame kind: the events, and
the triggers they were assembled from, each trigger carrying the label of the
event it belongs to. From those two everything downstream rebuilds, because
that is exactly the pair the graph builder is given.

The triggers are what the search wrote, with their surviving coefficients and
the noise scale of the block they came from, so the store is self-contained: a
reader needs neither the frames nor `pytsa`.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

#: Written beside the tables, so that a store states what produced it.
MANIFEST = "manifest.json"


def _paths(directory, ifo, kind):
    stem = os.path.join(directory, f"{ifo}-{kind}")
    return f"{stem}-events.parquet", f"{stem}-triggers.parquet"


def _series_path(directory, ifo, kind):
    return os.path.join(directory, f"{ifo}-{kind}-series.parquet")


def save_events(directory: str, ifo: str, kind: str, events: pd.DataFrame,
                triggers: pd.DataFrame, labels, provenance: dict | None = None,
                series: dict | None = None):
    """Write one detector's events and the triggers they were built from.

    :type directory: str
    :param directory: where the store lives; created if it does not exist.
    :type ifo: str
    :param ifo: the detector these events belong to.
    :type kind: str
    :param kind: which frames they came from, such as ``foreground``.
    :type events: pandas.DataFrame
    :param events: the event table, as `detector_events` returns it. Its order
        is preserved: the network stage indexes its prepared arrays by
        position, so a reordered catalogue is a different one.
    :type triggers: pandas.DataFrame
    :param triggers: the graph's nodes, carrying the surviving coefficients and
        the noise scale. Reordering these breaks the labels, so they are
        written as they are.
    :param labels: the event each trigger belongs to, one label per row of
        `triggers`, as `components` returns.
    :type provenance: dict or None
    :param provenance: what produced the store --- the data it read, the
        configuration, the livetime. Written to the manifest and never used by
        the reader, which is what keeps a run's identity out of the code.
    :type series: dict or None
    :param series: ``{cluster_id: (gps_start, samples)}``, each event's
        reconstruction. The network stage reads the arrival-time difference on
        it, and inverting an event's coefficients is the expensive half of the
        detector stage, so a store that carries the waveform is what lets a
        second stage run without repeating the first.
    :return: tuple -- the two paths written.
    :raises ValueError: if `labels` does not have one entry per trigger.
    """
    labels = np.asarray(labels)
    if len(labels) != len(triggers):
        raise ValueError(
            f"{len(labels)} labels for {len(triggers)} triggers: the label "
            "places a trigger in an event, so there is one of each")

    os.makedirs(directory, exist_ok=True)
    events_path, triggers_path = _paths(directory, ifo, kind)
    events.reset_index(drop=True).to_parquet(events_path)
    triggers.reset_index(drop=True).assign(
        cluster_id=labels.astype(np.int64)).to_parquet(triggers_path)

    if series:
        pd.DataFrame([
            dict(cluster_id=int(label), gps_start=float(start),
                 samples=np.asarray(samples, dtype=float))
            for label, (start, samples) in series.items()
        ]).to_parquet(_series_path(directory, ifo, kind))

    manifest_path = os.path.join(directory, MANIFEST)
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    manifest[f"{ifo}-{kind}"] = dict(
        provenance or {}, events=int(len(events)), triggers=int(len(triggers)))
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True, default=str)
    return events_path, triggers_path


def load_events(directory: str, ifo: str, kind: str):
    """Read back what `save_events` wrote.

    :type directory: str
    :param directory: the store.
    :type ifo: str
    :param ifo: the detector.
    :type kind: str
    :param kind: the frame kind.
    :return: tuple -- ``(events, triggers, labels, series)``, in the order they
        were written, with `labels` a numpy array of the event each trigger
        belongs to, the `cluster_id` column removed from the triggers, and
        `series` the reconstructions keyed by event, empty where none were
        written.
    :raises FileNotFoundError: if either table is missing.
    """
    events_path, triggers_path = _paths(directory, ifo, kind)
    for path in (events_path, triggers_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"no store at {path}")
    events = pd.read_parquet(events_path)
    triggers = pd.read_parquet(triggers_path)
    labels = triggers["cluster_id"].to_numpy(dtype=np.int64)

    series = {}
    series_path = _series_path(directory, ifo, kind)
    if os.path.exists(series_path):
        stored = pd.read_parquet(series_path)
        series = {int(row.cluster_id): (float(row.gps_start),
                                        np.asarray(row.samples, dtype=float))
                  for row in stored.itertuples()}
    return events, triggers.drop(columns=["cluster_id"]), labels, series


def stored_manifest(directory: str) -> dict:
    """What the store says about itself.

    :type directory: str
    :param directory: the store.
    :return: dict -- keyed ``"{ifo}-{kind}"``, empty if nothing was written.
    """
    path = os.path.join(directory, MANIFEST)
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)

def is_current(directory: str, ifo: str, kind: str, sources) -> bool:
    """Whether a store may be read instead of the stage being run again.

    A store describes the triggers it was built from. If those triggers have
    been written since, it describes a different search and reading it would
    report one run's events against another's data. The rule is the one a
    trigger cache is held to: every file the store consists of must exist and
    be newer than every file it was derived from.

    :type directory: str
    :param directory: the store.
    :type ifo: str
    :param ifo: the detector.
    :type kind: str
    :param kind: the frame kind.
    :param sources: paths the store was derived from, such as the trigger
        files. An empty collection makes any existing store current, which is
        what a caller means when nothing on disk precedes it.
    :return: bool -- True when the store may be read.
    """
    written = []
    for path in _paths(directory, ifo, kind):
        if not os.path.exists(path):
            return False
        written.append(os.path.getmtime(path))
    newest_source = max((os.path.getmtime(path) for path in sources
                         if os.path.exists(path)), default=0.0)
    return min(written) >= newest_source
