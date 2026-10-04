"""Write the paper-path snapshot that tests/test_network_driver.py compares against.

Two detectors (H1, L1) of synthetic strain with one chirp, L1 5 ms later, are
searched by `wdfUnitOnline` in the "zerophase" mode; their triggers go through
stage 2 (`build_detector_graph`, `detector_events`, `event_coefficients`, the
comparison map at twice the light travel time) and stage 3
(`TriggerGraphBuilder(ifos).build(...).candidate_table()`), the sequence of
the catalogue notebook ``known_events_online_offline``. The online triggers,
the detector events and the network table are written to
``paper_path_snapshot.npz``/``.parquet`` files beside this script.

Run it with the code whose output is to be frozen:

    python tests/fixtures/make_paper_path_snapshot.py
"""
import glob
import os
import sys
import tempfile
from itertools import combinations

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FS, T0, SECONDS = 4096.0, 1.0e9, 110.0
CHIRP_AT, CHIRP_S, DELAY = 80.0, 3.0, {"H1": 0.0, "L1": 0.005}
SEGMENT = (T0 + 26.0, T0 + SECONDS)


def parameters(ifo, outdir, **kw):
    from wdf.config.Parameters import Parameters

    par = dict(sampling=4096, ResamplingFactor=2, window=1024, overlap=256, threshold=5.0,
               channel=f"{ifo}:TEST", itf=ifo, ARorder=60, learn=20, len=1.0,
               LowFrequencyCut=12.0, BandPassFilter="butterworth", MinFrequency=16.0,
               BlockRemainder="merge", ARRefitInterval=0.0, ARFitContext=5.0,
               RidgeBackground=64.0, RidgeBackgroundMinimum=32.0, WhiteningMode="zerophase",
               ZeroPhaseDuration=8.0, outdir=outdir + "/", run="snapshot")
    par.update(kw)
    return Parameters(**par)


def strain(ifo):
    seed = {"H1": 1, "L1": 2}[ifo]
    rng = np.random.default_rng(seed)
    n = int(SECONDS * FS)
    x = 0.01 * np.cumsum(rng.standard_normal(n)) + rng.standard_normal(n)
    t = np.arange(n) / FS - CHIRP_AT - DELAY[ifo]
    inside = (t >= 0) & (t < CHIRP_S)
    f = 40.0 + 160.0 * np.clip(t, 0, CHIRP_S) / CHIRP_S
    c = np.where(inside, np.sin(2 * np.pi * np.cumsum(f) / FS)
                 * np.sin(np.pi * np.clip(t, 0, CHIRP_S) / CHIRP_S) ** 2, 0.0)
    return x + 0.6 * c


def run_detector(ifo, outdir, **kw):
    """One detector's online run; its trigger table read back as the
    catalogue reads it, and the online result."""
    from wdf.analysis.io import triggers_from_files
    from wdf.processes.wdfUnitOnline import ArraySource, wdfUnitOnline

    worker = wdfUnitOnline(parameters(ifo, outdir, **kw))
    result = worker.segmentProcess(SEGMENT, ArraySource(strain(ifo), T0, FS))
    files = sorted(glob.glob(worker.dir + "WDFTriggers-*.parquet"))
    return triggers_from_files(files, ifo), result, worker


def paper_network(triggers):
    """Stages 2 and 3 as the catalogue notebook runs them."""
    from wdf.analysis.detector_graph import (build_detector_graph, detector_events,
                                             event_coefficients)
    from wdf.analysis.detectors import light_travel_time
    from wdf.analysis.network_graph import TriggerGraphBuilder

    ifos = list(triggers)
    longest = max(light_travel_time(a, b) for a, b in combinations(ifos, 2))
    events, maps, comparison = {}, {}, {}
    for i in ifos:
        g = build_detector_graph(triggers[i])
        labels = g.components()
        ev = detector_events(g, labels=labels)
        ev["ifo"] = i
        events[i] = ev
        maps[i] = event_coefficients(g, labels=labels)
        comparison[i] = event_coefficients(g, labels=labels, bin_seconds=2.0 * longest)
    network = TriggerGraphBuilder(ifos=ifos).build(events, maps, comparison=comparison)
    cand = network.candidate_table()
    for side in ("i", "j"):
        cand[f"ifo_{side}"] = network.nodes["ifo"].to_numpy()[cand[f"node_{side}"].astype(int)]
        cand[f"cluster_{side}"] = network.nodes["cluster_id"].to_numpy()[cand[f"node_{side}"].astype(int)]
    return events, cand


def main():
    with tempfile.TemporaryDirectory() as outdir:
        runs = {ifo: run_detector(ifo, outdir) for ifo in DELAY}
    triggers = {ifo: r[0] for ifo, r in runs.items()}
    events, cand = paper_network(triggers)
    for ifo in DELAY:
        triggers[ifo].to_parquet(os.path.join(HERE, f"paper_path_triggers_{ifo}.parquet"), index=False)
        events[ifo].to_parquet(os.path.join(HERE, f"paper_path_events_{ifo}.parquet"), index=False)
        pd.DataFrame(runs[ifo][1].triggers).to_parquet(
            os.path.join(HERE, f"paper_path_online_{ifo}.parquet"), index=False)
    cand.to_parquet(os.path.join(HERE, "paper_path_network.parquet"), index=False)
    print({ifo: len(t) for ifo, t in triggers.items()}, {i: len(e) for i, e in events.items()},
          len(cand))


if __name__ == "__main__":
    sys.exit(main())
