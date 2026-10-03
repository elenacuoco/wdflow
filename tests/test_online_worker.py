"""The online worker: causal from the strain to the triggers and the ridge."""
import glob
import os

import numpy as np
import pytest

pytest.importorskip("py4tsa")

from wdf.config.Parameters import Parameters
from wdf.processes.wdfUnitOnline import (ArraySource, CausalPredictionError, fit_autoregressive,
                                         wdfUnitOnline)

FS, T0, SECONDS = 4096.0, 1.0e9, 170.0
CHIRP_AT, CHIRP_S = 128.0, 3.0


def parameters(**kw):
    par = dict(sampling=4096, ResamplingFactor=2, window=1024, overlap=256, threshold=5.0,
               channel="H1:TEST", itf="H1", ARorder=60, learn=20, len=1.0, LowFrequencyCut=12.0,
               BandPassFilter="butterworth", MinFrequency=16.0, BlockRemainder="merge",
               ARRefitInterval=30.0, ARFitContext=5.0, RidgeBackground=64.0,
               RidgeBackgroundMinimum=32.0)
    par.update(kw)
    return Parameters(**par)


def strain(seed=0):
    rng = np.random.default_rng(seed)
    n = int(SECONDS * FS)
    x = 0.01 * np.cumsum(rng.standard_normal(n)) + rng.standard_normal(n)
    t = np.arange(n) / FS - CHIRP_AT
    inside = (t >= 0) & (t < CHIRP_S)
    f = 40.0 + 160.0 * np.clip(t, 0, CHIRP_S) / CHIRP_S
    c = np.where(inside, np.sin(2 * np.pi * np.cumsum(f) / FS)
                 * np.sin(np.pi * np.clip(t, 0, CHIRP_S) / CHIRP_S) ** 2, 0.0)
    return x + 0.6 * c


@pytest.fixture(scope="module")
def run():
    worker = wdfUnitOnline(parameters())
    return worker.process(ArraySource(strain(), T0, FS), T0 + 26.0, T0 + SECONDS)


def test_altering_the_future_changes_nothing_emitted_before_it(run):
    """Altering the input after t leaves every output emitted at or before t
    as it was: triggers, ridge candidates, the running best, the refits."""
    t = T0 + 120.0
    x = strain()
    i = int(round((t - T0) * FS))
    x[i:] = np.random.default_rng(9).standard_normal(x.size - i) * 3.0
    altered = wdfUnitOnline(parameters()).process(ArraySource(x, T0, FS), T0 + 26.0, T0 + SECONDS)
    before, after = run.before(t), altered.before(t)
    assert before == after
    assert before["triggers"] and before["refits"] and len(before["running"]["score"]) > 100
    assert run.before(T0 + SECONDS) != altered.before(T0 + SECONDS)


def test_every_output_has_a_latency(run):
    assert run.triggers
    for row in run.triggers:
        assert row["emitted_at"] >= row["window_end"] and row["latency"] >= 0.0
        assert row["label"] in ("glitch", "candidate")
    for row in run.candidates:
        assert row["latency"] == pytest.approx(row["emitted_at"] - row["t_end"])
        assert row["latency"] >= 0.0
    for refit in run.refits:
        assert refit["fit_end"] <= refit["emitted_at"]
    assert np.all(run.running["emitted_at"] >= run.running["slot_end"])


def test_the_model_is_refitted_from_the_past(run):
    times = [r["emitted_at"] for r in run.refits]
    assert len(times) >= 4
    np.testing.assert_allclose(np.diff(times), 30.0)


def test_the_ridge_follows_the_chirp(run):
    on = [c for c in run.candidates
          if c["t_start"] < T0 + CHIRP_AT + CHIRP_S and c["t_end"] > T0 + CHIRP_AT]
    assert on
    best = max(run.candidates, key=lambda c: c["score"])
    assert best in on
    assert best["waveform_energy"] > 0.0


def test_the_whitening_is_the_lattice_in_direct_form():
    """After a warm-up of the model's order, the direct-form filter and
    p4TSA's lattice give the same stream."""
    from wdf.processes.Whitening import Whitening
    from wdf.processes.BandPassDownSampling import SV_to_array
    from wdf.structures.array2SeqView import array2SeqView

    rate, order = 2048.0, 40
    rng = np.random.default_rng(4)
    x = np.convolve(rng.standard_normal(int(60 * rate)), [1.0, 0.8, 0.3], "same")
    ar = fit_autoregressive(x[:int(30 * rate)], 0.0, rate, order)
    fir = CausalPredictionError(ar, x[:int(40 * rate)])
    ours = fir.process(x[int(40 * rate):])
    lattice = Whitening(order)
    view = array2SeqView(0.0, rate, int(30 * rate))
    view.Fill(0.0, x[:int(30 * rate)])
    lattice.ParametersEstimate(view.SV)
    from py4tsa.tsa import SeqView_double_t as SV
    out = SV()
    warm = array2SeqView(0.0, rate, int(40 * rate))
    warm.Fill(0.0, x[:int(40 * rate)])
    lattice.Process(warm.SV, out)
    rest = array2SeqView(40.0, rate, x.size - int(40 * rate))
    rest.Fill(40.0, x[int(40 * rate):])
    lattice.Process(rest.SV, out)
    theirs = SV_to_array(out) * out.GetScale()
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-8 * np.std(theirs))
    assert np.std(ours) == pytest.approx(ar[0], rel=0.05)


def test_the_online_worker_refuses_what_is_not_causal():
    with pytest.raises(ValueError, match="causal"):
        wdfUnitOnline(parameters(BandPassPhase="zero"))
    with pytest.raises(ValueError, match="ARWarmup"):
        wdfUnitOnline(parameters(ARWarmup=10))


def test_the_online_worker_reads_frames_and_writes_named_files(tmp_outdir):
    from conftest import GPS0, NOISE_GWF

    par = parameters(sampling=16384, ResamplingFactor=4, file=NOISE_GWF, channel="H1:TEST-STRAIN",
                     outdir=tmp_outdir, run="online", ARorder=30, learn=20, ARFitContext=5.0,
                     ARRefitInterval=0.0, window=512, overlap=128, threshold=0.1, Ridge=False)
    result = wdfUnitOnline(par).segmentProcess((GPS0 + 30.0, GPS0 + 70.0))
    folder = glob.glob(os.path.join(tmp_outdir, "online", "H1", "*"))[0]
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, "*")))
    assert any(n.startswith("WDFTriggers-") and n.endswith("-Searchonline.parquet") for n in names)
    assert "parametersUsed-Win512-Searchonline.json" in names
    assert "TriggerLatency-Searchonline.csv" in names
    assert result.triggers and len(result.refits) == 1
    used = Parameters()
    used.load(os.path.join(folder, "parametersUsed-Online-Searchonline.json"))
    assert used.BandPassPhase == "causal" and used.LowPassOrder > 8
