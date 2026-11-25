"""wdf.processes.noise_model: the quiet stretch from the masks, the rules,
the whitening check and the order of the fallback."""
import numpy as np
import pytest

from wdf.config.Parameters import Parameters
from wdf.processes.noise_model import (FAILED, WORKER, NoiseModelSettings, NoiseRun, good_seconds,
                                       quiet_stretch, transient_seconds)

RATE = 2048.0


def settings(**kw):
    values = dict(learn_s=300.0, context_s=20.0, lookback_s=6 * 3600.0, exclude_pad_s=0.1,
                  minimum_learn_s=150.0, order=8, low_frequency=16.0)
    values.update(kw)
    return NoiseModelSettings(**values)


# ------------------------------------------------------------------ the masks
def test_a_second_is_good_when_every_bit_asked_is_set():
    dq = np.array([511, 511, 510, 1023, np.nan, 511])
    inj = np.array([7, 3, 7, 7, 7, 31])
    assert good_seconds(dq, inj, 511, 7).tolist() == [True, False, False, True, False, True]
    assert good_seconds(dq, inj, 1, 0).tolist() == [True, True, False, True, False, True]   # 510 lacks DATA


def test_the_quiet_stretch_is_the_latest_with_good_masks():
    s = settings()
    origin, before = 1000, 1000 + 8000
    good = np.ones(8100, bool)
    found, rec = quiet_stretch(good, origin, before, s)
    assert found == (before - 300, before) and rec["candidates"] == 1
    # a bad second inside the stretch or its context: the stretch ends context before it
    good[8900 - origin] = False                         # GPS 8900
    found, _ = quiet_stretch(good, origin, before, s)
    assert found == (8900 - 20 - 300, 8900 - 20)
    a, end = found
    assert good[a - 20 - origin:end + 20 - origin].all()


def test_a_transient_moves_the_stretch_before_it_less_the_pad():
    s = settings(exclude_pad_s=2.0)
    origin, before = 0, 5000
    good = np.ones(5100, bool)
    seen = []

    def inspect(a, end):
        seen.append((a, end))
        if a <= 4800 < end:                              # a transient second at GPS 4800
            return int(np.floor(4800 - 2.0)), f"{a}-{end}:4800"
        return None

    found, rec = quiet_stretch(good, origin, before, s, inspect)
    assert found == (4798 - 300, 4798)
    assert seen[0] == (4700, 5000) and rec["rejected"] == ["4700-5000:4800"]


def test_no_stretch_within_the_lookback():
    s = settings(lookback_s=1000.0)
    good = np.ones(3000, bool)
    good[1000:2100:250] = False                            # a bad second every 250 s from 1000
    found, rec = quiet_stretch(good, 0, 2000.0, s)        # every candidate within 1000 s holds one
    assert found is None and rec["candidates"] == 0
    found, _ = quiet_stretch(good, 0, 2000.0, settings(lookback_s=2000.0))
    assert found == (1000 - 20 - 300, 1000 - 20)


def test_the_bands_default_to_octaves_from_the_low_frequency_cut():
    s = settings()
    assert s.bands_at(RATE) == ((16.0, 32.0), (32.0, 64.0), (64.0, 128.0), (128.0, 256.0),
                                (256.0, 512.0), (512.0, 1000.0))
    assert settings(bands=[[20, 40], [40, 80]]).bands_at(RATE) == ((20.0, 40.0), (40.0, 80.0))


def test_a_loud_second_is_a_transient():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(int(60 * RATE))
    t = np.arange(y.size) / RATE
    y += 40 * np.exp(-((t - 30.5) / 0.01) ** 2) * np.sin(2 * np.pi * 100 * t)
    loud, bad, secs = transient_seconds(y, 100.0, RATE, settings(exclude_pad_s=1.0))
    assert loud.tolist() == [130.0]
    # left out: every second meeting [130 - pad, 131 + pad]
    assert secs[bad].tolist() == [128.0, 129.0, 130.0, 131.0, 132.0]


def test_the_settings_keep_the_released_behaviour_by_default():
    par = Parameters(learn=300, ARorder=100, LowFrequencyCut=16.0)
    s = NoiseModelSettings.from_parameters(par)
    assert s.model == WORKER and s.rules() == ()
    assert par.NoiseModel == "worker" and par.NoiseModelFallback == ["piecewise", "spectral"]
    assert s.exclude_pad_s == pytest.approx(0.1) and s.minimum_learn_s == pytest.approx(150.0)
    par = Parameters(learn=300, ARorder=100, NoiseModel="quiet")
    assert NoiseModelSettings.from_parameters(par).rules() == ("quiet", "piecewise", "spectral")
    par = Parameters(learn=300, ARorder=100, NoiseModel="spectral", NoiseModelFallback=["piecewise"])
    assert NoiseModelSettings.from_parameters(par).rules() == ("spectral", "piecewise")
    with pytest.raises(ValueError):
        NoiseModelSettings(model="burg")
    with pytest.raises(ValueError):
        NoiseModelSettings(fallback=("quiet",))


# ------------------------------------------------------------------ the rules (compiled core)
def coloured(n, seed, a=(1.2, -0.5), scale=1.0):
    """An AR(2) process: a stable coloured noise with a known spectrum."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    return lfilter([scale], [1.0, -a[0], -a[1]], rng.standard_normal(n))


def check_parameters(**kw):
    values = dict(ZeroPhaseDuration=4.0, resampling=int(RATE), ARorder=8, learn=60, ARFitMinimumLearn=30.0,
                  LowFrequencyCut=16.0, WhiteningMode="zerophase", itf="H1")
    values.update(kw)
    return Parameters(**values)


def run_of(y, start=0.0):
    return NoiseRun(conditioned=(y, start, RATE), start=start + 8.0, end=start + y.size / RATE - 8.0,
                    bounds=(start + 10.0, start + y.size / RATE - 10.0))


def test_the_spectral_model_is_stable_and_whitens():
    pytest.importorskip("py4tsa")
    from wdf.processes.noise_model import check_models, model_sigmas, spectral_model

    par = check_parameters()
    s = NoiseModelSettings.from_parameters(par)
    run = run_of(coloured(int(200 * RATE), 1))
    models, stable, row = spectral_model(run, s)
    assert stable and row["max_reflection"] < 1 and len(models) == 1
    assert models[0]["switch_gps"] == run.start and models[0]["ar"].shape == (par.ARorder + 1,)
    # the model recovers the process's coefficients (a1, a2) in its first two predictors
    assert models[0]["ar"][1:3] == pytest.approx([1.2, -0.5], abs=0.05)
    spectra, rows = check_models(run, model_sigmas(models, par), par, s)
    assert rows and all(r["passes"] for r in rows), rows


def test_the_fallback_order_and_a_forced_failure(monkeypatch):
    pytest.importorskip("py4tsa")
    from wdf.processes import noise_model as nm

    par = check_parameters(NoiseModel="quiet")
    s = NoiseModelSettings.from_parameters(par)
    run = run_of(coloured(int(200 * RATE), 2))

    # no quiet stretch: the piecewise rule fits the stretch and passes
    choice = nm.choose_noise_model([run], par, s, quiet=(None, dict(status="no stretch")))
    assert choice.method == "piecewise" and choice.applied == "piecewise"
    assert [a["rule"] for a in choice.attempts] == ["quiet", "piecewise"]
    assert choice.attempts[0]["status"] == "no stretch" and choice.attempts[1]["passes"]
    assert len(choice.models[0]) == len(np.arange(run.start, run.end, par.learn))

    # a quiet model of another noise (white, wrong scale) fails the check: the next rule
    wrong = np.zeros(par.ARorder + 1)
    wrong[0] = 0.05
    choice = nm.choose_noise_model([run], par, s, quiet=(wrong, dict(fit_start=-400.0, fit_end=-100.0)))
    assert [(a["rule"], a["passes"]) for a in choice.attempts] == [("quiet", False), ("piecewise", True)]
    assert choice.method == "piecewise"

    # a quiet model fitted on the same noise passes, and is kept
    from wdf.processes.wdfUnitOnline import fit_autoregressive
    other = coloured(int(80 * RATE), 3)
    ar = fit_autoregressive(other[int(10 * RATE):int(70 * RATE)], 0.0, RATE, par.ARorder)
    choice = nm.choose_noise_model([run], par, s, quiet=(ar, dict(fit_start=-70.0, fit_end=-10.0)))
    assert choice.method == "quiet" and [a["rule"] for a in choice.attempts] == ["quiet"]
    assert choice.models[0][0]["switch_gps"] == run.start and choice.models[0][0]["fit_end"] == -10.0

    # forced failure: no rule can pass; the spectral rule unstable is skipped unchecked,
    # the detector is "failed" with the models of the last stable rule tried
    par = check_parameters(NoiseModel="quiet", NoiseCheckRange=[5.0, 6.0])
    s = NoiseModelSettings.from_parameters(par)
    real = nm.spectral_model
    monkeypatch.setattr(nm, "spectral_model", lambda r, st: (real(r, st)[0], False, dict(rule="spectral")))
    choice = nm.choose_noise_model([run], par, s, quiet=(ar, dict(fit_start=-70.0, fit_end=-10.0)))
    assert choice.method == FAILED and choice.applied == "piecewise"
    assert [(a["rule"], a["status"].split(":")[0]) for a in choice.attempts] == [
        ("quiet", "checked"), ("piecewise", "checked"), ("spectral", "unstable")]
    monkeypatch.setattr(nm, "spectral_model", real)
    choice = nm.choose_noise_model([run], par, s, quiet=(ar, dict(fit_start=-70.0, fit_end=-10.0)))
    assert choice.method == FAILED and choice.applied == "spectral"

    # "worker": nothing is chosen here
    assert nm.choose_noise_model([run], check_parameters()).method == WORKER


def test_the_worker_applies_the_chosen_models(tmp_path):
    """The models written by save_noise_models are the ones the zero-phase
    worker applies with ARModelsFrom."""
    pytest.importorskip("py4tsa")
    pytest.importorskip("h5py")
    from test_online_worker import FS, SECONDS, T0, parameters, strain

    from wdf.processes.noise_model import save_noise_models
    from wdf.processes.wdfUnitOnline import ArraySource, load_ar_models, wdfUnitOnline

    kw = dict(WhiteningMode="zerophase", ZeroPhaseDuration=8.0, ARRefitInterval=0.0)
    start = T0 + 26.0
    own = wdfUnitOnline(parameters(outdir=str(tmp_path) + "/", run="own", **kw))
    own.segmentProcess((start, T0 + SECONDS), ArraySource(strain(), T0, FS))
    theirs, _ = load_ar_models(own.par.ARModelsFile)
    assert len(theirs) == 1
    rng = np.random.default_rng(5)
    chosen = dict(theirs[0], ar=theirs[0]["ar"] * np.r_[1.0, 1.0 + 1e-3 * rng.standard_normal(theirs[0]["ar"].size - 1)],
                  fit_start=start - 400.0, fit_end=start - 100.0)
    path = save_noise_models(str(tmp_path / "chosen.h5"), [chosen], own.par)
    reuse = wdfUnitOnline(parameters(outdir=str(tmp_path) + "/", run="reuse", ARModelsFrom=path, **kw))
    reuse.segmentProcess((start, T0 + SECONDS), ArraySource(strain(), T0, FS))
    applied, _ = load_ar_models(reuse.par.ARModelsFile)
    assert reuse.par.ARModelsReused == 1 and len(applied) == 1
    assert applied[0]["ar"] == pytest.approx(chosen["ar"], rel=1e-12)
    assert applied[0]["fit_end"] == chosen["fit_end"]
    with pytest.raises(ValueError):
        save_noise_models(str(tmp_path / "x.h5"), [chosen], parameters())        # causal whitening


def test_the_quiet_rule_reads_the_masks_and_the_strain():
    """A bad data-quality second and a loud transient before the stretch: the
    model is fitted on the latest stretch clear of both."""
    pytest.importorskip("py4tsa")
    from test_online_worker import parameters

    from wdf.processes.noise_model import quiet_model
    from wdf.processes.wdfUnitOnline import wdfUnitOnline

    fs, t0, t1 = 4096.0, 1700.0, 2010.0
    rng = np.random.default_rng(4)
    x = rng.standard_normal(int((t1 - t0) * fs))
    t = t0 + np.arange(x.size) / fs
    x += 200 * np.exp(-((t - 1970.5) / 0.01) ** 2) * np.sin(2 * np.pi * 150 * t)
    dq = np.full(int(t1 - t0), 511.0)
    dq[int(1980 - t0)] = 510.0                           # DATA bit off at 1980
    inj = np.full(int(t1 - t0), 7.0)

    def read(channel, a, b):
        if channel.endswith("DQMASK") or channel.endswith("INJMASK"):
            src, rate = (dq if channel.endswith("DQMASK") else inj), 1.0
        else:
            src, rate = x, fs
        out = np.full(int(round((b - a) * rate)), np.nan)
        i0, i1 = int(round((max(a, t0) - t0) * rate)), int(round((min(b, t1) - t0) * rate))
        if i1 > i0:
            out[int(round((max(a, t0) - a) * rate)):][:i1 - i0] = src[i0:i1]
        return out, rate

    par = wdfUnitOnline(parameters(NoiseModel="quiet", NoiseLookback=250.0)).par
    ar, record = quiet_model(par, read, "H1", 2000.0)
    assert record["status"] == "fitted", record
    # 1980 is bad: end context before it (1975); 1970 is loud: end before 1970 - pad
    assert (record["fit_start"], record["fit_end"]) == (1949.0, 1969.0)
    assert record["rejected"] == "1955-1975:1970" and ar.shape == (par.ARorder + 1,)
    ar, record = quiet_model(par, read, "H1", 1720.0)   # the masks are missing before 1700
    assert ar is None and record["status"].startswith("no 20 s stretch")
