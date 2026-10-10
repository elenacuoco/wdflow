"""ARModelsFrom: a run applies another run's noise models, at its switch times."""
import numpy as np
import pytest

pytest.importorskip("py4tsa")
pytest.importorskip("h5py")

from test_online_worker import FS, SECONDS, T0, parameters, strain  # noqa: E402

from wdf.processes.wdfUnitOnline import ArraySource, load_ar_models, wdfUnitOnline  # noqa: E402

START = T0 + 26.0


class Recording(wdfUnitOnline):
    """The worker, keeping the whitened stream it searches."""

    def _consume(self, w, w_start):
        self.whitened.append((float(w_start), np.array(w, dtype=float)))
        super()._consume(w, w_start)

    def process(self, source, gps_start, gps_end):
        self.whitened = []
        return super().process(source, gps_start, gps_end)


def stream(worker):
    t0 = worker.whitened[0][0]
    return t0, np.concatenate([w for _, w in worker.whitened])


def injections(n, every=6.0, amplitude=40.0):
    """Dense loud sine-Gaussians, as an injection campaign."""
    t = np.arange(n) / FS
    out = np.zeros(n)
    for k, at in enumerate(np.arange(30.0, SECONDS - 5.0, every)):
        f = 100.0 + 40.0 * (k % 5)
        tau = 8.0 / (2 * np.pi * f) / np.sqrt(2)
        out += amplitude * np.exp(-(t - at) ** 2 / (2 * tau ** 2)) * np.cos(2 * np.pi * f * (t - at))
    return out


MODES = {"causal": {}, "zerophase": dict(WhiteningMode="zerophase", ZeroPhaseDuration=8.0)}


@pytest.fixture(scope="module", params=sorted(MODES))
def runs(request, tmp_path_factory):
    kw = MODES[request.param]
    out = str(tmp_path_factory.mktemp(request.param)) + "/"
    x = strain()
    inj = injections(x.size)
    bg = Recording(parameters(outdir=out, run="bg", **kw))
    bg.segmentProcess((START, T0 + SECONDS), ArraySource(x, T0, FS))
    path = bg.par.ARModelsFile
    plain = Recording(parameters(**kw))
    plain.process(ArraySource(x + inj, T0, FS), START, T0 + SECONDS)
    fg = Recording(parameters(ARModelsFrom=path, **kw))
    fg.process(ArraySource(x + inj, T0, FS), START, T0 + SECONDS)
    alone = Recording(parameters(ARModelsFrom=path, **kw))
    alone.process(ArraySource(inj, T0, FS), START, T0 + SECONDS)
    return dict(bg=bg, plain=plain, fg=fg, alone=alone, path=path, x=x, inj=inj, kw=kw)


def test_the_background_writes_its_model_history(runs):
    bg, path = runs["bg"], runs["path"]
    models, attributes = load_ar_models(path)
    accepted = [r for r in bg.result.refits if r.get("accepted", True)]
    assert len(models) == len(accepted) >= 3
    assert attributes["ARorder"] == bg.par.ARorder and attributes["itf"] == "H1"
    assert models[0]["ar"].shape == (bg.par.ARorder + 1,)
    assert np.all(np.diff([m["switch_gps"] for m in models]) > 0)
    for m in models:
        assert m["fit_end"] <= m["switch_gps"]


def test_without_reuse_the_injections_block_the_refits(runs):
    assert runs["plain"].par.RefitRejected >= 1


def test_the_foreground_applies_the_background_models(runs):
    """Same switch times and models; FG - BG whitened is the injections
    whitened by those models, sample for sample."""
    bg, fg, alone = runs["bg"], runs["fg"], runs["alone"]
    assert [m["switch_gps"] for m in fg.result.models] == [m["switch_gps"] for m in bg.result.models]
    for a, b in zip(fg.result.models, bg.result.models):
        np.testing.assert_array_equal(a["ar"], b["ar"])
    assert fg.par.ARModelsReused == len(bg.result.models) and fg.par.RefitRejected == 0
    (t_bg, w_bg), (t_fg, w_fg), (t_in, w_in) = stream(bg), stream(fg), stream(alone)
    assert t_bg == t_fg == t_in and w_bg.size == w_fg.size == w_in.size
    scale = np.max(np.abs(w_in))
    np.testing.assert_allclose(w_fg - w_bg, w_in, atol=1e-9 * scale)
    assert scale > 10 * np.std(w_bg)


def test_reuse_keeps_causality_and_latency(runs):
    """Altering the input after t changes nothing emitted at or before t (at
    or before t - Latency_s in the zero-phase mode)."""
    kw, path = runs["kw"], runs["path"]
    t = T0 + 120.0
    x = runs["x"] + runs["inj"]
    y = x.copy()
    i = int(round((t - T0) * FS))
    y[i:] = np.random.default_rng(9).standard_normal(y.size - i) * 3.0
    a = wdfUnitOnline(parameters(ARModelsFrom=path, **kw)).process(ArraySource(x, T0, FS), START,
                                                                   T0 + SECONDS)
    b = wdfUnitOnline(parameters(ARModelsFrom=path, **kw)).process(ArraySource(y, T0, FS), START,
                                                                   T0 + SECONDS)
    assert a.before(t) == b.before(t) and a.before(t)["triggers"]
    if kw:
        assert a.parameters.Latency_s == runs["bg"].par.Latency_s


def test_a_history_of_another_configuration_is_refused(runs):
    with pytest.raises(ValueError, match="ARorder"):
        wdfUnitOnline(parameters(ARModelsFrom=runs["path"], ARorder=40, ARWarmup=40,
                                 **runs["kw"])).process(ArraySource(runs["x"], T0, FS), START,
                                                        T0 + SECONDS)


@pytest.mark.parametrize("mode", sorted(MODES))
def test_a_sub_stretch_model_is_reused_at_its_switch_time(mode, tmp_path):
    """A background whose refit is fitted on a sub-stretch between glitches
    writes it with its fit stretch; the foreground applies the same model at
    the same switch time."""
    from test_online_fit_quality import glitched

    kw = dict(MODES[mode], ARFitStep=10.0, ARFitCandidates=3)
    x = glitched()
    end = T0 + 100.0
    bg = Recording(parameters(outdir=str(tmp_path) + "/", run="bg", ARFitSubStretch="latest",
                              ARFitMinimumLearn=9.0, **kw))
    bg.segmentProcess((START, end), ArraySource(x, T0, FS))
    sub = [r for r in bg.result.refits if r.get("substretch")]
    assert sub and sub[0]["fit_end"] - sub[0]["fit_start"] < bg.par.learn
    models, _ = load_ar_models(bg.par.ARModelsFile)
    assert [(m["switch_gps"], m["fit_start"], m["fit_end"]) for m in models] == [
        (m["switch_gps"], m["fit_start"], m["fit_end"]) for m in bg.result.models]
    assert any(m["fit_start"] == sub[0]["fit_start"] and m["fit_end"] == sub[0]["fit_end"]
               for m in models)
    fg = Recording(parameters(ARModelsFrom=bg.par.ARModelsFile, **kw))
    fg.process(ArraySource(x + injections(x.size), T0, FS), START, end)
    assert [m["switch_gps"] for m in fg.result.models] == [m["switch_gps"] for m in models]
    for a, b in zip(fg.result.models, models):
        np.testing.assert_array_equal(a["ar"], b["ar"])
        assert (a["fit_start"], a["fit_end"]) == (b["fit_start"], b["fit_end"])
