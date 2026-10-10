"""The online model's fit stretch, chosen from the past by a quality rule."""
import numpy as np
import pytest

pytest.importorskip("py4tsa")

from test_online_worker import FS, T0, parameters, strain  # noqa: E402

from wdf.processes.wdfUnitOnline import ArraySource, wdfUnitOnline  # noqa: E402

END = T0 + 100.0


def with_glitch(x, at, amplitude=60.0, f=200.0, q=8.0):
    t = np.arange(x.size) / FS - at
    tau = q / (2 * np.pi * f) / np.sqrt(2)
    return x + amplitude * np.exp(-t ** 2 / (2 * tau ** 2)) * np.cos(2 * np.pi * f * t)


def loud(x, start, stop, factor=4.0):
    y = x.copy()
    a, b = int(start * FS), int(stop * FS)
    y[a:b] += (factor - 1.0) * np.random.default_rng(7).standard_normal(b - a)
    return y


def run(x, **kw):
    par = parameters(ARRefitInterval=30.0, ARFitStep=10.0, ARFitCandidates=3, **kw)
    return wdfUnitOnline(par).process(ArraySource(x, T0, FS), T0 + 26.0, END)


@pytest.mark.parametrize("mode", ["causal", "zerophase"])
def test_a_refit_skips_a_stretch_holding_a_glitch(mode):
    kw = dict(WhiteningMode="zerophase", ZeroPhaseDuration=8.0) if mode == "zerophase" else {}
    r = run(with_glitch(strain(), 50.0), **kw)
    refit = [f for f in r.refits if f["emitted_at"] >= T0 + 56.0][0]
    assert refit["accepted"] and refit["candidate"] == 1
    assert refit["fit_end"] <= T0 + 50.0 - 0.5
    clean = run(strain(), **kw)
    first = [f for f in clean.refits if f["emitted_at"] >= T0 + 56.0][0]
    assert first["candidate"] == 0 and first["fit_end"] == pytest.approx(T0 + 56.0, abs=1.0)


def test_a_loud_stretch_is_rejected_by_its_level():
    r = run(loud(strain(), 38.0, 58.0))
    refit = [f for f in r.refits if f["emitted_at"] >= T0 + 56.0][0]
    assert refit["accepted"] and refit["candidate"] == 2 and refit["fit_end"] <= T0 + 38.5


def test_with_no_stretch_passing_the_current_model_is_kept():
    r = run(loud(strain(), 28.0, 60.0))
    first_refit = [f for f in r.refits if f["emitted_at"] >= T0 + 56.0][0]
    assert first_refit["accepted"] is False
    assert r.parameters.RefitRejected >= 1


def test_the_rule_is_off_when_asked():
    r = run(with_glitch(strain(), 50.0), ARFitQuality=False)
    refit = [f for f in r.refits if f["emitted_at"] >= T0 + 56.0][0]
    assert refit["candidate"] == 0


def test_the_first_fit_waits_for_a_clean_stretch():
    r = run(with_glitch(strain(), 15.0))
    first = r.refits[0]
    assert first["accepted"] and first["candidate"] == 1 and first["fit_start"] > T0 + 15.5
    assert r.triggers and min(t["gps"] for t in r.triggers) >= first["emitted_at"] - 1.0


def test_the_rule_reads_only_the_past():
    """Altering the input after t changes nothing emitted at or before t, the
    refits included."""
    x = with_glitch(strain(), 50.0)
    t = T0 + 70.0
    y = x.copy()
    i = int(round((t - T0) * FS))
    y[i:] = np.random.default_rng(9).standard_normal(y.size - i) * 3.0
    a, b = run(x), run(y)
    assert a.before(t) == b.before(t)
    assert len(a.before(t)["refits"]) >= 2


# ---- ARFitSubStretch: a fit on the stretch between census transients
GLITCHES = (30.0, 40.0, 50.0)     # in each of the refit's candidates 36-56, 26-46, 16-36
MODES = {"causal": {}, "zerophase": dict(WhiteningMode="zerophase", ZeroPhaseDuration=8.0)}


def glitched(times=GLITCHES):
    x = strain()
    for at in times:
        x = with_glitch(x, at)
    return x


def refit_at_56(r):
    return [f for f in r.refits if f["emitted_at"] >= T0 + 56.0][0]


@pytest.mark.parametrize("mode", sorted(MODES))
def test_a_refit_takes_the_latest_stretch_between_glitches(mode):
    """Every candidate holds a glitch: with "latest" the refit is fitted on the
    most recent clean stretch long enough, inside its candidate, before the
    switch."""
    r = run(glitched(), ARFitSubStretch="latest", ARFitMinimumLearn=9.0, **MODES[mode])
    refit = refit_at_56(r)
    assert refit["accepted"] and refit["substretch"] and refit["candidate"] == 0
    assert T0 + 40.0 < refit["fit_start"] and refit["fit_end"] < T0 + 50.0
    assert refit["fit_end"] - refit["fit_start"] >= 9.0 - 1.0 / FS
    assert refit["candidate_start"] <= refit["fit_start"] and refit["fit_end"] <= refit["candidate_end"]
    assert refit["excluded_s"] == pytest.approx(
        20.0 - (refit["fit_end"] - refit["fit_start"]), abs=1e-6)
    assert refit["fit_end"] <= refit["emitted_at"]
    model = [m for m in r.models if m["fit_start"] == refit["fit_start"]]
    assert len(model) == 1 and model[0]["fit_end"] == refit["fit_end"]


def test_a_refit_takes_the_longest_stretch_between_glitches():
    r = run(glitched(), ARFitSubStretch="longest", ARFitMinimumLearn=9.0)
    refit = refit_at_56(r)
    assert refit["accepted"] and refit["substretch"] and refit["candidate"] == 2
    assert refit["fit_start"] == pytest.approx(T0 + 16.0, abs=1e-3) and refit["fit_end"] < T0 + 30.0
    assert refit["fit_end"] - refit["fit_start"] > 13.0


def test_without_sub_stretches_nothing_changes():
    """"none" (the default) keeps the current model when every candidate holds
    a glitch, as before; the rejected row holds every candidate's levels."""
    x = glitched()
    a, b = run(x), run(x, ARFitSubStretch="none")
    assert a.before(END) == b.before(END)
    refit = refit_at_56(a)
    assert refit["accepted"] is False and "substretch" not in refit
    assert len(refit["candidate_census_peaks"].split(";")) == refit["candidates"] == 3
    assert min(float(p) for p in refit["candidate_census_peaks"].split(";")) == pytest.approx(
        refit["census_peak"], rel=1e-5)
    assert len(refit["candidate_level_max"].split(";")) == 3


def test_a_sub_stretch_too_short_is_not_taken():
    r = run(glitched(), ARFitSubStretch="latest", ARFitMinimumLearn=15.0)
    refit = refit_at_56(r)
    assert refit["accepted"] is False and refit["substretches_tried"] == 0
    assert r.parameters.RefitRejected >= 1


def test_the_shortest_sub_stretch_must_exceed_the_filter():
    with pytest.raises(ValueError, match="ARFitMinimumLearn"):
        wdfUnitOnline(parameters(ARFitSubStretch="latest", ARFitMinimumLearn=5.0, **MODES["zerophase"]))
    with pytest.raises(ValueError, match="ARFitMinimumLearn"):
        wdfUnitOnline(parameters(ARFitSubStretch="latest", ARFitMinimumLearn=25.0))
    with pytest.raises(ValueError, match="ARFitSubStretch"):
        wdfUnitOnline(parameters(ARFitSubStretch="first"))
    par = wdfUnitOnline(parameters(ARFitSubStretch="longest")).par
    assert par.ARFitMinimumLearn == pytest.approx(0.5 * par.learn)
    assert par.ARFitExcludePad == pytest.approx(0.1) and par.ARFitLevelCheck is True


@pytest.mark.parametrize("mode", sorted(MODES))
def test_the_first_fit_takes_a_sub_stretch(mode):
    """The first candidate (6-26) holds a glitch at 15: the first model is
    fitted on its latest clean part, at once, judged by its own model."""
    r = run(with_glitch(strain(), 15.0), ARFitSubStretch="latest", ARFitMinimumLearn=9.0,
            **MODES[mode])
    first = r.refits[0]
    assert first["accepted"] and first["substretch"] and first["candidate"] == 0
    assert first["fit_start"] > T0 + 15.0 and first["fit_end"] == pytest.approx(T0 + 26.0, abs=1.0)
    assert first["fit_end"] <= first["emitted_at"]


def test_the_level_check_can_be_off():
    r = run(loud(strain(), 28.0, 60.0), ARFitLevelCheck=False)
    refit = refit_at_56(r)
    assert refit["accepted"] and refit["candidate"] == 0 and refit["level_max"] > 4.0


def test_sub_stretches_read_only_the_past():
    x = glitched()
    t = T0 + 70.0
    y = x.copy()
    i = int(round((t - T0) * FS))
    y[i:] = np.random.default_rng(9).standard_normal(y.size - i) * 3.0
    kw = dict(ARFitSubStretch="latest", ARFitMinimumLearn=9.0)
    a, b = run(x, **kw), run(y, **kw)
    assert a.before(t) == b.before(t)
    assert any(f.get("substretch") for f in a.before(t)["refits"])


def scale_too_low(monkeypatch, after=1):
    """Every fit after the first `after` returns its model with the
    prediction-error scale four times too low."""
    import wdf.processes.wdfUnitOnline as online

    fitted = online.fit_autoregressive
    calls = []

    def fit(samples, start, rate, order):
        ar = np.array(fitted(samples, start, rate, order), dtype=float)
        calls.append(start)
        if len(calls) > after:
            ar[0] *= 0.25
        return ar

    monkeypatch.setattr(online, "fit_autoregressive", fit)


def test_the_own_level_check_is_off_by_default_and_parsed():
    par = wdfUnitOnline(parameters()).par
    assert par.ARFitOwnLevelCheck is False and par.ARFitOwnLevelRatio == par.ARFitLevelRatio
    par = wdfUnitOnline(parameters(ARFitOwnLevelCheck=True, ARFitOwnLevelRatio=2.0)).par
    assert par.ARFitOwnLevelCheck is True and par.ARFitOwnLevelRatio == 2.0
    with pytest.raises(ValueError, match="ARFitOwnLevelRatio"):
        wdfUnitOnline(parameters(ARFitOwnLevelRatio=1.0))


def test_a_refit_with_a_wrong_scale_passes_without_the_own_level_check(monkeypatch):
    scale_too_low(monkeypatch)
    refit = refit_at_56(run(strain()))
    assert refit["accepted"] and refit["sigma"] > 0 and "own_level_max" not in refit


def test_the_own_level_check_rejects_a_refit_with_a_wrong_scale(monkeypatch):
    scale_too_low(monkeypatch)
    r = run(strain(), ARFitOwnLevelCheck=True)
    first = r.refits[0]
    assert first["accepted"] and first["own_level_ok"]
    assert 1 / 4 <= first["own_level_min"] <= first["own_level_max"] <= 4
    refit = refit_at_56(r)
    assert refit["accepted"] is False and r.parameters.RefitRejected >= 1
    assert all(float(v) > 4.0 for v in refit["candidate_own_level_max"].split(";"))


def test_the_own_level_check_keeps_a_good_refit():
    plain = refit_at_56(run(strain()))
    checked = refit_at_56(run(strain(), ARFitOwnLevelCheck=True))
    assert checked["accepted"] and checked["own_level_ok"]
    assert (checked["candidate"], checked["fit_start"]) == (plain["candidate"], plain["fit_start"])
