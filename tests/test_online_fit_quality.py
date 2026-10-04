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
