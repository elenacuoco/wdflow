"""The online glitch label: a causal census of the whitened stream."""
import numpy as np
import pytest

pytest.importorskip("py4tsa")

from test_online_worker import FS, SECONDS, T0, parameters, strain  # noqa: E402

from wdf.processes.causal_census import CausalCensus, CensusSettings  # noqa: E402
from wdf.processes.wdfUnitOnline import ArraySource, wdfUnitOnline  # noqa: E402

GLITCH_AT, GLITCH_F, GLITCH_Q = 100.0, 300.0, 8.0


def glitchy():
    x = strain()
    t = np.arange(x.size) / FS - GLITCH_AT
    tau = GLITCH_Q / (2 * np.pi * GLITCH_F) / np.sqrt(2)
    return x + 40.0 * np.exp(-t ** 2 / (2 * tau ** 2)) * np.cos(2 * np.pi * GLITCH_F * t)


@pytest.fixture(scope="module", params=["causal", "zerophase"])
def labelled(request):
    kw = dict(WhiteningMode="zerophase", ZeroPhaseDuration=16.0) if request.param == "zerophase" else {}
    return wdfUnitOnline(parameters(**kw)).process(ArraySource(glitchy(), T0, FS), T0 + 26.0,
                                                   T0 + SECONDS)


def test_the_glitch_is_labelled_and_the_chirp_is_not(labelled):
    at = [r for r in labelled.triggers if r["gps"] < T0 + GLITCH_AT < r["window_end"]]
    chirp = [r for r in labelled.triggers if r["gps"] < T0 + 131.0 and r["window_end"] > T0 + 128.0]
    assert at and all(r["label"] == "glitch" for r in at)
    assert chirp and all(r["label"] == "candidate" for r in chirp)
    assert len(labelled.triggers) > len(at) + len(chirp)
    assert labelled.parameters.GlitchLabel == "census"
    assert any(abs(c["peak_time"] - T0 - GLITCH_AT) < 0.1 for c in labelled.census)


def test_the_label_removes_nothing():
    plain = wdfUnitOnline(parameters(GlitchLabel="bases")).process(
        ArraySource(glitchy(), T0, FS), T0 + 26.0, T0 + SECONDS)
    census = wdfUnitOnline(parameters()).process(ArraySource(glitchy(), T0, FS), T0 + 26.0,
                                                 T0 + SECONDS)
    strip = [{k: v for k, v in r.items() if k != "label"} for r in plain.triggers]
    assert strip == [{k: v for k, v in r.items() if k != "label"} for r in census.triggers]
    assert {r["label"] for r in plain.triggers} <= {"glitch", "candidate"}
    with pytest.raises(ValueError, match="GlitchLabel"):
        wdfUnitOnline(parameters(GlitchLabel="gate"))


def test_the_census_is_causal():
    """Altering the stream after t changes no transient closed by t, and no
    level read before t."""
    from types import SimpleNamespace

    s = CensusSettings.from_parameters(SimpleNamespace(MinFrequency=16.0, LowFrequencyCut=12.0))
    rate = 2048.0
    rng = np.random.default_rng(4)
    w = rng.standard_normal(int(60 * rate))
    w[int(40 * rate):int(40 * rate) + 20] += 30.0
    v = w.copy()
    v[int(50 * rate):] = rng.standard_normal(v.size - int(50 * rate)) * 5

    def run(x):
        c = CausalCensus(s, rate)
        out = []
        for k in range(0, x.size, int(rate)):
            out += [dict(r, at=k) for r in c.push(x[k:k + int(rate)], k / rate)]
        return out

    a, b = run(w), run(v)
    early = [r for r in a if r["at"] < 50 * rate]
    assert early and early == [r for r in b if r["at"] < 50 * rate]
    assert any(r["extent_first"] <= 40.005 <= r["extent_last"] for r in early)
