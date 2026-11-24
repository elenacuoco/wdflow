"""The online worker's warm-up: `preWhite` blocks of `len` seconds band-passed,
whitened and discarded before the search starts."""
import numpy as np
import pytest

pytest.importorskip("py4tsa")

from test_online_worker import FS, T0, parameters, strain  # noqa: E402

from wdf.processes.wdfUnitOnline import ArraySource, wdfUnitOnline  # noqa: E402

END = T0 + 60.0
GLITCH_AT = 27.5
MODES = {"causal": {}, "zerophase": dict(WhiteningMode="zerophase", ZeroPhaseDuration=8.0,
                                          ARRefitInterval=0.0)}


def glitched():
    """The test strain with a loud sine-Gaussian 1.5 s after the first model
    takes over, inside a 4-block warm-up."""
    x = strain()
    t = np.arange(x.size) / FS - GLITCH_AT
    return x + 40.0 * np.exp(-(t / 0.02) ** 2) * np.sin(2 * np.pi * 150.0 * t)


def run(mode, pre):
    worker = wdfUnitOnline(parameters(preWhite=pre, **MODES[mode]))
    return worker.process(ArraySource(glitched(), T0, FS), T0 + 26.0, END)


@pytest.fixture(scope="module", params=sorted(MODES))
def pair(request):
    return request.param, run(request.param, 4), run(request.param, 0)


def test_the_search_starts_pre_white_blocks_after_the_whitening(pair):
    """The first searched sample is preWhite * len after the warm-up start, and
    every sample before it was whitened and discarded."""
    mode, warm, plain = pair
    par = warm.parameters
    assert par.preWhite == 4 and par.WarmupSeconds == 4.0
    assert par.WarmupStart == plain.parameters.WarmupStart == plain.parameters.AnalysisStart
    assert par.AnalysisStart == pytest.approx(par.WarmupStart + par.preWhite * par.len)
    assert par.WarmupSamplesDiscarded == int(round(par.WarmupSeconds * par.resampling))
    assert plain.parameters.WarmupSamplesDiscarded == 0
    # the same model whitens both runs
    assert warm.models[0]["switch_gps"] == plain.models[0]["switch_gps"]
    np.testing.assert_array_equal(warm.models[0]["ar"], plain.models[0]["ar"])


def test_the_discarded_blocks_produce_no_triggers(pair):
    """Nothing is searched in the warm-up: the glitch inside it leaves no
    trigger, no census transient and no ridge slot, which it does without the
    warm-up."""
    mode, warm, plain = pair
    start = warm.parameters.AnalysisStart
    assert warm.triggers and all(row["gps"] >= start - 1e-9 for row in warm.triggers)
    assert all(row["first"] >= start - 1e-9 for row in warm.census)
    assert warm.running["slot_end"].size and np.all(warm.running["slot_end"] > start)
    on_glitch = [row for row in plain.triggers
                 if row["gps"] <= T0 + GLITCH_AT <= row["window_end"]]
    assert on_glitch
    assert not [row for row in warm.triggers if row["gps"] <= T0 + GLITCH_AT <= row["window_end"]]


def test_the_warm_up_is_required():
    par = parameters()
    del par.preWhite
    with pytest.raises(ValueError, match="preWhite"):
        wdfUnitOnline(par)
    with pytest.raises(ValueError, match="preWhite"):
        wdfUnitOnline(parameters(preWhite=-1))
    with pytest.raises(ValueError, match="preWhite"):
        wdfUnitOnline(parameters(preWhite=1.5))
