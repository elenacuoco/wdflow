"""The shipped configurations: complete, commented, read by the code, and
building their workers."""
import glob
import os
import re

import pytest

from wdf.config.Parameters import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = {name: os.path.join(ROOT, "configs", f"{name}_default.toml") for name in ("online", "offline")}
SOURCE = "\n".join(open(p).read() for p in glob.glob(os.path.join(ROOT, "wdf", "**", "*.py"),
                                                     recursive=True))
#: Keys the configurations set that only the run layout reads.
LAYOUT = {"run", "outdir"}


def assignments(path):
    for line in open(path):
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if m:
            yield m.group(1), line


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_every_value_is_commented_and_read_by_the_code(name):
    for key, line in assignments(CONFIGS[name]):
        assert "#" in line, f"{name}: {key} has no comment"
        if key not in LAYOUT:
            assert re.search(rf"[\"']{key}[\"']|\.{key}\b", SOURCE), f"{name}: nothing reads {key}"


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_detector_tables_apply(name):
    h1 = load_config(CONFIGS[name], "H1")
    v1 = load_config(CONFIGS[name], "V1")
    assert (h1.ARorder, v1.ARorder) == (3000, 16000)
    assert h1.itf == "H1" and v1.learn == 600
    assert h1.Bases == ["Haar", "DaubC4", "DaubC8", "Sym4", "DaubC16", "Sym8", "Coif3", "DaubC24",
                        "Sym12", "Coif5"]
    assert (h1.window, h1.overlap, h1.threshold, h1.MinFrequency, h1.BlockRemainder) == \
        (1024, 256, 5.0, 16.0, "merge")
    with pytest.raises(KeyError):
        load_config(CONFIGS[name], "K1")


def test_the_modes_differ_where_they_must():
    on, off = load_config(CONFIGS["online"], "H1"), load_config(CONFIGS["offline"], "H1")
    assert on.BandPassPhase == "causal" and off.BandPassPhase == "zero"
    assert off.ZeroPhaseFilter == "truncated" and off.ZeroPhaseDuration == 16.0
    assert off.GlitchSubtraction is True and off.GlitchProtectCoincident is True
    assert not hasattr(on, "GlitchSubtraction") and not hasattr(on, "Gates")
    assert load_config(CONFIGS["offline"], "V1").ZeroPhaseDuration == 32.0


@pytest.mark.parametrize("sampling, order", [(4096, 8), (16384, 12)])
def test_the_band_pass_follows_the_rate(sampling, order):
    pytest.importorskip("py4tsa")
    from wdf.processes.BandPassDownSampling import BandPassDownSampling

    par = load_config(CONFIGS["offline"], "H1", sampling=sampling,
                      ResamplingFactor=sampling // 2048)
    par.resampling = 2048
    filt = BandPassDownSampling(par)
    assert filt.cutoff_frequency == pytest.approx(900.0) and filt.lowpass_order == order


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_workers_build(name):
    pytest.importorskip("py4tsa")
    from wdf.processes.wdfUnitOffline import wdfUnitOffline
    from wdf.processes.wdfUnitOnline import wdfUnitOnline

    par = load_config(CONFIGS[name], "L1", sampling=4096, channel="L1:X", file="x.gwf")
    worker = (wdfUnitOnline if name == "online" else wdfUnitOffline)(par)
    assert worker.par.SearchName == name
