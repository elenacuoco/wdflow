"""The shipped configurations: complete, commented, read by the code, and
building their workers."""
import glob
import os
import re

import pytest

from wdf.config.Parameters import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = {name: os.path.join(ROOT, "configs", f"{name}_default.toml") for name in ("online", "online_zerophase", "offline")}
SOURCE = "\n".join(open(p).read() for p in glob.glob(os.path.join(ROOT, "wdf", "**", "*.py"),
                                                     recursive=True))
#: The 10 wavelet bases of the shipped configurations.
TEN_BASES = ["Haar", "DaubC4", "DaubC8", "Sym4", "DaubC16", "Sym8", "Coif3", "DaubC24", "Sym12",
             "Coif5"]
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
    assert h1.Bases == TEN_BASES
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


def test_online_zerophase_is_online_with_the_offline_whitening():
    """The same file as online_default.toml except the whitening, the band-pass
    phase and the names."""
    on, zp = load_config(CONFIGS["online"], "H1"), load_config(CONFIGS["online_zerophase"], "H1")
    off = load_config(CONFIGS["offline"], "H1")
    differ = {"ConfigFile", "SearchName", "run", "BandPassPhase", "WhiteningMode", "ZeroPhaseFilter",
              "ZeroPhaseDuration", "HoldOutsideBand"}
    a, b = vars(on), vars(zp)
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == differ
    assert zp.WhiteningMode == "zerophase" and zp.BandPassPhase == off.BandPassPhase == "zero"
    for key in ("ZeroPhaseFilter", "ZeroPhaseDuration", "HoldOutsideBand"):
        assert getattr(zp, key) == getattr(off, key)
    assert load_config(CONFIGS["online_zerophase"], "V1").ZeroPhaseDuration == 32.0


def test_the_online_zerophase_latency_at_2048_hz():
    pytest.importorskip("py4tsa")
    from wdf.processes.wdfUnitOnline import wdfUnitOnline

    for sampling in (4096, 16384):
        par = load_config(CONFIGS["online_zerophase"], "H1", sampling=sampling,
                          ResamplingFactor=sampling // 2048, channel="H1:X", file="x.gwf")
        used = wdfUnitOnline(par).par
        assert (used.WhiteningLookahead_s, used.BandPassLookahead_s, used.Latency_s) == (8.0, 2.0, 10.0)


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
    worker = (wdfUnitOnline if name.startswith("online") else wdfUnitOffline)(par)
    assert worker.par.SearchName == name


def build(name, **kw):
    from wdf.processes.wdfUnitOffline import wdfUnitOffline
    from wdf.processes.wdfUnitOnline import wdfUnitOnline

    par = load_config(CONFIGS[name], "H1", sampling=4096, channel="H1:X", file="x.gwf", **kw)
    return (wdfUnitOnline if name.startswith("online") else wdfUnitOffline)(par)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_ten_bases_remain_reachable(name):
    pytest.importorskip("py4tsa")
    assert build(name, Bases=TEN_BASES).par.Bases == TEN_BASES


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_a_local_cosine_beyond_the_block_cap_is_refused_before_running(name):
    """LocalCos128 needs window / 128 >= L: 1024 (L 7) passes, 512 (L 6) does
    not, and the error names Bases, the window and the largest M allowed."""
    pytest.importorskip("py4tsa")
    with pytest.raises(ValueError, match=r"Bases holds 'LocalCos128'.*window 512.*LocalCos64"):
        build(name, window=512, overlap=128, Bases=TEN_BASES + ["LocalCos128"])
    with pytest.raises(ValueError, match=r"Bases holds 'LocalCos256'.*window 1024.*LocalCos128"):
        build(name, Bases=TEN_BASES + ["LocalCos256"])
    build(name, window=512, overlap=128, Bases=TEN_BASES + ["LocalCos64"])
    build(name, window=512, overlap=128, ThresholdRule="dohonojohnston")


@pytest.mark.parametrize("window, basis", [(1024, "LocalCos128"), (1024, "LocalCos256"),
                                           (512, "LocalCos128"), (512, "LocalCos64"),
                                           (2048, "LocalCos256"), (1024, "HaarP4")])
def test_the_check_agrees_with_the_core(window, basis):
    """`check_bases` refuses exactly what p4TSA's SetBases refuses."""
    pytest.importorskip("py4tsa")
    from wdf.config.Parameters import Parameters, check_bases
    from wdf.processes.wdf import wdf

    par = Parameters(window=window, overlap=window // 4, Bases=["Haar", basis],
                     ThresholdRule="block", BlockLength=0, threshold=5.0, sampling=2048,
                     resampling=2048, Ncoeff=window, MinFrequency=16.0, sigma=1.0)
    try:
        wdf(par)
        core = True
    except ValueError:
        core = False
    try:
        check_bases(par, [(window, window // 4)])
        ours = True
    except ValueError:
        ours = False
    assert ours == core
