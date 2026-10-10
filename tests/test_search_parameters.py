"""The search reads its rule, its block rule and its name from the parameters."""
from types import SimpleNamespace

import pytest

pytest.importorskip("py4tsa")

from wdf.analysis.io import (parameters_file_name, run_parameters, search_suffix,
                             trigger_file_name)
from wdf.config.Parameters import Parameters, threshold_rule
from wdf.processes.wdf import wdf
from wdf.processes.wdfUnitDSWorker import search_names


def search_par(**kw):
    par = Parameters(window=1024, overlap=256, threshold=5.0, sigma=1.0, Ncoeff=1024,
                     resampling=2048.0)
    par.__dict__.update(kw)
    return par


def test_the_threshold_rule_is_a_parameter():
    from py4tsa.tsa import WaveletThreshold

    assert threshold_rule(SimpleNamespace()) == WaveletThreshold.block
    assert threshold_rule(SimpleNamespace(ThresholdRule="dohonojohnston")) == WaveletThreshold.dohonojohnston
    with pytest.raises(ValueError, match="ThresholdRule"):
        threshold_rule(SimpleNamespace(ThresholdRule="sure"))
    par = search_par(ThresholdRule="dohonojohnston")
    wdf(par)
    assert par.ThresholdRule == "dohonojohnston"


def test_the_block_rule_is_set_and_recorded():
    par = search_par()
    search = wdf(par)
    assert (par.BlockLength, par.BlockLambda) == (7, pytest.approx(4.505))
    par = search_par(BlockLength=9, BlockLambda=6.0)
    search = wdf(par)
    assert search.wdf2classify.GetBlockLength() == 9
    assert (par.BlockLength, par.BlockLambda) == (9, 6.0)


def test_the_block_length_is_checked_against_the_bases():
    with pytest.raises(ValueError):
        wdf(search_par(BlockLength=16, Bases="DaubC8,LocalCos128"))


def test_a_search_name_separates_the_files(tmp_path):
    base = dict(channel="H1:X", gps=1.0e9, ARorder=3000, window=1024, overlap=256, threshold=5.0)
    plain = SimpleNamespace(**base)
    named = SimpleNamespace(**base, SearchName="online")
    assert search_suffix(plain) == ""
    assert trigger_file_name(plain) != trigger_file_name(named)
    assert parameters_file_name(named) == "parametersUsed-Win1024-Searchonline.json"
    with pytest.raises(ValueError):
        search_suffix(SimpleNamespace(SearchName="a b"))
    for par in (plain, named):
        Parameters(**vars(par)).dump(str(tmp_path / parameters_file_name(par)))
        found = run_parameters(str(tmp_path / trigger_file_name(par)))
        assert getattr(found, "SearchName", None) == getattr(par, "SearchName", None)


def test_repeated_windows_get_their_own_names():
    assert search_names([(1024, 256), (512, 64)]) == [(1024, 256, None), (512, 64, None)]
    assert search_names([(1024, 256), (1024, 0)], "D0") == [(1024, 256, "D00"), (1024, 0, "D01")]
