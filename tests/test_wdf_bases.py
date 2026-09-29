"""The candidate bases of the per-window competition, as a run configures them."""
import pytest

pytest.importorskip("py4tsa")

from _synth import packets_or_skip
from wdf.config.Parameters import Parameters
from wdf.processes.wdf import use_bases, wdf

DEFAULT_BASES = ["Haar", "DaubC4", "DaubC8", "DaubC12", "DaubC16", "DaubC20",
                 "Sym4", "Sym8", "Coif1", "Coif2"]


def search_parameters(**overrides):
    values = dict(window=1024, overlap=256, threshold=0.1, sigma=1.0, Ncoeff=1024)
    values.update(overrides)
    return Parameters(**values)


class CoreWithoutBases:
    """A search as an older py4tsa builds it, with no way to change its bases."""


def test_a_core_without_bases_is_refused_with_the_names_asked_for():
    with pytest.raises(RuntimeError, match="Coif1P6"):
        use_bases(CoreWithoutBases(), ["DaubC8", "Coif1P6"])


def test_an_empty_list_of_bases_is_refused():
    with pytest.raises(ValueError, match="no candidate basis"):
        use_bases(CoreWithoutBases(), [" ", ""])


def test_the_names_reach_the_core_as_one_comma_separated_list():
    class Core:
        def SetBases(self, names):
            self.names = names

    core = Core()
    use_bases(core, ["DaubC8", " Coif1P6,Haar "])
    assert core.names == "DaubC8,Coif1P6,Haar"


def test_without_bases_the_default_pyramids_compete_and_are_recorded():
    packets_or_skip()
    par = search_parameters()
    wdf(par)
    assert par.bases == DEFAULT_BASES


def test_the_bases_given_are_the_ones_that_compete_and_are_recorded():
    packets_or_skip()
    par = search_parameters(bases=["DaubC8", "Coif1P6"])
    search = wdf(par)
    assert search.wdf2classify.GetBases() == "DaubC8,Coif1P6"
    assert par.bases == ["DaubC8", "Coif1P6"]


def test_a_basis_the_core_does_not_know_is_refused():
    packets_or_skip()
    with pytest.raises(ValueError, match="unknown basis"):
        wdf(search_parameters(bases=["Bspline103"]))
