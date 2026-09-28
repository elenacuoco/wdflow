"""The worker's whitening filters other than the zero-phase magnitude: the
default square root, and the causal lattice filter as an option."""
import glob
import json

import numpy as np
import pytest

from conftest import TEST_PARAMS, run_segment_process


def used_parameters(outdir):
    with open(glob.glob(f"{outdir}**/parametersUsed-Win*.json", recursive=True)[0]) as fh:
        return json.load(fh)


def record_causal_run(monkeypatch, outdir, **overrides):
    """Run the worker with the causal filter, keeping what the whitening was
    given and what the search was given.

    :return: tuple -- the input blocks and the whitened blocks, each a list of
        ``(start, samples)``, and the run parameters.
    """
    import wdf.processes.wdfUnitDSWorker as W
    from wdf.processes.BandPassDownSampling import SV_to_array

    given, searched = [], []
    library_causal, library_search = W.CausalWhitening, W.wdf

    class Causal(library_causal):
        def Input(self, data):
            given.append((data.GetStart(), SV_to_array(data) * data.GetScale()))
            library_causal.Input(self, data)

    class Search(library_search):
        def SetData(self, data):
            if not searched or searched[-1][0] != data.GetStart():
                searched.append((data.GetStart(), SV_to_array(data)))
            library_search.SetData(self, data)

    monkeypatch.setattr(W, "CausalWhitening", Causal)
    monkeypatch.setattr(W, "wdf", Search)
    run_segment_process(outdir, ZeroPhaseFilter="causal", **overrides)
    return given, searched, used_parameters(outdir)


def test_the_default_is_the_root_at_the_model_order(tmp_outdir):
    """Unset, the whitening is the square root run both ways, of order the
    model's or 256, whichever is more."""
    run_segment_process(tmp_outdir)
    used = used_parameters(tmp_outdir)
    assert used["ZeroPhaseFilter"] == "root"
    assert used["SqrtWhiteningOrder"] == max(256, TEST_PARAMS["ARorder"])
    assert used["ZeroPhaseLatency"] == used["SqrtWhiteningOrder"]


def test_the_causal_stream_is_the_lattice_filter_run_at_once(tmp_outdir, monkeypatch):
    """Block by block, the causal option emits the fitted lattice filter run
    over the whole conditioned stream in one call, and its blocks join without
    a gap or an overlap."""
    from py4tsa.tsa import SeqView_double_t as SV

    from wdf.processes.BandPassDownSampling import SV_to_array
    from wdf.processes.Whitening import Whitening
    from wdf.structures.array2SeqView import array2SeqView

    given, searched, used = record_causal_run(monkeypatch, tmp_outdir)
    rate = used["resampling"]
    assert used["ZeroPhaseLatency"] == 0

    for (start, samples), (following, _) in zip(given, given[1:]):
        assert following == pytest.approx(start + samples.size / rate, abs=1e-9)
    for (start, samples), (following, _) in zip(searched, searched[1:]):
        assert following == pytest.approx(start + samples.size / rate, abs=1e-9)

    stream = np.concatenate([samples for _, samples in given])
    view = array2SeqView(given[0][0], rate, stream.size)
    view.Fill(given[0][0], stream)
    reference = Whitening(used["ARorder"])
    reference.ParametersLoad(used["ARfile"], used["LVfile"])
    out = SV()
    reference.Process(view.SV, out)
    whole = SV_to_array(out) * out.GetScale()

    emitted = np.concatenate([samples for _, samples in searched])
    first = int(round((searched[0][0] - given[0][0]) * rate))
    assert first >= used["preWhite"] * rate
    expected = whole[first:first + emitted.size]
    assert expected.size == emitted.size
    np.testing.assert_allclose(emitted, expected, rtol=0,
                               atol=1e-12 * np.std(expected))


def test_the_causal_warm_up_covers_the_model_and_the_band_pass(tmp_outdir):
    """With no warm-up asked for, the causal filter's is lengthened to the
    model's order and the band-pass's settling."""
    from wdf.config.Parameters import Parameters
    from wdf.processes.BandPassDownSampling import BandPassDownSampling

    run_segment_process(tmp_outdir, ZeroPhaseFilter="causal", preWhite=0)
    used = used_parameters(tmp_outdir)
    par = Parameters()
    for key, value in used.items():
        setattr(par, key, value)
    front = BandPassDownSampling(par)
    past = used["ARorder"] / used["resampling"] + front.padlen / front.sampling
    assert used["preWhite"] == int(np.ceil(past))


def test_the_causal_filter_needs_the_autoregressive_model(tmp_outdir):
    with pytest.raises(ValueError, match="causal"):
        run_segment_process(tmp_outdir, ZeroPhaseFilter="causal",
                            WhiteningModel="spectrum")
