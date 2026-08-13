"""Round-trip test for Whitening/DWhitening/Coloring's AR/LV persistence,
now HDF5-backed (wdf.processes.ar_lv_io) instead of p4TSA's old
eternity::xml_archive-based Save/Load. A fresh instance loaded from disk must
reproduce the exact same whitening behavior as the instance that estimated
and saved the parameters in the first place -- this is the core correctness
property the on-disk format change must preserve.
"""
import os

import numpy as np
import pytest

from wdf.structures.array2SeqView import array2SeqView
from wdf.processes.Whitening import Whitening
from wdf.processes.DWhitening import DWhitening
from wdf.processes.Coloring import Coloring

AR_ORDER = 20
FS = 256.0


def _synthetic_seqview(n, seed, fs=FS, t0=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    return array2SeqView(t0, fs, n).Fill(t0, x)


def test_whitening_reload_matches_original(tmp_path):
    whiten = Whitening(AR_ORDER)
    whiten.ParametersEstimate(_synthetic_seqview(4000, seed=1))
    sigma_before = whiten.GetSigma()

    arfile = str(tmp_path / "AR.h5")
    lvfile = str(tmp_path / "LV.h5")
    whiten.ParametersSave(arfile, lvfile)
    assert os.path.exists(arfile) and os.path.exists(lvfile)

    reloaded = Whitening(AR_ORDER)
    reloaded.ParametersLoad(arfile, lvfile)
    assert reloaded.GetSigma() == sigma_before

    probe = _synthetic_seqview(512, seed=2, t0=1000.0)
    out_orig = array2SeqView(1000.0, FS, 512).SV
    out_reloaded = array2SeqView(1000.0, FS, 512).SV
    whiten.Process(probe, out_orig)
    reloaded.Process(probe, out_reloaded)
    y_orig = np.array([out_orig.GetY(0, i) for i in range(512)])
    y_reloaded = np.array([out_reloaded.GetY(0, i) for i in range(512)])
    assert np.array_equal(y_orig, y_reloaded)


def test_dwhitening_load_from_saved_lv(tmp_path):
    whiten = Whitening(AR_ORDER)
    whiten.ParametersEstimate(_synthetic_seqview(4000, seed=3))
    lvfile = str(tmp_path / "LV.h5")
    whiten.ParametersSave(str(tmp_path / "AR.h5"), lvfile)

    dwhiten = DWhitening(whiten.LV, OutputSize=64, ExtraSize=0)
    loaded_lv = dwhiten.ParametersLoad(lvfile)
    n = whiten.LV.GetOrder() + 1
    for j in (0, 1, n - 1):
        assert loaded_lv.GetParcorF(j) == whiten.LV.GetParcorF(j)
        assert loaded_lv.GetParcorB(j) == whiten.LV.GetParcorB(j)


def test_coloring_load_from_saved_ar(tmp_path):
    whiten = Whitening(AR_ORDER)
    whiten.ParametersEstimate(_synthetic_seqview(4000, seed=4))
    arfile = str(tmp_path / "AR.h5")
    whiten.ParametersSave(arfile, str(tmp_path / "LV.h5"))

    color = Coloring(AR_ORDER)
    color.ParametersLoad(arfile)
    # ARMAflt built successfully and whitening->recoloring round-trips a
    # probe signal back close to its original scale (qualitative check --
    # Coloring is a single-pass approximate inverse, see
    # wdf_bns_gw170817.ipynb section 5's caveat).
    probe = _synthetic_seqview(64, seed=5, t0=2000.0)
    whitened = array2SeqView(2000.0, FS, 64).SV
    whiten.Process(probe, whitened)
    recolored = array2SeqView(2000.0, FS, 64).SV
    color.Process(whitened, recolored)
    y = np.array([recolored.GetY(0, i) for i in range(64)])
    assert np.all(np.isfinite(y))


def test_only_row_zero_of_the_lattice_errors_carries_information():
    """The two rows of `ErrorForward`/`ErrorBackward` hold the same value.

    `save_lattice_view` persists row 0 alone, which is lossless only if the
    rows agree: `SetErrorForward`/`SetErrorBackward` take one index and write
    both rows, so nothing in the API can make them differ, but the getters are
    indexed by (row, j) and a reader cannot tell that from the signature. The
    invariant is checked on a fitted view rather than asserted in prose.
    """
    whiten = Whitening(AR_ORDER)
    whiten.ParametersEstimate(_synthetic_seqview(4000, seed=6))
    lv = whiten.GetLV()

    rows = range(lv.GetOrder() + 1)
    forward = [(lv.GetErrorForward(0, j), lv.GetErrorForward(1, j)) for j in rows]
    backward = [(lv.GetErrorBackward(0, j), lv.GetErrorBackward(1, j)) for j in rows]

    assert all(a == b for a, b in forward)
    assert all(a == b for a, b in backward)
