"""Modern (HDF5) on-disk format for the AR-Burg / lattice-filter whitening
coefficients estimated once per segment and reused across
Whitening/DWhitening/Coloring.

Replaces p4TSA's `ArBurgEstimator.Save`/`.Load` and `LatticeView.Save`/`.Load`
-- both go through `eternity::xml_archive`, an old, verbose, uncompressed XML
serializer (their own `fmt` argument is actually ignored: it is always XML,
regardless of what is passed). An AR file lands as verbose ASCII text; the
equivalent HDF5 file below holds the same coefficients as binary arrays, and
loads without a bespoke XML parser.

Reads/writes state through `ArBurgEstimator`/`LatticeView`'s own existing
get/set accessors (`GetAR`/`SetAR`, `GetOrder`/`SetOrder`,
`GetParcorF`/`SetParcorF`, etc.) -- the same fields their own
`xml_serialize` persists (confirmed against `ArBurgEstimator.hpp`/
`LatticeView.hpp`: just `mArOrder`+`mAR` for the former, `mOrder`+
`mErrorForward`+`mErrorBackward`+`mParcorF`+`mParcorB` for the latter). No
p4TSA/C++ changes needed.

One subtlety: `LatticeView.GetErrorForward`/`GetErrorBackward` are indexed by
(row, j) with 2 rows, but `SetErrorForward`/`SetErrorBackward` take only `j`
and set both rows to the same value, so the two rows cannot differ. Only row 0
is persisted; `test_whitening_persistence.py` checks the invariant on a fitted
view rather than leaving it stated here.
"""
from __future__ import annotations

import h5py
import numpy as np


def save_ar_burg(h5path: str, ade) -> None:
    """Writes an ArBurgEstimator's state (`mArOrder`, `mAR`) to `h5path`.

    :type h5path: str
    :param h5path: output file path.
    :type ade: pytsa.tsa.ArBurgEstimator
    :param ade: the estimator to persist.
    """
    order = ade.GetArOrder()
    ar = np.array([ade.GetAR(j) for j in range(order + 1)])
    with h5py.File(h5path, "w") as fh:
        fh.attrs["ar_order"] = order
        fh.create_dataset("ar", data=ar, compression="gzip")


def load_ar_burg(h5path: str, ade) -> None:
    """Populates an ArBurgEstimator in place from `h5path` (see `save_ar_burg`).

    :type h5path: str
    :param h5path: input file path.
    :type ade: pytsa.tsa.ArBurgEstimator
    :param ade: the estimator to populate (mutated in place).
    """
    with h5py.File(h5path, "r") as fh:
        order = int(fh.attrs["ar_order"])
        ar = fh["ar"][()]
    ade.SetArOrder(order)
    for j, v in enumerate(ar):
        ade.SetAR(j, float(v))


def save_lattice_view(h5path: str, lv) -> None:
    """Writes a LatticeView's state (`mOrder`, `mErrorForward`,
    `mErrorBackward`, `mParcorF`, `mParcorB`) to `h5path`.

    :type h5path: str
    :param h5path: output file path.
    :type lv: pytsa.tsa.LatticeView
    :param lv: the lattice view to persist.
    """
    order = lv.GetOrder()
    n = order + 1
    # row 0 only -- see module docstring, row 1 is always identical.
    error_forward = np.array([lv.GetErrorForward(0, j) for j in range(n)])
    error_backward = np.array([lv.GetErrorBackward(0, j) for j in range(n)])
    parcor_f = np.array([lv.GetParcorF(j) for j in range(n)])
    parcor_b = np.array([lv.GetParcorB(j) for j in range(n)])
    with h5py.File(h5path, "w") as fh:
        fh.attrs["order"] = order
        fh.create_dataset("error_forward", data=error_forward, compression="gzip")
        fh.create_dataset("error_backward", data=error_backward, compression="gzip")
        fh.create_dataset("parcor_f", data=parcor_f, compression="gzip")
        fh.create_dataset("parcor_b", data=parcor_b, compression="gzip")


def load_lattice_view(h5path: str, lv) -> None:
    """Populates a LatticeView in place from `h5path` (see `save_lattice_view`).

    :type h5path: str
    :param h5path: input file path.
    :type lv: pytsa.tsa.LatticeView
    :param lv: the lattice view to populate (mutated in place).
    """
    with h5py.File(h5path, "r") as fh:
        order = int(fh.attrs["order"])
        error_forward = fh["error_forward"][()]
        error_backward = fh["error_backward"][()]
        parcor_f = fh["parcor_f"][()]
        parcor_b = fh["parcor_b"][()]
    lv.SetOrder(order)
    for j in range(order + 1):
        lv.SetErrorForward(j, float(error_forward[j]))
        lv.SetErrorBackward(j, float(error_backward[j]))
        lv.SetParcorF(j, float(parcor_f[j]))
        lv.SetParcorB(j, float(parcor_b[j]))
