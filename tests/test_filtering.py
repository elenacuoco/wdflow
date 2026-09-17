"""wdf.filtering reproduces scipy 1.17's zero-phase filtering exactly.

scipy 1.18 changed how sosfiltfilt computes its initial conditions, and the
conditioned data moved by a few parts in 1e10 -- enough to shift the golden
triggers, and to make them depend on which scipy is installed. wdf.filtering
does that computation itself. These tests hold it to the numbers scipy 1.17
gave, bit for bit, whatever scipy is present now: if they fail, the triggers
the pipeline emits have changed.
"""
import os

import numpy as np
import pytest

from wdf.filtering import sosfilt_zi, sosfiltfilt

REFERENCE = os.path.join(os.path.dirname(__file__), "fixtures",
                         "sosfiltfilt_scipy117.npz")
FILTERS = ["cheby2_o4", "cheby2_o10", "butter_lp", "butter_hp"]


@pytest.fixture(scope="module")
def ref():
    with np.load(REFERENCE) as data:
        return dict(data)


@pytest.mark.parametrize("name", FILTERS)
def test_the_initial_conditions_are_scipy_117s(ref, name):
    assert np.array_equal(sosfilt_zi(ref[f"{name}_sos"]), ref[f"{name}_zi"])


@pytest.mark.parametrize("name", FILTERS)
@pytest.mark.parametrize("padlen, key", [(None, "y"), (0, "y_pad0"), (50, "y_pad50")])
def test_the_filtered_data_are_scipy_117s(ref, name, padlen, key):
    y = sosfiltfilt(ref[f"{name}_sos"], ref["x"], padlen=padlen)
    assert np.array_equal(y, ref[f"{name}_{key}"])


def test_a_series_no_longer_than_the_padding_is_refused(ref):
    with pytest.raises(ValueError):
        sosfiltfilt(ref["cheby2_o4_sos"], np.zeros(10), padlen=10)
