"""wdf.filtering against reference arrays, bit for bit.

The references are written with scipy 1.17 -- see
fixtures/make_sosfiltfilt_reference.py.
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
def test_the_initial_conditions_match(ref, name):
    assert np.array_equal(sosfilt_zi(ref[f"{name}_sos"]), ref[f"{name}_zi"])


@pytest.mark.parametrize("name", FILTERS)
@pytest.mark.parametrize("padlen, key", [(None, "y"), (0, "y_pad0"), (50, "y_pad50")])
def test_the_filtered_data_match(ref, name, padlen, key):
    y = sosfiltfilt(ref[f"{name}_sos"], ref["x"], padlen=padlen)
    assert np.array_equal(y, ref[f"{name}_{key}"])


def test_a_series_no_longer_than_the_padding_is_refused(ref):
    with pytest.raises(ValueError):
        sosfiltfilt(ref["cheby2_o4_sos"], np.zeros(10), padlen=10)
