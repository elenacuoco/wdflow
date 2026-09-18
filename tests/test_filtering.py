"""wdf.filtering against reference arrays.

The references are written with scipy 1.17 -- see
fixtures/make_sosfiltfilt_reference.py. The bound is 1e-9, not equality: the
initial conditions are solved through LAPACK, whose last bits follow the BLAS
kernel the machine runs. It is still two orders of magnitude below the float32
the triggers are stored in, and the golden output is what pins those.
"""
import os

import numpy as np
import pytest

from wdf.filtering import sosfilt_zi, sosfiltfilt

REFERENCE = os.path.join(os.path.dirname(__file__), "fixtures",
                         "sosfiltfilt_scipy117.npz")
FILTERS = ["cheby2_o4", "cheby2_o10", "butter_lp", "butter_hp"]
RTOL = 1e-9


def close(actual, expected):
    """Relative to the largest value, so that near-zero entries, whose own
    relative error is unbounded, are held to the same absolute bound."""
    scale = float(np.abs(expected).max())
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=RTOL * scale)


@pytest.fixture(scope="module")
def ref():
    with np.load(REFERENCE) as data:
        return dict(data)


@pytest.mark.parametrize("name", FILTERS)
def test_the_initial_conditions_match(ref, name):
    close(sosfilt_zi(ref[f"{name}_sos"]), ref[f"{name}_zi"])


@pytest.mark.parametrize("name", FILTERS)
@pytest.mark.parametrize("padlen, key", [(None, "y"), (0, "y_pad0"), (50, "y_pad50")])
def test_the_filtered_data_match(ref, name, padlen, key):
    close(sosfiltfilt(ref[f"{name}_sos"], ref["x"], padlen=padlen),
          ref[f"{name}_{key}"])


def test_a_series_no_longer_than_the_padding_is_refused(ref):
    with pytest.raises(ValueError):
        sosfiltfilt(ref["cheby2_o4_sos"], np.zeros(10), padlen=10)
