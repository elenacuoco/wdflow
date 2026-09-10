"""The absolute-time rendering of an event's tiles."""
import numpy as np

from wdf.analysis.wavegram_match import render

BANDS = np.array([[32.0, 64.0], [64.0, 128.0]])


def cloud(*tiles):
    """Tiles as (t_lo, t_hi, f_lo, f_hi, amplitude)."""
    t_lo, t_hi, f_lo, f_hi, amplitude = (np.array(field, dtype=float) for field in zip(*tiles))
    return t_lo, t_hi, f_lo, f_hi, amplitude ** 2, amplitude


def test_the_cells_no_tile_covers_are_exactly_zero():
    # Two tiles of one band meet in one cell: the end of the first and the
    # start of the second land in the same column of the difference array,
    # so the sum adds the first weight, then the second less the first, then
    # subtracts the second -- and 1 + (0.1 - 1) - 0.1 is not zero in floating
    # point. Left in, the round-off trails behind the tiles to the end of
    # the grid, as it did 27 s behind a Virgo event around GW170817.
    grid = render(cloud((0.0, 1.0, 32.0, 64.0, 1.0), (1.0, 2.0, 32.0, 64.0, 0.1)),
                  BANDS, first=0.0, last=10.0, bin_seconds=1.0)
    assert np.flatnonzero(grid[0]).tolist() == [0, 1]
    assert np.all(grid[0, 2:] == 0.0)
    assert np.all(grid[1] == 0.0)


def test_the_covered_cells_carry_the_tiles_density():
    grid = render(cloud((0.0, 0.3, 32.0, 64.0, 0.1), (0.2, 0.5, 32.0, 64.0, 0.7)),
                  BANDS, first=0.0, last=1.0, bin_seconds=0.1)
    first, second = 0.1 / np.sqrt(0.3), 0.7 / np.sqrt(0.3)
    assert np.allclose(grid[0, :2], first)
    assert np.allclose(grid[0, 2:3], first + second)
    assert np.allclose(grid[0, 3:5], second)
    assert np.all(grid[0, 5:] == 0.0)


def test_tiles_of_opposite_sign_that_cancel_leave_nothing_behind():
    grid = render(cloud((0.0, 0.3, 64.0, 128.0, 0.3), (0.0, 0.3, 64.0, 128.0, -0.3),
                        (0.1, 0.2, 64.0, 128.0, 0.1)),
                  BANDS, first=0.0, last=2.0, bin_seconds=0.1)
    assert np.all(grid[1, 3:] == 0.0)
