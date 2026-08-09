import numpy as np
import pytest

from wdf.analysis.pairs import cross_pairs, neighbour_pairs


def _collect(iterator):
    pairs = [np.column_stack(block) for block in iterator]
    if not pairs:
        return np.zeros((0, 2), dtype=int)
    joined = np.concatenate(pairs)
    return joined[np.lexsort((joined[:, 1], joined[:, 0]))]


def _dense_neighbour(time, eps):
    i, j = np.triu_indices(len(time), k=1)
    keep = np.abs(time[i] - time[j]) <= eps
    return np.column_stack([i[keep], j[keep]])


def _dense_cross(left, right, eps):
    i, j = np.nonzero(np.abs(left[:, None] - right[None, :]) <= eps)
    return np.column_stack([i, j])


@pytest.mark.parametrize("eps", [0.0, 0.05, 0.5, 5.0])
def test_neighbour_pairs_match_the_dense_matrix(eps):
    time = np.sort(np.random.default_rng(0).uniform(0, 10, 400))
    assert np.array_equal(_collect(neighbour_pairs(time, eps)),
                          _dense_neighbour(time, eps))


@pytest.mark.parametrize("eps", [0.0, 0.05, 0.5, 5.0])
def test_cross_pairs_match_the_dense_matrix(eps):
    rng = np.random.default_rng(1)
    left = np.sort(rng.uniform(0, 10, 300))
    right = np.sort(rng.uniform(0, 10, 250))
    assert np.array_equal(_collect(cross_pairs(left, right, eps)),
                          _dense_cross(left, right, eps))


def test_pairs_are_yielded_in_blocks_smaller_than_the_run():
    time = np.sort(np.random.default_rng(2).uniform(0, 100, 20000))
    blocks = list(neighbour_pairs(time, 0.05, block=256))
    assert len(blocks) > 1
    assert max(len(left) for left, _ in blocks) < len(time)


def test_a_long_run_costs_its_pairs_and_not_its_square():
    """The dense matrix this replaces would be 10^10 entries here."""
    time = np.arange(100000, dtype=float) * 0.01
    total = sum(len(left) for left, _ in neighbour_pairs(time, 0.045))
    assert total == 4 * len(time) - 10


def test_an_empty_axis_yields_nothing():
    assert _collect(neighbour_pairs(np.zeros(0), 1.0)).shape == (0, 2)
    assert _collect(cross_pairs(np.zeros(0), np.arange(3.0), 1.0)).shape == (0, 2)
    assert _collect(cross_pairs(np.arange(3.0), np.zeros(0), 1.0)).shape == (0, 2)
