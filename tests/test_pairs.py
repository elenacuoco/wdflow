import numpy as np
import pytest

from wdf.analysis.pairs import cross_pairs, neighbour_pairs, paired_dot


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


def test_a_node_quantity_is_not_a_pair_quantity():
    """`paired_dot(a, a, i, i)` is each row's own dot product, which belongs to
    the node. Computing it per pair repeats one row's reduction once for every
    pair the row belongs to, and the network stage forms tens of millions."""
    rng = np.random.default_rng(3)
    raw = rng.standard_normal((200, 17))
    i = rng.integers(0, 200, 5000)
    per_pair = paired_dot(raw, raw, i, i, gpu=False)
    per_node = np.einsum("ij,ij->i", raw, raw)[i]
    assert np.allclose(per_pair, per_node, rtol=0, atol=1e-12)


def test_a_matrix_kept_on_the_device_gives_the_same_answer():
    """The residency is an optimisation and must not be a change of arithmetic.
    Without a GPU the mapping is ignored, which is what makes the same code run
    on a machine that has none."""
    rng = np.random.default_rng(4)
    left = rng.standard_normal((300, 11))
    right = rng.standard_normal((300, 11))
    i = rng.integers(0, 300, 900)
    j = rng.integers(0, 300, 900)

    plain = paired_dot(left, right, i, j)
    held: dict = {}
    once = paired_dot(left, right, i, j, resident=held)
    twice = paired_dot(left, right, i, j, resident=held)
    assert np.allclose(plain, once, rtol=0, atol=1e-10)
    assert np.array_equal(once, twice)

    # A different matrix at the same identity slot is not served the old copy.
    other = rng.standard_normal((300, 11))
    assert not np.allclose(paired_dot(other, right, i, j, resident=held),
                           once, rtol=0, atol=1e-10)
