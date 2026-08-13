"""Index pairs of events close in time, without the dense pairwise matrix.

Every stage that joins events -- clustering windows into an event, pairing
events across detectors, wiring a candidate graph -- asks the same question:
which pairs lie within some tolerance in time? Answered with a pairwise
difference matrix that costs O(n^2) memory, which at the event rates a search
without a per-detector threshold produces is tens of gigabytes. Answered by
searching a sorted time axis it costs O(n log n) plus the pairs themselves,
which are few because the tolerance is short compared with the run.

Pairs are yielded in blocks so that a dense stretch of the run does not
allocate one array element per pair of the whole run at once.
"""
from __future__ import annotations

import numpy as np

PAIR_BLOCK = 4096


def _blocks(start, stop, block):
    count = np.maximum(stop - start, 0)
    for begin in range(0, len(start), block):
        end = min(begin + block, len(start))
        piece = count[begin:end]
        total = int(piece.sum())
        if total == 0:
            continue
        left = np.repeat(np.arange(begin, end), piece)
        offset = np.repeat(np.cumsum(piece) - piece, piece)
        right = np.arange(total) - offset + np.repeat(start[begin:end], piece)
        yield left, right


def neighbour_pairs(time, time_eps, block: int = PAIR_BLOCK):
    """Yield index pairs of events no further apart in time than `time_eps`.

    :param time: sorted event times, seconds.
    :param time_eps: largest separation that still pairs two events, seconds.
    :param block: how many left indices to expand per yielded block.
    :return: iterator of (left, right) integer index arrays, left < right.
    """
    time = np.asarray(time, dtype=float)
    start = np.arange(len(time)) + 1
    stop = np.searchsorted(time, time + time_eps, side="right")
    yield from _blocks(start, stop, block)


def cross_pairs(left_time, right_time, time_eps, block: int = PAIR_BLOCK):
    """Yield index pairs drawn one from each of two sorted time axes.

    :param left_time: sorted event times of the first set, seconds.
    :param right_time: sorted event times of the second set, seconds.
    :param time_eps: largest separation that still pairs two events, seconds.
    :param block: how many left indices to expand per yielded block.
    :return: iterator of (left, right) integer index arrays, indexing
        `left_time` and `right_time` respectively.
    """
    left_time = np.asarray(left_time, dtype=float)
    right_time = np.asarray(right_time, dtype=float)
    start = np.searchsorted(right_time, left_time - time_eps, side="left")
    stop = np.searchsorted(right_time, left_time + time_eps, side="right")
    yield from _blocks(start, stop, block)


def interval_pairs(left_lo, left_hi, right_anchor, right_reach=0.0,
                   block: int = PAIR_BLOCK):
    """Yield index pairs whose intervals can touch, sized by each interval.

    Enumerates pairs ``(i, j)`` with ``right_anchor[j]`` inside
    ``[left_lo[i] - right_reach, left_hi[i]]`` on a sorted right axis. With
    `right_reach` the longest extent any right event adds beyond its anchor,
    this is a superset of every pair of touching intervals in which each left
    event sweeps a window set by its own extent: one long event widens its
    own enumeration and nobody else's. A single global window --- the longest
    extent either set holds --- multiplies that one event's reach onto every
    event of both sets, which is quadratic in all but name.

    :param left_lo: lower edge of each left interval, seconds. Any order.
    :param left_hi: upper edge of each left interval, seconds.
    :param right_anchor: sorted anchor of each right event, seconds.
    :param right_reach: how far beyond its anchor a right event can extend,
        seconds.
    :param block: how many left indices to expand per yielded block.
    :return: iterator of (left, right) integer index arrays, indexing
        `left_lo` and `right_anchor` respectively.
    """
    left_lo = np.asarray(left_lo, dtype=float)
    left_hi = np.asarray(left_hi, dtype=float)
    right_anchor = np.asarray(right_anchor, dtype=float)
    start = np.searchsorted(right_anchor, left_lo - float(right_reach),
                            side="left")
    stop = np.searchsorted(right_anchor, left_hi, side="right")
    yield from _blocks(start, stop, block)


# Rows gathered per pair are the memory wall of every pairwise stage. A dot
# product between the maps of two events is one number, but writing it as
# `(maps[i] * maps[j]).sum(1)` first builds one copy of a map per pair: at a
# few million pairs and a map of a few thousand cells that is tens of
# gigabytes, on any device. The rows are therefore gathered a block at a time,
# and only the numbers survive the block.
PAIRED_BLOCK = 200_000


def _paired_device(prefer_gpu):
    """The device the paired reductions run on, or None to stay in numpy.

    :type prefer_gpu: bool | None
    :param prefer_gpu: True to require a GPU, False to refuse one, None to use
        one when it is there.
    :return: torch.device | None
    :raises RuntimeError: if a GPU is required and none is usable.
    """
    if prefer_gpu is False:
        return None
    try:
        import torch
    except ImportError:
        if prefer_gpu:
            raise RuntimeError("a GPU was asked for but torch is not installed")
        return None
    if not torch.cuda.is_available():
        if prefer_gpu:
            raise RuntimeError("a GPU was asked for but CUDA is not available")
        return None
    return torch.device("cuda")


def paired_dot(left, right, i, j, block: int = PAIRED_BLOCK, gpu=None):
    """Row-wise dot products of gathered rows, without gathering them all.

    Computes ``sum(left[i] * right[j], axis=1)`` for every pair, in blocks, so
    that the memory held is one block of rows rather than one row per pair.

    The reduction runs on a GPU when one is present, in double precision. The
    result then differs from the numpy one only in the order the products are
    summed, at the level of floating-point rounding; it is not bit-identical
    and is not meant to be. What is identical is the arithmetic asked for.

    :type left: numpy.ndarray
    :param left: (n_rows, n_features) matrix the first member is taken from.
    :type right: numpy.ndarray
    :param right: (n_rows, n_features) matrix the second member is taken from.
    :type i: numpy.ndarray
    :param i: row of `left` for each pair.
    :type j: numpy.ndarray
    :param j: row of `right` for each pair.
    :type block: int
    :param block: pairs reduced at once. Bounds the memory whatever the device.
    :type gpu: bool | None
    :param gpu: True to require a GPU, False to stay in numpy, None to use one
        when available.
    :return: numpy.ndarray -- one value per pair.
    :raises ValueError: if the two matrices have different widths, or the two
        index arrays different lengths.
    """
    left = np.asarray(left)
    right = np.asarray(right)
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError(
            f"matrices must be 2-D and equally wide, got {left.shape} and "
            f"{right.shape}")
    if i.shape != j.shape:
        raise ValueError(f"index arrays differ: {i.shape} and {j.shape}")
    n = i.size
    out = np.empty(n, dtype=np.float64)
    if n == 0 or left.shape[1] == 0:
        return np.zeros(n, dtype=np.float64)

    device = _paired_device(gpu)
    if device is None:
        for start in range(0, n, block):
            stop = min(start + block, n)
            out[start:stop] = np.einsum("ij,ij->i", left[i[start:stop]],
                                        right[j[start:stop]])
        return out

    import torch
    # Both matrices are resident for the whole reduction: they are indexed by
    # every block, so sending them once is what makes the blocks cheap.
    tl = torch.as_tensor(left, device=device, dtype=torch.float64)
    tr = (tl if right is left
          else torch.as_tensor(right, device=device, dtype=torch.float64))
    ti = torch.as_tensor(i, device=device)
    tj = torch.as_tensor(j, device=device)
    for start in range(0, n, block):
        stop = min(start + block, n)
        piece = (tl[ti[start:stop]] * tr[tj[start:stop]]).sum(dim=1)
        out[start:stop] = piece.cpu().numpy()
    return out
