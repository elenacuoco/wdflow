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
