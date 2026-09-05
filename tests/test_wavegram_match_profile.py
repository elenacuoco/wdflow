import numpy as np

from wdf.analysis.wavegram_match import correlation_profile, correlation_profiles


def _at(lags, value):
    """Index of `value` on a lag axis, to the precision the axis is built at."""
    return int(np.flatnonzero(np.isclose(lags, value))[0])


def test_batched_profile_is_on_magnitudes_over_the_admitted_lags():
    left = np.zeros((2, 5))
    left[0, 2] = 2.0
    left[1, 3] = -1.0
    right = left.copy()
    single, lags, *_ = correlation_profile(left, right, 1.0, 1.0)
    batched, batch_lags, residual = correlation_profiles(
        left[None], right[None], [0], [0], [1.0], 1.0)
    assert batched.shape == (1, 2, len(batch_lags))
    # Two events on one instant have no anchor difference to split.
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)
    shared = np.flatnonzero(np.isin(np.round(batch_lags, 9), np.round(lags, 9)))
    np.testing.assert_allclose(batched[0][:, shared], single, atol=1e-12)
    zero = _at(batch_lags, 0.0)
    assert batched[0, 0, zero] > 0.0
    assert batched[0, 1, zero] > 0.0
    # A tolerance of zero admits one displacement, and the axis carries nothing
    # at the others: an inadmissible lag is not a small agreement, it is none.
    narrow, narrow_lags, _ = correlation_profiles(
        left[None], right[None], [0], [0], [0.0], 1.0)
    inadmissible = np.ones(len(narrow_lags), dtype=bool)
    inadmissible[_at(narrow_lags, 0.0)] = False
    assert narrow[0, 0, _at(narrow_lags, 0.0)] > 0.0
    assert not np.any(narrow[0][:, inadmissible])
    # The comparison is on magnitudes, so the polarity of a map is not a
    # disagreement: it says where the source sits with respect to the two
    # detectors, and what this measures is the shape.
    flipped, _, _ = correlation_profiles(
        left[None], -right[None], [0], [0], [0.0], 1.0)
    np.testing.assert_allclose(flipped, narrow, atol=1e-12)


def test_the_lag_axis_reaches_every_displacement_the_pair_admits():
    """Anchoring on each event does not shrink what the coincidence may test.

    A pair whose two anchors already differ by the whole tolerance reaches zero
    displacement only by sliding one map back by that much. The whole bins of
    the anchor difference are applied as a shift of the map, so what the axis
    has to span is the tolerance itself and not the tolerance plus however far
    apart the anchors happen to be.
    """
    bin_seconds, tolerance = 0.002, 0.025
    maps = np.zeros((2, 1, 60))
    maps[:, 0, 30] = 1.0
    _, lags, residual = correlation_profiles(maps, maps, [0], [1], [tolerance],
                                             bin_seconds,
                                             offset_s=np.array([tolerance]))
    absolute = residual[0] + lags
    assert absolute.min() <= 0.0 <= absolute.max()
    assert np.all(np.abs(absolute) <= tolerance + bin_seconds)
    # And the axis is the tolerance's, not the anchor difference's: it does not
    # grow when the two anchors are moved further apart.
    _, far, _ = correlation_profiles(maps, maps, [0], [1], [tolerance],
                                     bin_seconds, offset_s=np.array([2.0]))
    assert len(far) == len(lags)


def test_two_events_sharing_no_sample_agree_at_no_displacement():
    """Maps narrower than the shift between their anchors overlap in nothing.

    An agreement of zero is then the measurement --- it is what two events
    sharing no sample are worth --- and not an axis the reduction cannot index.
    """
    profiles, lags, _ = correlation_profiles(
        np.ones((2, 1, 3)), np.ones((2, 1, 3)), [0], [1], [0.001], 1.0,
        offset_s=np.array([100.0]))
    assert profiles.shape == (1, 1, len(lags))
    assert not np.any(profiles)


def test_the_norm_is_the_event_s_and_not_the_edge_s():
    """One event used in many edges is normalised by one number.

    The norm of a map is a property of the event. Taking it per edge gathers
    one map per pair, which is the memory this stage cannot afford, and the
    cheap form must agree with the dense one exactly.
    """
    rng = np.random.default_rng(0)
    maps = rng.standard_normal((4, 2, 9))
    i, j = [0, 0, 1, 2], [1, 2, 3, 3]
    profiles, lags, _ = correlation_profiles(maps, maps, i, j, np.full(4, 2.0), 1.0)
    flat = maps.reshape(len(maps), -1)
    at = _at(lags, 0.0)
    expected = [float(np.abs(flat[a]) @ np.abs(flat[b])
                      / (np.linalg.norm(flat[a]) * np.linalg.norm(flat[b])))
                for a, b in zip(i, j)]
    np.testing.assert_allclose(profiles[:, :, at].sum(axis=1), expected, atol=1e-12)


def test_the_whole_bin_shift_is_bookkeeping_and_not_a_different_measure():
    """Splitting the anchor difference does not change what is compared.

    Declaring two events' anchors `o` apart and asking for the absolute
    displacement `D` must slide one map exactly as far as declaring them
    together and asking for `D - o`: the shift applied to the map is the same,
    so the agreement found is the same number.
    """
    rng = np.random.default_rng(3)
    maps = rng.standard_normal((2, 3, 40))
    bin_seconds, tolerance, apart_by = 0.5, 3.0, 2.0
    together, lags_a, res_a = correlation_profiles(
        maps, maps, [0], [1], [tolerance], bin_seconds, offset_s=np.array([0.0]))
    apart, lags_b, res_b = correlation_profiles(
        maps, maps, [0], [1], [tolerance], bin_seconds,
        offset_s=np.array([apart_by]))
    for displacement in (1.0, 2.0, 3.0):
        a = together[0, :, _at(res_a[0] + lags_a, displacement - apart_by)]
        b = apart[0, :, _at(res_b[0] + lags_b, displacement)]
        np.testing.assert_allclose(a, b, atol=1e-12)
        assert np.any(b)


def test_the_maximisation_is_over_the_admitted_arrival_time_differences():
    """The displacement is maximised over, and only over, what geometry allows.

    The tolerance is on the pair's absolute arrival-time difference, whatever
    its two anchors are, so a pair declared far apart is still compared only at
    the displacements the light travel time admits --- the map slides to bring
    the two transients into a causally allowed alignment, and the maximum is
    taken there and nowhere else.
    """
    rng = np.random.default_rng(5)
    maps = rng.standard_normal((2, 2, 200))
    bin_seconds, tolerance = 0.01, 0.05
    for anchors_apart in (0.0, 0.3):
        profiles, lags, residual = correlation_profiles(
            maps, maps, [0], [1], [tolerance], bin_seconds,
            offset_s=np.array([anchors_apart]))
        absolute = residual[0] + lags
        carried = np.any(profiles[0], axis=0)
        assert np.all(np.abs(absolute[carried]) <= tolerance + 1e-12)
        assert carried.any()
