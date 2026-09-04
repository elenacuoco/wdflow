"""The arrival-time difference estimator, against constructions with a known
answer: the lag is exact where the shift is exact, the sign follows the
convention stated, absolute placement cancels, and the bound is respected."""
import numpy as np
import pandas as pd
import pytest

from wdf.analysis.timing import (arrival_time_difference, envelope_instant,
                                 envelope_instants, referred_to_instant)


FS = 2048.0


def _pulse(n=400, centre=200, width=6.0):
    t = np.arange(n)
    return np.exp(-0.5 * ((t - centre) / width) ** 2) * np.sin(0.7 * t)


def test_recovers_a_known_shift_with_positive_sign():
    # The first series carries the pulse seven samples later, so it arrives
    # after the second and dt must be positive by the stated convention.
    dt, _ = arrival_time_difference(
        (0.0, np.roll(_pulse(), 7)), (0.0, _pulse()), FS)
    assert dt == pytest.approx(7.0 / FS, abs=0.5 / FS)


def test_absolute_placement_cancels():
    # Moving both series by the same start time changes nothing, and a start
    # offset is equivalent to a sample shift: the difference is what counts.
    shift_s = 5.0 / FS
    dt, _ = arrival_time_difference(
        (1e9 + shift_s, _pulse()), (1e9, _pulse()), FS)
    assert dt == pytest.approx(shift_s, abs=0.5 / FS)


def test_sigma_is_floored_at_one_sample():
    _, sigma = arrival_time_difference((0.0, _pulse()), (0.0, _pulse()), FS)
    assert sigma >= 1.0 / FS


def test_lag_search_is_bounded():
    # A shift beyond the bound cannot be returned: the estimator answers
    # inside its stated window rather than chasing a distant overlap.
    far = int(0.2 * FS)
    dt, _ = arrival_time_difference(
        (0.0, np.roll(_pulse(2048, 1024), far)), (0.0, _pulse(2048, 1024)),
        FS, max_lag_s=0.05)
    assert abs(dt) <= 0.05 + 1.0 / FS


def test_empty_series_is_refused():
    with pytest.raises(ValueError):
        arrival_time_difference((0.0, np.array([])), (0.0, _pulse()), FS)


def test_a_placement_the_search_cannot_reach_is_refused():
    """The lag search is bounded, so two supports further apart than it may
    shift them can never be brought into contact. Answering with the lag at
    the edge of the search would be a number where there is no measurement,
    and laying the gap out to find it is the cost of every such pair."""
    fs = 512.0
    wave = np.sin(2 * np.pi * 70.0 * np.arange(512) / fs)
    with pytest.raises(ValueError, match="not on one clock"):
        arrival_time_difference((0.0, wave), (600.0, wave), fs, max_lag_s=0.05)


def test_the_offset_is_what_a_slide_leaves_alone():
    """A slide displaces the catalogue and carries the waveforms with it, so
    the waveform's place inside its own event is the same before and after.
    That is the quantity the estimator has to be given: referring the series
    to a displaced instant while it still carries an absolute start puts the
    whole displacement into the pair."""
    events = pd.DataFrame({"cluster_id": [3, 7],
                           "gpsPeak": [1000.25, 1000.40]})
    series = {3: (1000.0, np.zeros(4)), 7: (1000.3, np.zeros(4))}

    offsets = referred_to_instant(series, events)
    assert offsets[3][0] == pytest.approx(-0.25)
    assert offsets[7][0] == pytest.approx(-0.10)

    # The same events and waveforms, both displaced by one slide.
    shift = 600.0
    slid_events = events.assign(gpsPeak=events.gpsPeak + shift)
    slid_series = {label: (start + shift, samples)
                   for label, (start, samples) in series.items()}
    slid = referred_to_instant(slid_series, slid_events)
    assert slid[3][0] == pytest.approx(offsets[3][0])
    assert slid[7][0] == pytest.approx(offsets[7][0])


def test_a_reconstruction_of_no_event_is_refused():
    events = pd.DataFrame({"cluster_id": [3], "gpsPeak": [1000.25]})
    with pytest.raises(ValueError, match="belongs to no event"):
        referred_to_instant({9: (1000.0, np.zeros(4))}, events)


def test_the_cost_follows_the_length_and_not_its_square():
    """An event assembled from many blocks is long, and only the few hundred
    lags the geometry allows are ever read. Forming the whole correlation to
    keep them costs the product of the two lengths; forming each kept lag
    costs the length, once per lag."""
    import time as _time

    fs = 2048.0
    n = 200_000
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n)
    # The same noise, carried seven samples earlier in the second series, so
    # the first arrives that much after it and the answer is known as well.
    b = np.roll(a, -7)

    began = _time.time()
    dt, _ = arrival_time_difference((0.0, a), (0.0, b), fs, max_lag_s=0.05)
    elapsed = _time.time() - began

    assert dt == pytest.approx(+7.0 / fs, abs=0.5 / fs)
    # Quadratic in the length would be 4e10 products here, minutes of work.
    assert elapsed < 5.0, f"the whole correlation was formed ({elapsed:.1f} s)"


def test_the_envelope_peak_is_read_at_the_sample():
    """The instant is where the amplitude is largest, on the sample grid, and
    the tile centre it is sought about only says where to look."""
    fs = 2048.0
    t = np.arange(1024) / fs
    wave = np.sin(2 * np.pi * 150.0 * t) * np.exp(-((t - 0.30) / 0.01) ** 2)
    read = envelope_instant((1000.0, wave), 1000.30, fs, 0.25)
    assert read == pytest.approx(1000.30, abs=1.0 / fs)
    # The tile centre may be off by a good fraction of the window and the
    # answer does not move, because it is the envelope that is read.
    assert envelope_instant((1000.0, wave), 1000.38, fs, 0.25) == \
        pytest.approx(read, abs=1.0 / fs)


def test_the_search_is_bounded_by_the_block():
    """An event assembled from many blocks spreads its energy over its extent,
    and the envelope of a long transient peaks where that energy concentrated.
    The instant stays on the block the event was ranked on."""
    fs = 2048.0
    t = np.arange(8192) / fs
    early = np.sin(2 * np.pi * 150.0 * t) * np.exp(-((t - 0.30) / 0.01) ** 2)
    late = 5.0 * np.sin(2 * np.pi * 150.0 * t) * np.exp(-((t - 3.50) / 0.01) ** 2)
    wave = early + late

    # Sought about the early feature within one block: the later and louder
    # one is outside the window and does not win.
    assert envelope_instant((1000.0, wave), 1000.30, fs, 0.25) == \
        pytest.approx(1000.30, abs=1.0 / fs)
    # Sought over the whole event, the loudest feature wins instead.
    assert envelope_instant((1000.0, wave), 1000.30, fs, 8.0) == \
        pytest.approx(1003.50, abs=1.0 / fs)


def test_a_series_with_nothing_to_read_gives_no_instant():
    fs = 2048.0
    assert np.isnan(envelope_instant((1000.0, np.zeros(512)), 1000.1, fs, 0.25))
    assert np.isnan(envelope_instant((1000.0, np.zeros(0)), 1000.1, fs, 0.25))
    # A block that lies outside the samples is not a reading of anything.
    wave = np.sin(2 * np.pi * 150.0 * np.arange(512) / fs)
    assert np.isnan(envelope_instant((1000.0, wave), 1005.0, fs, 0.25))


def test_the_instant_is_refused_without_a_rate_or_a_width():
    wave = np.ones(64)
    with pytest.raises(ValueError, match="sampling frequency"):
        envelope_instant((0.0, wave), 0.0, 0.0, 0.25)
    with pytest.raises(ValueError, match="search width"):
        envelope_instant((0.0, wave), 0.0, 2048.0, 0.0)


def test_every_event_gets_its_own_instant_and_none_gets_another_s():
    fs = 2048.0
    t = np.arange(1024) / fs
    events = pd.DataFrame({"cluster_id": [3, 7, 9],
                           "gpsPeak": [1000.30, 2000.40, 3000.10]})
    series = {
        3: (1000.0, np.sin(2 * np.pi * 150 * t) * np.exp(-((t - 0.30) / 0.01) ** 2)),
        7: (2000.0, np.sin(2 * np.pi * 150 * t) * np.exp(-((t - 0.40) / 0.01) ** 2)),
        # 9 has no reconstruction
    }
    out = envelope_instants(series, events, fs, 0.25)
    assert out[0] == pytest.approx(1000.30, abs=1.0 / fs)
    assert out[1] == pytest.approx(2000.40, abs=1.0 / fs)
    assert np.isnan(out[2])
