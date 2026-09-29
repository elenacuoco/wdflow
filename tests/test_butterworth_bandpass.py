"""The Butterworth conditioning: no zero inside the decimated band, an
anti-alias deep enough where the strain's lines fold back, and the same
streaming contract as the Chebyshev."""
import numpy as np
import pytest
from types import SimpleNamespace
from scipy.signal import sosfreqz
from wdf.filtering import sosfiltfilt

pytest.importorskip("py4tsa")

from wdf.processes.BandPassDownSampling import (BandPassDownSampling,
                                                settling_length)

SAMPLING, FACTOR = 4096, 2
RESAMPLING = SAMPLING // FACTOR          # 2048 Hz, final Nyquist 1024 Hz


def parameters(**overrides):
    par = dict(sampling=SAMPLING, resampling=RESAMPLING, ResamplingFactor=FACTOR,
               LowFrequencyCut=12.0, BandPassFilter="butterworth")
    par.update(overrides)
    return SimpleNamespace(**par)


class _Block:
    def __init__(self, samples, start):
        self._samples, self._start = np.asarray(samples, dtype=float), float(start)

    def GetSize(self):
        return len(self._samples)

    def GetStart(self):
        return self._start

    def GetY(self, _channel, i):
        return self._samples[i]


def read_back(view):
    return np.array([view.GetY(0, i) for i in range(view.GetSize())])


def stream(filt, samples, block=SAMPLING):
    out, starts = [], []
    for first in range(0, len(samples) - block + 1, block):
        view = filt.Process(_Block(samples[first:first + block], first / SAMPLING))
        if view is not None:
            out.append(read_back(view))
            starts.append(view.GetStart())
    return out, starts


def response_db(filt, freq):
    """20 log10 |H|^2: the filter runs forward and backward."""
    _, h = sosfreqz(filt.sos, worN=freq, fs=SAMPLING)
    return 20.0 * np.log10(np.abs(h) ** 2 + 1e-300)


def test_the_default_is_still_the_chebyshev():
    par = parameters()
    del par.BandPassFilter
    assert BandPassDownSampling(par).family == "cheby2"


def test_there_is_no_notch_inside_the_decimated_band():
    """An all-pole noise model cannot follow a spectral zero, so a zero of the
    conditioning inside the decimated band comes out of the whitening as a
    notch. The response rises to the passband and falls after it, nothing else."""
    filt = BandPassDownSampling(parameters())
    freq = np.arange(1.0, 0.5 * RESAMPLING, 0.25)
    gain = response_db(filt, freq)
    top = int(np.argmax(gain))
    assert np.all(np.diff(gain[:top]) > -1e-9)
    assert np.all(np.diff(gain[top:]) < 1e-9)


def test_the_corners_are_where_the_data_lose_half_their_amplitude():
    filt = BandPassDownSampling(parameters())
    assert filt.cutoff_frequency == pytest.approx(800.0)
    assert response_db(filt, np.array([12.0, 800.0])) == pytest.approx([-6.02, -6.02], abs=0.05)
    low, high = filt.passband()
    assert 13.0 < low < 14.0 and 760.0 < high < 770.0


def test_what_folds_back_into_the_band_is_attenuated():
    """Above 1148 Hz the input folds below 900 Hz after decimation, and the
    third violin harmonics at 1450-1510 Hz stand up to 200 times above the
    floor in the strain."""
    filt = BandPassDownSampling(parameters())
    assert response_db(filt, np.arange(1148.0, 0.5 * SAMPLING, 1.0)).max() < -70.0
    assert response_db(filt, np.arange(1450.0, 1510.0, 1.0)).max() < -140.0


def test_a_tone_inside_the_band_survives():
    filt = BandPassDownSampling(parameters())
    t = np.arange(SAMPLING * 12) / SAMPLING
    blocks, _ = stream(filt, np.sin(2 * np.pi * 200.0 * t))
    assert np.std(np.concatenate(blocks[2:])) == pytest.approx(np.sqrt(0.5), rel=0.01)


def test_a_block_matches_the_whole_stream_filtered_at_once():
    filt = BandPassDownSampling(parameters())
    samples = np.random.default_rng(1).standard_normal(SAMPLING * 12)
    reference = sosfiltfilt(filt.sos, samples)[::FACTOR]
    blocks, starts = stream(filt, samples)
    for block, start in list(zip(blocks, starts))[1:]:
        offset = int(round(start * RESAMPLING))
        expected = reference[offset:offset + len(block)]
        assert (np.abs(block - expected) / np.std(expected)).max() < 1e-9


def test_it_settles_faster_than_the_chebyshev():
    butter = BandPassDownSampling(parameters())
    cheby = BandPassDownSampling(parameters(BandPassFilter="cheby2", FilterOrder=10))
    assert butter.padlen == settling_length(butter.sos, SAMPLING)
    assert butter.padlen < cheby.padlen / 2


def test_the_orders_and_the_corner_are_parameters():
    filt = BandPassDownSampling(parameters(HighPassOrder=2, LowPassOrder=10, LowPassCorner=850.0))
    assert (filt.highpass_order, filt.lowpass_order, filt.cutoff_frequency) == (2, 10, 850.0)
    assert filt.sos.shape == (6, 6)


@pytest.mark.parametrize("corner", [1024.0, 10.0])
def test_a_low_pass_corner_outside_the_band_is_refused(corner):
    with pytest.raises(ValueError, match="low-pass corner"):
        BandPassDownSampling(parameters(LowPassCorner=corner))


def test_an_unknown_family_is_refused():
    with pytest.raises(ValueError, match="BandPassFilter"):
        BandPassDownSampling(parameters(BandPassFilter="elliptic"))


@pytest.mark.parametrize("family", ["cheby2", "butterworth"])
def test_the_worker_records_the_band_pass_it_used(tmp_outdir, family):
    import glob
    import json

    from conftest import run_segment_process

    triggers = run_segment_process(tmp_outdir, BandPassFilter=family)
    assert len(triggers) > 0
    with open(glob.glob(f"{tmp_outdir}**/parametersUsed-Win*.json", recursive=True)[0]) as fh:
        used = json.load(fh)
    assert used["BandPassFilter"] == family
    low, high = used["BandPassPassband"]
    assert used["LowFrequencyCut"] < low < high < 0.5 * used["resampling"]
