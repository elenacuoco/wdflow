import numpy as np
import pytest
from py4tsa.tsa import SeqView_double_t as SV
from scipy.signal import fftconvolve, lfilter, welch

from wdf.processes.zero_phase_whitening import (
    MagnitudeWhitening,
    magnitude_taps,
    prediction_error_polynomial,
)
from wdf.structures.array2SeqView import array2SeqView

FS = 2048.0


def line_model(radius=0.9995, frequency=300.0, scale=2.0):
    """An AR model with a smooth colour and one narrow line.

    The line is a pair of zeros of ``A`` at `radius` from the origin, which is
    what makes ``|A|`` ring: the closer to the unit circle, the longer.
    """
    line = [1.0, -2.0 * radius * np.cos(2 * np.pi * frequency / FS), radius ** 2]
    polynomial = np.convolve(line, [1.0, -0.9])
    return np.concatenate([[scale], -polynomial[1:]])


def coloured_noise(ar, seconds, seed):
    rng = np.random.default_rng(seed)
    return lfilter([ar[0]], prediction_error_polynomial(ar),
                   rng.standard_normal(int(seconds * FS)))


def stream(whitening, x, cuts):
    """Feed `x` in chunks split at `cuts`, drain every block the filter can give.

    :return: the whitened samples and the start time of each output block.
    """
    out, starts = [], []
    for chunk_start, chunk in zip(np.r_[0, cuts], np.split(x, cuts)):
        view = array2SeqView(chunk_start / FS, FS, chunk.size).Fill(chunk_start / FS, chunk)
        whitening.Input(view)
        while whitening.DataNeeded() >= 0:
            block = SV()
            whitening.Output(block)
            out.append(np.array([block.GetY(0, i) for i in range(block.GetSize())]))
            starts.append(block.GetStart())
    return np.concatenate(out), np.array(starts)


def test_the_response_is_real_and_positive():
    """The taps are even, so the response is real: its phase is zero.

    The phase is measured on the response of the taps as they are applied,
    lag zero at the centre, and it must vanish in the band and not only on
    average: a real response that went negative somewhere would have phase pi
    there.
    """
    taps = magnitude_taps(line_model())
    support = (taps.size - 1) // 2
    assert np.array_equal(taps, taps[::-1])

    grid = 1 << 16
    centred = np.roll(np.concatenate([taps, np.zeros(grid - taps.size)]), -support)
    response = np.fft.rfft(centred)
    phase = np.angle(response)
    assert np.max(np.abs(phase)) < 1e-9
    assert response.real.min() > 0.0


def test_the_response_is_the_modulus_of_the_model():
    """What is applied is ``|A|``, to the truncation's error."""
    ar = line_model()
    taps = magnitude_taps(ar)
    support = (taps.size - 1) // 2
    grid = 1 << 16
    centred = np.roll(np.concatenate([taps, np.zeros(grid - taps.size)]), -support)
    applied = np.fft.rfft(centred).real
    model = np.abs(np.fft.rfft(prediction_error_polynomial(ar), grid))
    assert np.median(np.abs(applied / model - 1.0)) < 1e-4


def test_the_whitened_spectrum_is_the_causal_one():
    """Same modulus as the causal whitening, so the same spectrum, line included.

    The two outputs differ in phase only, so their spectra are compared bin by
    bin on the same data, not against a white level each estimates on its own.
    """
    ar = line_model()
    x = coloured_noise(ar, 200.0, seed=0)
    whitening = MagnitudeWhitening(ar, 1024, 0)
    whitening.SetOutputSize(1024, whitening.latency)
    zero_phase, _ = stream(whitening, x, [int(37.3 * FS), int(120.1 * FS)])
    causal = lfilter(prediction_error_polynomial(ar), [1.0], x)

    settled = slice(2 * whitening.latency, zero_phase.size - 2 * whitening.latency)
    f, p_zero = welch(zero_phase[settled], fs=FS, nperseg=8192)
    _, p_causal = welch(causal[settled], fs=FS, nperseg=8192)
    departure = np.abs(np.log10(p_zero / p_causal))[f > 16.0]
    assert np.median(departure) < 0.03
    assert departure.max() < 0.2
    assert np.std(zero_phase[settled]) == pytest.approx(whitening.sigma, rel=0.02)


def test_the_stream_is_the_stream_whitened_at_once():
    """Blocks joined are one linear convolution with the whole stream.

    Real past and real future enter every block, so the output does not depend
    on where the reads or the output blocks begin: at the joins as everywhere
    else, it equals the whole stream filtered in one piece.
    """
    ar = line_model()
    x = coloured_noise(ar, 60.0, seed=1)
    whitening = MagnitudeWhitening(ar, 2000, 0)
    whitening.SetOutputSize(2000, whitening.latency)
    cuts = [777, 5000, 5001, 40000, 90001]
    streamed, starts = stream(whitening, x, cuts)

    reference = fftconvolve(x, whitening.taps, mode="full")[whitening.latency:]
    reference = reference[:streamed.size]
    settled = slice(whitening.latency, None)
    error = np.abs(streamed[settled] - reference[settled]).max()
    assert error < 1e-12 * np.abs(reference).max()

    joins = np.arange(2000, streamed.size, 2000)
    joins = joins[joins > whitening.latency]
    around = np.concatenate([joins - 1, joins])
    assert np.abs(streamed[around] - reference[around]).max() < 1e-12 * np.abs(reference).max()
    assert np.allclose(np.diff(starts), 2000 / FS)
    assert starts[0] == 0.0


def test_a_transient_stays_where_it_is():
    """Zero phase: the whitened pulse is centred where the data put it."""
    ar = line_model()
    t = (np.arange(16384) - 8192) / FS
    pulse = np.exp(-((t / 0.01) ** 2) / 2.0) * np.cos(2 * np.pi * 150.0 * t)
    whitening = MagnitudeWhitening(ar, 1, 0)
    zero_phase = fftconvolve(pulse, whitening.taps, mode="same")
    causal = lfilter(prediction_error_polynomial(ar), [1.0], pulse)

    def centroid(x):
        e = np.asarray(x) ** 2
        return float((np.arange(len(e)) * e).sum() / e.sum())

    assert abs(centroid(zero_phase) - centroid(pulse)) / FS < 1e-6
    assert abs(centroid(causal) - centroid(pulse)) / FS > 1e-4


def test_the_support_follows_the_zeros_not_the_order():
    """The same order rings longer when its zeros come closer to the circle."""
    near = (magnitude_taps(line_model(radius=0.9999)).size - 1) // 2
    far = (magnitude_taps(line_model(radius=0.99)).size - 1) // 2
    assert near > far


def test_a_response_truncated_at_a_few_orders_loses_the_line():
    """Four times the order is not the support of a model with a narrow line.

    Truncated there, the filter no longer notches the line, and the line comes
    through the whitening far above the causal output; the measured support
    keeps it near the causal level. Near, not at: the error left at a line
    falls as the inverse of the support, and a line this narrow is where it is
    largest.
    """
    ar = line_model(radius=0.9999)
    x = coloured_noise(ar, 200.0, seed=2)
    causal = lfilter(prediction_error_polynomial(ar), [1.0], x)
    order = len(ar) - 1
    edge = int(20 * FS)
    f, p_causal = welch(causal[edge:-edge], fs=FS, nperseg=8192)
    at_line = np.argmin(np.abs(f - 300.0))

    def line_excess(taps):
        _, p = welch(fftconvolve(x, taps, mode="same")[edge:-edge], fs=FS, nperseg=8192)
        return p[at_line] / p_causal[at_line]

    assert line_excess(magnitude_taps(ar, support=2 * order)) > 10.0
    assert line_excess(magnitude_taps(ar)) == pytest.approx(1.0, rel=0.2)


def test_a_lookahead_shorter_than_the_support_is_refused():
    whitening = MagnitudeWhitening(line_model(), 1024, 0)
    with pytest.raises(ValueError, match="WhiteningExtraSize"):
        whitening.SetOutputSize(1024, whitening.latency - 1)
    with pytest.raises(ValueError, match="WhiteningExtraSize"):
        MagnitudeWhitening(line_model(), 1024, whitening.latency - 1)


def test_a_block_is_not_produced_before_its_future_has_arrived():
    ar = line_model()
    whitening = MagnitudeWhitening(ar, 1000, 0)
    whitening.SetOutputSize(1000, whitening.latency)
    x = coloured_noise(ar, 1.0, seed=3)[:whitening.latency + 999]
    whitening.Input(array2SeqView(0.0, FS, x.size).Fill(0.0, x))
    assert whitening.DataNeeded() < 0
    with pytest.raises(RuntimeError):
        whitening.Output(SV())


def test_the_spectral_response_whitens_what_it_was_measured_on():
    """From a measured spectrum, the same filter: white at the scale it reports."""
    rng = np.random.default_rng(11)
    coloured = lfilter([1.0], [1.0, -0.9, 0.2], rng.standard_normal(200000))
    whitening = MagnitudeWhitening.from_spectrum(
        coloured, FS, 2048, 0, band=(8.0, FS / 2))
    assert whitening.latency == 8192 // 2 - 1

    whitened = fftconvolve(coloured, whitening.taps, mode="valid")
    freq, power = welch(whitened, fs=FS, nperseg=8192)
    white = np.sqrt(power) / (whitening.sigma * np.sqrt(2.0 / FS))
    band = (freq >= 16.0) & (freq <= 0.45 * FS)
    assert 0.9 < np.median(white[band]) < 1.1
    assert np.std(np.log10(white[band])) < 0.05


@pytest.mark.parametrize("name", ["magnitude", "root"])
def test_the_worker_records_the_filter_it_ran(tmp_outdir, name):
    """Either filter runs, and the run parameters say which, with its latency,
    a warm-up that covers the filter's past and a lookahead that covers its
    future."""
    import glob
    import json

    from conftest import run_segment_process

    triggers = run_segment_process(tmp_outdir, ZeroPhaseFilter=name)
    assert len(triggers) > 0
    with open(glob.glob(f"{tmp_outdir}**/parametersUsed-Win*.json", recursive=True)[0]) as fh:
        used = json.load(fh)
    assert used["ZeroPhaseFilter"] == name
    assert used["preWhite"] * used["resampling"] >= used["ZeroPhaseLatency"]
    assert used["WhiteningExtraSize"] >= used["ZeroPhaseLatency"]


def test_the_worker_refuses_an_unknown_filter(tmp_outdir):
    from conftest import run_segment_process

    with pytest.raises(ValueError, match="ZeroPhaseFilter"):
        run_segment_process(tmp_outdir, ZeroPhaseFilter="causal")
