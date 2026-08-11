import numpy as np
import pandas as pd
import pytest

from _synth import forward, triggers_from_signal
from wdf.analysis.reconstruction import combined_snr, inverse_transform, stitch

FS = 2048.0
WINDOW, OVERLAP = 512, 128
STEP = WINDOW - OVERLAP
WAVE = "DaubC12"
SIGMA = 1.0


def triggers_from(signal, gps0=1000.0, wave=WAVE):
    """One trigger per analysis window covering `signal`, as WDF would emit."""
    return triggers_from_signal(signal, FS, WINDOW, OVERLAP, gps0=gps0,
                                wave=wave, sigma=SIGMA)


def test_inverse_transform_round_trips():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(WINDOW)
    assert inverse_transform(forward(x, WAVE), WAVE) == pytest.approx(x, rel=1e-3, abs=1e-6)


def test_stitching_recovers_the_signal_it_was_cut_from():
    """The step regions tile the span, so the pieces reassemble the original."""
    rng = np.random.default_rng(1)
    signal = rng.standard_normal(WINDOW + 4 * STEP)

    gps_start, stitched = stitch(triggers_from(signal), FS, WINDOW, OVERLAP)

    assert gps_start == pytest.approx(1000.0)
    assert stitched[:len(signal)] == pytest.approx(signal[:len(stitched)], rel=1e-3, abs=1e-6)


def test_combined_snr_of_a_long_signal_exceeds_its_loudest_window():
    """A signal spread over many windows carries more than any one of them."""
    t = np.arange(WINDOW + 8 * STEP) / FS
    chirp = np.sin(2 * np.pi * (40.0 * t + 30.0 * t ** 2))

    result = combined_snr(triggers_from(chirp), FS, WINDOW, OVERLAP)

    assert result["windows"] > 1
    assert result["EnWDF"] > result["loudest_window"]
    assert result["EnWDF"] == pytest.approx(np.linalg.norm(chirp) / SIGMA, rel=0.02)


def test_a_single_window_signal_gains_nothing():
    """Stitching must not inflate a burst that already fits in one window."""
    t = (np.arange(WINDOW) - WINDOW / 2) / FS
    burst = np.exp(-((t / 0.005) ** 2) / 2.0) * np.cos(2 * np.pi * 200.0 * t)

    triggers = triggers_from(np.concatenate([burst, np.zeros(2 * STEP)]))
    result = combined_snr(triggers, FS, WINDOW, OVERLAP)

    assert result["EnWDF"] == pytest.approx(result["loudest_window"], rel=0.05)


def test_stitch_needs_coefficients():
    triggers = pd.DataFrame([dict(gps=1000.0, wave=WAVE, sigma=SIGMA, EnWDF=1.0)])
    with pytest.raises(KeyError, match="n_coeff"):
        stitch(triggers, FS, WINDOW, OVERLAP)


def test_wavegram_events_score_a_multi_window_signal_on_its_reconstruction():
    """Percolation on the wavegram gathers the windows; the score comes from
    the reconstruction, so it exceeds the loudest single window."""
    from wdf.analysis.clustering import wavegram_events

    rng = np.random.default_rng(3)
    signal = np.zeros(WINDOW + 4 * STEP)
    envelope = np.hanning(WINDOW + 4 * STEP)
    signal += envelope * np.sin(2 * np.pi * 200.0 * np.arange(signal.size) / FS)
    signal += 1e-3 * rng.standard_normal(signal.size)

    triggers = triggers_from(signal)
    events = wavegram_events(triggers, FS, WINDOW, OVERLAP, time_tol_s=0.05)

    assert len(events) >= 1
    loudest = events.sort_values("EnWDF").iloc[-1]
    assert loudest.EnWDF > loudest.loudest_window
    assert loudest.windows > 1
    assert loudest.freqMin <= loudest.freqMean <= loudest.freqMax


def test_the_reconstruction_centroid_recovers_a_known_frequency():
    import numpy as np
    from _synth import triggers_from_signal
    from wdf.analysis.reconstruction import spectral_centroid

    fs = 2048.0
    t = np.arange(int(2 * fs)) / fs
    signal = np.sin(2 * np.pi * 180.0 * t) * np.exp(-((t - 1.0) / 0.02) ** 2)
    centroid = spectral_centroid(triggers_from_signal(signal, fs, 512, 128))
    assert np.nanmedian(centroid) == pytest.approx(180.0, rel=0.05)


def test_the_reconstruction_centroid_escapes_the_octave_ladder():
    """The tile moment cannot resolve inside an octave; the reconstruction is a
    time series and its spectrum is not tied to the ladder."""
    import numpy as np
    from _synth import triggers_from_signal
    from wdf.analysis.reconstruction import spectral_centroid
    from wdf.analysis.wavelets import coeff_freq_bands, tile_frequency

    fs = 2048.0
    t = np.arange(int(4 * fs)) / fs
    rng = np.random.default_rng(0)
    triggers = triggers_from_signal(rng.normal(size=len(t)), fs, 512, 128)
    centroid = spectral_centroid(triggers)
    finite = centroid[np.isfinite(centroid)]
    assert finite.size

    f_lo, f_hi = coeff_freq_bands(512, fs)
    ladder = {round(tile_frequency(a, b), 6) for a, b in zip(f_lo, f_hi)}
    on_ladder = sum(1 for v in finite if round(float(v), 6) in ladder)
    assert on_ladder < 0.2 * finite.size


def test_an_empty_frame_has_no_centroid():
    import pandas as pd
    from wdf.analysis.reconstruction import spectral_centroid

    assert spectral_centroid(
        pd.DataFrame(columns=["n_coeff", "fs", "wave", "wt_index", "wt_value"])).size == 0


def _boundary_jumps(samples, window, overlap, n_windows):
    """Sample-to-sample change at each window boundary of a stitched series."""
    step = window - overlap
    edges = [k * step for k in range(1, n_windows)]
    return np.array([abs(samples[e] - samples[e - 1])
                     for e in edges if 0 < e < len(samples)])


def _thresholded(triggers, window, sigma=1.0):
    """The triggers as the search writes them, one threshold per window.

    Two windows covering the same samples keep different coefficients, so their
    reconstructions of the overlapped region disagree. That disagreement is the
    only thing a stitching policy has to reconcile, and without it every policy
    gives the same answer.
    """
    from wdf.analysis.coefficients import coefficient_matrix, from_dense
    from wdf.analysis.wavelets import donoho_johnstone_threshold

    cut = donoho_johnstone_threshold(sigma, window)
    kept = [from_dense(np.where(np.abs(row) > cut, row, 0.0))
            for row in coefficient_matrix(triggers)]
    triggers = triggers.copy()
    triggers["wt_index"] = [index for index, _ in kept]
    triggers["wt_value"] = [value for _, value in kept]
    return triggers


@pytest.mark.parametrize("overlap", [32, 256])
def test_overlap_add_crosses_the_window_boundary_without_a_step(overlap):
    """A boundary is not visible as a jump the signal does not make itself.

    Both the overlap the search runs at and half a window: the crossfade ramp
    is as long as the overlap, so a short one is the case that could fail.
    """
    from _synth import triggers_from_signal
    from wdf.analysis.reconstruction import stitch

    fs, window = 2048.0, 512
    n = 4 * window
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    signal = 6.0 * np.sin(2.0 * np.pi * 150.0 * t) + rng.normal(size=n)
    triggers = _thresholded(triggers_from_signal(signal, fs, window, overlap),
                            window)

    _, smooth = stitch(triggers, fs, window, overlap,
                       overlap_policy="overlap_add")
    _, hard = stitch(triggers, fs, window, overlap,
                     overlap_policy="central_window")

    smooth_edges = _boundary_jumps(smooth, window, overlap, len(triggers))
    hard_edges = _boundary_jumps(hard, window, overlap, len(triggers))
    typical = float(np.median(np.abs(np.diff(smooth))))
    assert len(smooth_edges)
    assert smooth_edges.max() < hard_edges.max()
    assert smooth_edges.max() <= 1.5 * typical


def test_overlap_add_recovers_the_signal_it_was_cut_from():
    """Averaging the two estimates does not bias the reconstruction."""
    from _synth import triggers_from_signal
    from wdf.analysis.reconstruction import stitch

    fs, window, overlap = 2048.0, 512, 256
    n = 4 * window
    t = np.arange(n) / fs
    signal = 20.0 * np.sin(2.0 * np.pi * 150.0 * t) * np.exp(-((t - 0.5) ** 2) / 0.02)
    triggers = triggers_from_signal(signal, fs, window, overlap)

    _, stitched = stitch(triggers, fs, window, overlap)
    covered = slice(0, len(stitched))
    reference = signal[covered]
    error = np.linalg.norm(stitched - reference) / np.linalg.norm(reference)
    assert error < 0.05


def test_the_synthesis_weight_vanishes_where_two_windows_meet():
    """The weight is zero at the block edge, which is what removes the step."""
    from wdf.analysis.reconstruction import synthesis_weight

    weight = synthesis_weight(512, 256)
    assert weight[0] < 0.05 and weight[-1] < 0.05
    assert weight.max() == pytest.approx(1.0, rel=1e-3)
    assert np.all(np.diff(weight[:256]) > 0)
    assert np.all(weight > 0)


def test_an_unknown_overlap_policy_is_refused():
    from _synth import triggers_from_signal
    from wdf.analysis.reconstruction import stitch

    triggers = triggers_from_signal(np.zeros(1024), 2048.0, 512, 256)
    with pytest.raises(ValueError, match="unknown overlap policy"):
        stitch(triggers, 2048.0, 512, 256, overlap_policy="average")


def test_the_overlap_is_blind_to_a_drift_the_phase_residual_sees():
    """An aggregate agreement can hide a phase that walks between windows."""
    from wdf.analysis.reconstruction import phase_residual, waveform_overlap

    fs, window, n = 2048.0, 512, 4 * 512
    t = np.arange(n) / fs
    injected = np.sin(2.0 * np.pi * 120.0 * t) * np.exp(-((t - 0.5) ** 2) / 0.05)

    # The same waveform, but each window placed one sample late: every piece is
    # correct on its own and the whole is not.
    drifted = injected.copy()
    for k in range(1, n // window):
        piece = slice(k * window, (k + 1) * window)
        drifted[piece] = np.roll(injected[piece], k)

    # The overlap still reads as a recovery, while the phase is wrong by a
    # sizeable fraction of a radian: one is an average over the series and the
    # other is a statement about each sample.
    assert waveform_overlap(drifted, injected)["overlap"] > 0.8
    walked = phase_residual(drifted, injected)
    clean = phase_residual(injected, injected)
    assert clean["median_abs"] == pytest.approx(0.0, abs=1e-9)
    assert np.abs(walked["residual"]).max() > 0.3


def test_a_reconstruction_out_of_phase_does_not_score_as_recovered():
    """The overlap is phase sensitive, not a comparison of envelopes."""
    from wdf.analysis.reconstruction import waveform_overlap

    fs, n = 2048.0, 2048
    t = np.arange(n) / fs
    envelope = np.exp(-((t - 0.5) ** 2) / 0.01)
    injected = envelope * np.sin(2.0 * np.pi * 120.0 * t)
    inverted = envelope * np.sin(2.0 * np.pi * 120.0 * t + np.pi)

    assert waveform_overlap(injected, injected)["overlap"] == pytest.approx(1.0)
    assert waveform_overlap(inverted, injected)["overlap"] < -0.9


def test_series_of_different_lengths_are_refused():
    from wdf.analysis.reconstruction import waveform_overlap

    with pytest.raises(ValueError, match="common time base"):
        waveform_overlap(np.zeros(10), np.zeros(11))
