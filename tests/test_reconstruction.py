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
