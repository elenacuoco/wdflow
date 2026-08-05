import os

import numpy as np
import pytest

from wdf.mock import waveforms as w
from wdf.mock.dataset import draw_injections, generate_dataset, optimal_snr

FS = 2048
LIGHT_TRAVEL_H1L1 = 0.0100


@pytest.mark.parametrize("name", sorted(w.GLITCH_GENERATORS))
def test_glitch_generators_are_finite_and_non_trivial(name):
    y = w.GLITCH_GENERATORS[name]()
    assert len(y) > 1
    assert np.isfinite(y).all()
    assert np.abs(y).max() > 0


@pytest.mark.parametrize("name", sorted(w.GLITCH_GENERATORS))
def test_glitch_generators_respect_sample_rate(name):
    slow = w.GLITCH_GENERATORS[name](sample_rate=FS)
    fast = w.GLITCH_GENERATORS[name](sample_rate=2 * FS)
    assert len(fast) > len(slow)


def test_optimal_snr_scales_linearly_with_amplitude():
    y = w.sine_gaussian()
    assert optimal_snr(2 * y) == pytest.approx(2 * optimal_snr(y), rel=1e-6)


def test_scattered_light_stays_below_its_peak_frequency():
    """Instantaneous frequency of the arches never exceeds f_peak."""
    f_peak = 40.0
    y = w.scattered_light(f_peak=f_peak, arch_period=1.0, n_arches=2, sample_rate=FS)
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / FS)
    assert freqs[np.argmax(spectrum)] <= f_peak


def test_draw_injections_are_ordered_and_do_not_overlap():
    """`gps` is the merger time for a CBC and the centre for a glitch, so the
    occupied span is given by the recorded support, not by the duration alone."""
    injections = draw_injections(n_cbc=5, n_glitch=20, duration=3600.0,
                                 start_gps=1000.0, edge_pad=200.0, seed=3)
    assert injections
    end = -np.inf
    for spec in injections:
        start = spec["gps"] - spec["support_before"]
        assert start >= end
        end = spec["gps"] + spec["support_after"]
    assert end <= 1000.0 + 3600.0


def test_draw_injections_stay_inside_the_padded_span():
    injections = draw_injections(n_cbc=4, n_glitch=10, duration=2400.0,
                                 start_gps=500.0, edge_pad=64.0, seed=5)
    for spec in injections:
        assert spec["gps"] - spec["support_before"] >= 500.0 + 64.0
        assert spec["gps"] + spec["support_after"] <= 500.0 + 2400.0 - 64.0


def test_draw_injections_is_reproducible_from_seed():
    a = draw_injections(n_cbc=3, n_glitch=8, duration=2400.0, edge_pad=100.0, seed=11)
    b = draw_injections(n_cbc=3, n_glitch=8, duration=2400.0, edge_pad=100.0, seed=11)
    assert [x["subclass"] for x in a] == [x["subclass"] for x in b]
    assert [x["gps"] for x in a] == [x["gps"] for x in b]


def test_generate_dataset_writes_frames_and_ground_truth(tmp_path):
    outdir = str(tmp_path)
    table = generate_dataset(outdir, duration=1200.0, start_gps=1400000000.0,
                             n_cbc=3, n_glitch=8, seed=2, edge_pad=100.0)
    for name in ("H1-MOCK-FOREGROUND.ffl", "L1-MOCK-FOREGROUND.ffl",
                 "H1-MOCK-BACKGROUND.ffl", "L1-MOCK-BACKGROUND.ffl",
                 "injections.parquet"):
        assert os.path.getsize(os.path.join(outdir, name)) > 0
    for name in ("H1-MOCK-FOREGROUND", "L1-MOCK-BACKGROUND"):
        frames = os.listdir(os.path.join(outdir, name))
        assert frames and all(f.endswith(".gwf") for f in frames)
    assert len(table) > 0
    assert set(table["category"]) <= {"cbc", "glitch"}


def test_cbc_injections_are_coincident_and_glitches_are_not(tmp_path):
    table = generate_dataset(str(tmp_path), duration=2400.0, start_gps=1400000000.0,
                             n_cbc=6, n_glitch=6, seed=4, edge_pad=100.0,
                             write_background=False)
    cbc = table[table["category"] == "cbc"]
    assert len(cbc) > 0
    dt = (cbc["gps_H1"] - cbc["gps_L1"]).abs()
    assert (dt <= LIGHT_TRAVEL_H1L1 + 1e-6).all()
    assert cbc[["snr_H1", "snr_L1"]].notna().all().all()

    # A glitch lives in one detector; the other one records a zero contribution,
    # not a missing value, so it is counted by amplitude and not by presence.
    glitches = table[table["category"] == "glitch"]
    single = (glitches[["snr_H1", "snr_L1"]] > 0.0).sum(axis=1)
    assert (single == 1).all()


def test_network_snr_is_the_quadrature_sum_for_cbc(tmp_path):
    table = generate_dataset(str(tmp_path), duration=2400.0, start_gps=1400000000.0,
                             n_cbc=6, n_glitch=0, seed=6, edge_pad=100.0,
                             write_background=False)
    cbc = table[table["category"] == "cbc"]
    quad = np.sqrt(cbc["snr_H1"] ** 2 + cbc["snr_L1"] ** 2)
    assert np.allclose(quad, cbc["network_snr"], rtol=1e-6)


def test_foreground_differs_from_background_only_where_injected(tmp_path):
    from gwpy.timeseries import TimeSeries

    outdir = str(tmp_path)
    table = generate_dataset(outdir, duration=1200.0, start_gps=1400000000.0,
                             n_cbc=2, n_glitch=4, seed=8, edge_pad=100.0)

    def read(kind):
        directory = os.path.join(outdir, f"H1-MOCK-{kind}")
        frames = sorted(os.path.join(directory, f) for f in os.listdir(directory))
        return TimeSeries.read(frames, channel="H1:MOCK-STRAIN")

    fg, bg = read("FOREGROUND"), read("BACKGROUND")
    diff = np.asarray(fg) - np.asarray(bg)
    assert np.abs(diff).max() > 0
    assert (np.abs(diff) > 0).mean() < 0.5


def test_band_limit_removes_power_below_the_cutoff():
    """Glitch shapes carry power down to zero frequency; the noise does not."""
    from wdf.mock.dataset import band_limit

    y = w.gaussian(sigma_t=0.01, sample_rate=FS)
    freqs = np.fft.rfftfreq(len(y), 1.0 / FS)
    low = freqs < 15.0

    before = np.abs(np.fft.rfft(y))[low].sum()
    after = np.abs(np.fft.rfft(band_limit(y, FS, 15.0)))[low].sum()
    assert after < 0.05 * before


def test_band_limit_keeps_the_signal_in_band():
    from wdf.mock.dataset import band_limit

    y = w.sine_gaussian(f0=200.0, q=12.0, sample_rate=FS)
    filtered = band_limit(y, FS, 15.0)
    assert np.abs(filtered).max() == pytest.approx(np.abs(y).max(), rel=0.1)


def test_band_limit_does_not_shift_the_signal_in_time():
    """Zero-phase filtering, so the injection stays where the truth says it is.

    Compared by energy centroid: the peak sample of an oscillating burst sits on
    one of its cycles and can hop a half period for a negligible amplitude change.
    """
    from wdf.mock.dataset import band_limit

    def centroid(x):
        e = np.asarray(x) ** 2
        return float((np.arange(len(e)) * e).sum() / e.sum())

    y = w.sine_gaussian(f0=200.0, q=12.0, sample_rate=FS)
    shift_s = abs(centroid(band_limit(y, FS, 15.0)) - centroid(y)) / FS
    assert shift_s < 1e-3


def test_band_limit_at_nyquist_is_a_high_pass():
    """A band reaching Nyquist is a high pass, not a bandpass one bin below it.

    A full-band Butterworth bandpass puts its poles on top of z = +-1 and its
    initial-condition solve is singular, so the top of the band has to be read
    before it is nudged inside Nyquist.
    """
    from wdf.mock.dataset import band_limit

    y = w.gaussian(sigma_t=0.01, sample_rate=FS)
    filtered = band_limit(y, FS, 5.0, high_frequency_cutoff=0.5 * FS)
    assert np.isfinite(filtered).all()
    assert filtered == pytest.approx(band_limit(y, FS, 5.0), rel=1e-9, abs=1e-30)


def test_start_of_data_is_left_free_for_noise_estimation():
    """A search estimating its noise model from the first minutes needs them clean."""
    pad = 500.0
    injections = draw_injections(n_cbc=5, n_glitch=20, duration=3600.0,
                                 start_gps=1000.0, edge_pad=pad, seed=13)
    first = min(s["gps"] - s["support_before"] for s in injections)
    assert first >= 1000.0 + pad


@pytest.mark.parametrize("name", sorted(w.GLITCH_GENERATORS))
def test_injected_power_stays_inside_the_analysed_band(name):
    """Ground truth counts SNR the search can see: injections must not carry
    power above the analysis Nyquist, which sits below the generation Nyquist.
    """
    fs, analysis_nyquist = 4096, 1024.0
    y = w.GLITCH_GENERATORS[name](sample_rate=fs)
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1.0 / fs)
    above = spectrum[freqs > analysis_nyquist].sum()
    assert above < 1e-3 * spectrum.sum()


@pytest.mark.parametrize("rate", [2048, 4096])
def test_glitch_duration_is_independent_of_sample_rate(rate):
    """Waveforms follow the data's sampling rate without changing shape."""
    y = w.sine_gaussian(f0=150.0, q=12.0, sample_rate=rate)
    assert len(y) / rate == pytest.approx(0.102, abs=0.005)
