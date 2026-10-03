"""The local cosine bases (``LocalCos<M>``) as candidates of the competition: their
geometry, their inversion, and a worker run with `Bases` and `MinFrequency`."""
import glob
import json

import numpy as np
import pytest

from wdf.analysis.wavelets import (coeff_freq_bands, coeff_levels, coeff_time_bounds, mother_wavelet,
                                   packet_depth, packet_depths)

FS = 2048.0
N = 1024


@pytest.mark.parametrize("name, depth, mother", [
    ("LocalCos128", 7, "LocalCos"), ("LocalCos64", 6, "LocalCos"), ("LocalCos2", 1, "LocalCos"),
    ("LocalCos1024", 10, "LocalCos"),
    ("Coif1P6", 6, "Coif1"), ("DaubC8", 0, "DaubC8"), ("Sym8P3", 3, "Sym8"), ("Haar", 0, "Haar"),
])
def test_names_declare_their_layout(name, depth, mother):
    assert packet_depth(name) == depth
    assert mother_wavelet(name) == mother


@pytest.mark.parametrize("name", ["LocalCos", "LocalCos0", "LocalCos1", "LocalCos100", "LocalCos064",
                                  "LocalCosX"])
def test_a_bad_local_cosine_name_is_refused(name):
    with pytest.raises(ValueError):
        packet_depth(name)
    with pytest.raises(ValueError):
        mother_wavelet(name)


def test_a_column_of_mixed_names():
    assert packet_depths(["Haar", "LocalCos128", "Coif1P6", "LocalCos128"]).tolist() == [0, 7, 6, 7]


def test_local_cosine_geometry_is_frequency_major():
    # LocalCos128 at N 1024: 128 rows of fs / 256 = 8 Hz, 8 segments of 128 samples (62.5 ms) per row;
    # index k * 8 + j is DCT-IV bin k of segment j.
    f_lo, f_hi = coeff_freq_bands(N, FS, "LocalCos128")
    t_lo, t_hi = coeff_time_bounds(N, FS, "LocalCos128")
    k, j = np.divmod(np.arange(N), N // 128)
    assert np.allclose(f_lo, 8.0 * k) and np.allclose(f_hi, 8.0 * (k + 1))
    assert np.allclose(t_lo, j * 128 / FS) and np.allclose(t_hi, (j + 1) * 128 / FS)
    level, position = coeff_levels(N, "LocalCos128")
    assert np.array_equal(position, j)
    # the octave row of each tile holds its frequency on the pyramid ladder
    pyramid_lo, pyramid_hi = coeff_freq_bands(N, FS)
    centre = 0.5 * (f_lo + f_hi)
    rows = np.unique(np.stack([pyramid_lo, pyramid_hi], 1), axis=0)
    for lv, c in zip(level[::8], centre[::8]):
        lo, hi = rows[lv + 1]
        assert lo <= c <= hi or c <= rows[0][1]


def test_the_geometry_places_what_the_transform_puts_there():
    tsa = pytest.importorskip("py4tsa.tsa")
    if not hasattr(tsa.WaveletTransform, "LocalCos"):
        pytest.skip("the installed py4tsa has no local cosine basis")
    from _synth import forward
    t = np.arange(N) / FS
    for f0, segment in ((100.0, 2), (412.0, 5), (37.0, 6)):
        burst = np.where((t >= segment * 128 / FS) & (t < (segment + 1) * 128 / FS),
                         np.sin(2 * np.pi * f0 * t) * np.sin(np.pi * (t * FS - segment * 128) / 128) ** 2, 0.0)
        c = forward(burst, "LocalCos128")
        i = int(np.argmax(np.abs(c)))
        f_lo, f_hi = coeff_freq_bands(N, FS, "LocalCos128")
        t_lo, t_hi = coeff_time_bounds(N, FS, "LocalCos128")
        assert f_lo[i] - 8.0 <= f0 <= f_hi[i] + 8.0
        assert t_lo[i] == pytest.approx(segment * 128 / FS)


@pytest.mark.parametrize("name", ["LocalCos128", "LocalCos64", "LocalCos16", "Coif1P6", "DaubC8"])
def test_reconstruction_round_trip(name):
    tsa = pytest.importorskip("py4tsa.tsa")
    if name.startswith("LocalCos") and not hasattr(tsa.WaveletTransform, "LocalCos"):
        pytest.skip("the installed py4tsa has no local cosine basis")
    from _synth import forward
    from wdf.analysis.reconstruction import inverse_transform
    x = np.random.default_rng(5).normal(size=N)
    c = forward(x, name)
    assert np.sum(c ** 2) == pytest.approx(np.sum(x ** 2), rel=1e-12)          # orthonormal
    assert np.max(np.abs(inverse_transform(c, name) - x)) < 1e-11


def test_the_worker_searches_local_cosine_with_a_minimum_frequency(tmp_outdir):
    from _synth import packets_or_skip
    from conftest import run_segment_process
    from wdf.analysis.coefficients import to_dense

    packets_or_skip()
    tsa = pytest.importorskip("py4tsa.tsa")
    if not hasattr(tsa.WDF2Classify, "SetMinFrequency"):
        pytest.skip("the installed py4tsa has no minimum frequency")
    bases = ["Haar", "DaubC8", "Sym8", "Coif5", "LocalCos128", "LocalCos64"]
    triggers = run_segment_process(tmp_outdir, window=1024, overlap=256, Bases=",".join(bases),
                                   MinFrequency=16.0)
    assert len(triggers)
    assert set(triggers.wave) <= set(bases)
    assert {"LocalCos128", "LocalCos64"} & set(triggers.wave)
    [used] = glob.glob(f"{tmp_outdir}**/parametersUsed-Win1024.json", recursive=True)
    with open(used) as fh:
        dumped = json.load(fh)
    assert dumped["bases"] == bases and dumped["Bases"] == bases
    assert dumped["MinFrequency"] == 16.0
    for _, row in triggers.iterrows():
        dense = to_dense(row.wt_index, row.wt_value, int(row.n_coeff))
        _, f_hi = coeff_freq_bands(int(row.n_coeff), float(row.fs), row.wave)
        assert not np.any(dense[f_hi <= 16.0 + 1e-9])
        assert np.any(dense[f_hi > 16.0])


def test_without_a_minimum_frequency_the_search_is_the_core_s_own(tmp_outdir):
    from _synth import packets_or_skip
    from conftest import run_segment_process

    import os

    packets_or_skip()
    runs = []
    for sub, extra in (("a/", {}), ("b/", {"MinFrequency": 0.0})):
        os.makedirs(tmp_outdir + sub)
        runs.append(run_segment_process(tmp_outdir + sub, window=1024, overlap=256, Bases="Haar,LocalCos128",
                                        **extra))
    a, b = runs
    assert len(a) == len(b)
    assert (a.wave.values == b.wave.values).all()
    assert np.array_equal(a.EnWDF.values, b.EnWDF.values)


def test_use_min_frequency_needs_the_core():
    from wdf.processes.wdf import use_min_frequency

    class Old:
        pass

    assert use_min_frequency(Old(), 0.0, FS) == 0.0
    with pytest.raises(RuntimeError, match="SetMinFrequency"):
        use_min_frequency(Old(), 16.0, FS)

    class Core:
        def SetMinFrequency(self, f, fs):
            self.args = (f, fs)

    core = Core()
    assert use_min_frequency(core, "16", 2048) == 16.0 and core.args == (16.0, 2048.0)


def test_use_block_remainder():
    from wdf.processes.wdf import use_block_remainder

    class Old:
        pass

    assert use_block_remainder(Old(), None) == "legacy"
    assert use_block_remainder(Old(), "legacy") == "legacy"
    with pytest.raises(RuntimeError, match="SetBlockRemainder"):
        use_block_remainder(Old(), "merge")
    with pytest.raises(ValueError, match="BlockRemainder"):
        use_block_remainder(Old(), "join")

    class Core:
        def SetBlockRemainder(self, mode):
            self.mode = mode

    core = Core()
    assert use_block_remainder(core, " Merge ") == "merge" and core.mode == "merge"


def test_the_worker_merges_the_block_remainders(tmp_outdir):
    import os

    from _synth import packets_or_skip
    from conftest import run_segment_process

    packets_or_skip()
    tsa = pytest.importorskip("py4tsa.tsa")
    if not hasattr(tsa.WDF2Classify, "SetBlockRemainder"):
        pytest.skip("the installed py4tsa has no block remainder option")
    runs = {}
    for mode in ("legacy", "merge"):
        os.makedirs(tmp_outdir + mode)
        runs[mode] = run_segment_process(tmp_outdir + mode + "/", window=1024, overlap=256,
                                         Bases="Haar,Sym8,LocalCos128", MinFrequency=16.0, BlockRemainder=mode)
        [used] = glob.glob(f"{tmp_outdir}{mode}/**/parametersUsed-Win1024.json", recursive=True)
        with open(used) as fh:
            assert json.load(fh)["BlockRemainder"] == mode
    # a merged remainder is judged with its block, so fewer lone coefficients survive in noise
    assert len(runs["merge"]) < len(runs["legacy"])
    os.makedirs(tmp_outdir + "default")
    default = run_segment_process(tmp_outdir + "default/", window=1024, overlap=256, Bases="Haar,Sym8,LocalCos128",
                                  MinFrequency=16.0)
    assert len(default) == len(runs["legacy"])
    assert np.array_equal(default.EnWDF.values, runs["legacy"].EnWDF.values)
