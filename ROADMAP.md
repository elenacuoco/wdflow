# Roadmap

What is deliberately left for after the current release, and why. Each entry
says what the change is, what it touches, and what would have to be true for it
to be worth doing.

## 1. Store the coefficients sparsely

**The observation.** Thresholding is what the search does: coefficients below
the Donoho–Johnstone universal threshold are set to zero, and almost all of them
are. Measured on the simulated data set, 512 coefficients per window at a
collection threshold of 5:

| | |
|---|---|
| non-zero per trigger, median | **2** |
| non-zero per trigger, mean | 2.6 |
| density | **0.51 %** |
| 99 % of triggers have at most | 13 non-zero |

A representation carrying `(index, value)` pairs instead of the full vector
holds about **130× fewer numbers**. Everything currently writes and reads 512
columns of which roughly 510 are zero.

**On disk that factor does not survive, and the measurement says so.** Written
out and compared on the same 20 000 real triggers at a window of 512, the whole
file goes from 2.42 MB to 1.78 MB -- **1.4×**, not 130×. Parquet had already
taken most of it: the 512 dense coefficient columns cost 0.83 MB of the 2.42,
because a column of exact zeros encodes to almost nothing, while each scalar
metadata column of incompressible floats costs about 0.2 MB on its own. The
file was never dominated by the coefficients. Dropping the reconstructed
waveform is where a large factor genuinely is -- `rw*` is incompressible, and
against `fullPrint=3` the saving is 8.3×.

So the size argument is a small one, and the reason to make the change is the
other one below.

**Why it is more than an optimisation.** The sparse pairs are what the algorithm
actually produces; the dense vector is an artefact of how it is written down. A
real-time implementation makes this concrete — on an FPGA the thing to transport
downstream is a handful of `(index, value)` pairs per window, not a fixed-length
vector, and the bandwidth that costs is the design constraint. Getting the
software format right first means the two agree, and that the offline analysis
is reading the same object the hardware would emit.

**What it touches.**

- `wdf/observers/SingleEventPrintFileObserver.py` — builds the column list from
  `par.Ncoeff` and writes one column per coefficient. This is the only writer,
  and it lives here rather than in p4TSA, so the C++ core does not change.
- `wdf/analysis/io.py` — `_read_trigger_file` and the single-precision cast.
- `wdf/analysis/cluster_coefficients.py` — `coefficient_columns` and
  `ClusterCoefficients.from_triggers`, which assemble the matrix.
- `wdf/analysis/clustering.py` — `collect_significant_pixels`, which already
  discards the zeros immediately after reading them.
- `wdf/analysis/reconstruction.py` and `robust_events.py` — both discover the
  `wt*` columns the same way.
- `wdf/analysis/report.py` — the tile plots.
- The notebooks in `wdf-detection-pipeline`, which read the columns directly.

Six modules and one writer, all going through two helpers that find the columns.
Introducing the sparse form behind those helpers, and keeping a reader for the
dense files already on disk, is the shape of the change: the rest of the code
asks for a coefficient vector and should not learn how it was stored.

**When.** After the current paper. Doing it now would invalidate every trigger
file produced so far, in the middle of the runs those results rest on, and the
comparison against a stable reference is what makes the change safe to verify.

**What is already done.** Both the stored matrices and the loaded frames use
single precision (`float32`), which halves them at no cost, and a segment's
coefficients are streamed one event at a time rather than held together. The
sparse form is the remaining factor.

## 2. Percolation clustering that scales

`WaveletPixelClusterer` builds a dense pairwise adjacency over pixels, so it is
O(n²) in memory and cannot run over a whole segment — its own docstring says so.
`cluster_detector_triggers` is the one that scales and is what the analysis uses.
Either the pixel clusterer gets a spatial index, or it should be documented as
an investigative tool for a short span and nothing more.

## 3. The learned coincidence on real noise

The graph coincidence is trained and evaluated on simulated data, where the
noise is stationary and Gaussian and the glitches come from analytic
morphologies. Real detector noise has neither property. Before any claim about
real data, the model has to be trained on injections into real noise, and its
score distribution on a real time-slide background compared against the
simulated one — if those disagree, that disagreement is the measurement.

## 4. Parameter estimation by normalising flow

On the `(n_triggers, n_coeff)` coefficient matrix an event already carries: the
same representation the coincidence and the reconstruction read. This is the
last piece, and the one the package is named for.
