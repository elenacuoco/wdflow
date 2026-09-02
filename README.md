# wdflow

[![Documentation](https://readthedocs.org/projects/wdflow/badge/?version=latest)](https://wdflow.readthedocs.io/en/latest/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22025594-blue.svg)](https://doi.org/10.5281/zenodo.22025594)

A maintained, standalone package for the WDF (Wavelet Detection Filter) trigger-generation
pipeline for transient time-series signals, built on the C++ core
[p4TSA](https://github.com/elenacuoco/p4TSA) (exposed to Python as `pytsa`), plus the downstream
trigger analysis that turns raw per-window triggers into candidate events.

**It runs in real time**, at a latency that is fixed and known before the filter
runs, and its per-window arithmetic -- an autoregressive filter of known order,
an orthonormal transform of a power-of-two window, a threshold and a sum of
squares, with no iteration to convergence, no dynamic memory and no
data-dependent branching -- is simple enough to put on an FPGA. No template and
no signal model: every trigger carries the coefficients that produced it, so
what a transient is can be decided afterwards.

## Layout

- `wdf.config`, `wdf.processes`, `wdf.observers`, `wdf.structures` -- trigger generation. Needs
  the compiled `pytsa`/p4TSA core (`pip install -e ".[pipeline]"`).
- `wdf.analysis` -- clustering, multi-detector coincidence (classical + GNN), background/
  false-alarm-probability, ROC analysis, sky localisation, and the submission writer that
  puts the surviving candidates in a challenge's own columns. Operates on plain pandas
  DataFrames / saved trigger files, no `pytsa` dependency, so it works standalone
  (`pip install -e .`, no extras needed).
- `wdf.mock` -- the simulated two-detector data set: coloured Gaussian noise, compact-binary
  injections projected through the antenna responses, single-detector glitch morphologies, and
  waveforms read from a catalogue when a class has no closed form. Paired foreground and
  background frames with a truth table (`pip install -e ".[mock]"`).

## Relationship to the legacy `wdf` package

`wdflow` carries forward the parts of the retired `wdf` package
that are actually used by the real trigger-search pipeline (`wdfUnitDSWorker` and everything it
calls) -- same file/class/module names and shapes, no gratuitous renaming -- plus the
`wdf.analysis` layer (formerly a separate `wdfLib` package), merged in here as one namespace to
install and import from.

Changes from the legacy `wdf` package:

- **AR-whitening lookahead is fixed-size, independent of `par.len`.** `DoubleWhitening`'s backward
  pass needs a lookahead buffer of real future data to settle before producing a good estimate for
  the current chunk. This lookahead (`WhiteningExtraSize`, default 20 resampled-rate seconds) is
  now a fixed size, decoupled from `par.len` (the streaming chunk size, an I/O batching/throughput
  knob). Set `parameters.WhiteningExtraSize = 0` to reproduce the legacy behavior.

- **The detection loop stops `par.len` seconds before the requested segment end** (unchanged from
  the legacy package): it checks the previous read's start before issuing the next one, so its
  last read can extend up to `par.len` past the checked bound, and `FrameIChannel` does not always
  raise cleanly when asked to read past the last available frame data. This margin keeps the last
  read within legitimately available data. A larger `par.len` therefore analyzes up to `par.len`
  fewer seconds at a segment's tail.

- **Per-trigger statistics remain somewhat sensitive to `par.len`** even with the fixed whitening
  lookahead above: `WaveletThreshold`'s `dohonojohnston` mode recomputes its detection threshold
  fresh per window from that window's own coefficient median, so triggers whose statistic sits
  close to that threshold can flip in or out of detection depending on floating-point roundoff
  accumulated differently across different chunk sizes. This is an inherent sensitivity of any
  hard/soft-threshold statistic to input near its decision boundary, not a data-corruption bug --
  triggers well away from threshold are unaffected.

- **`EnWDF` is the detection statistic, everywhere.** It is the norm of a window's surviving
  wavelet coefficients on the noise scale, `||c||/sigma`, computed in p4TSA and carried on every
  trigger. The candidate bases are orthonormal, so by Parseval this is also `||x_hat||/sigma`, the
  matched-filter signal-to-noise ratio of the reconstructed transient. Clustering, coincidence,
  GNN scoring and the false-alarm-probability estimate all rank on it, per window, per event and
  per network -- foreground and background have to be ranked on the same quantity for a
  probability to mean anything.

  `snrPeak` and `snrMean` sit alongside it as amplitudes on the same scale, `max|x|/sigma` and the
  r.m.s. of `x` over its own support divided by `sigma`. `EnWDF` grows as the square root of the
  number of samples a transient occupies, as an energy statistic must; the other two do not.

- **Every frequency a trigger reports is a moment over the wavelet tiles**, not a periodogram.
  `freqMean` is the energy-weighted band, taken in the log-frequency the dyadic tiling is uniform
  in; `freqMin`/`freqMax` are the support the surviving tiles cover, which is what the band
  overlap tests read; `freqQ05`/`freqQ95` are the band the energy occupies, which follows the
  signal rather than its faintest coefficient. There is no single peak frequency: a band an octave
  wide is what the transform can resolve, and naming one frequency inside it would state more than
  was measured.

- **Events are clustered on the wavegram; the energy that measures is not the statistic that
  selects.** `wdf.analysis.clustering.wavegram_events` percolates over the time-frequency tiles of
  the surviving coefficients to decide which windows belong to the same transient. The event's
  energy is then `||x_hat||/sigma` over the stitched reconstruction of those windows, which counts
  each sample once and is what the event is worth. What ranks it for detection is its loudest
  block: a hard threshold admits every tile at a floor of `2 ln N` in normalised energy, so a sum
  over tiles accumulates that floor in the noise as well as in the signal, while a maximum over
  blocks the search has already scored has a background that is a subset of the ungrouped one ---
  the grouping reduces the trials factor without being able to lose a candidate. `TriggerClusterer`
  (DBSCAN or a greedy merge on the per-window scalar summaries) is kept as a cross-check.

- **A coincidence is timed on the reconstructions, below the tile.**
  `wdf.analysis.timing.arrival_time_difference` places the two events' stitched reconstructions
  on one absolute time grid and reads the arrival-time difference at the lag that maximises
  their cross-correlation, with an uncertainty the pair itself declares (the half-width of the
  correlation peak). The instants a trigger records are moments of the tiling and cannot resolve
  better than the tile they sit on; the reconstruction carries the waveform at the sample, and
  the estimator runs downstream on the coefficients the trigger already carries, adding nothing
  to the front end.

- **Trigger output is Parquet, not CSV** (`wdf.observers.SingleEventPrintFileObserver`), written
  incrementally in row-group batches (`flush_every`, default 500 triggers) and finalized by a
  `close()` call at the end of `segmentProcess`. `wdf.analysis.io`'s loaders accept both
  `*.parquet` (default) and `*.csv` (for older runs), dispatching on file extension.

- **The downstream analysis layer (formerly the separate `wdfLib` package) is merged in** as the
  `wdf.analysis` subpackage: clustering, multi-detector coincidence (classical + GNN),
  background/false-alarm-probability, and ROC analysis. It has no `pytsa` dependency and operates
  on plain pandas DataFrames / saved trigger files, so it works standalone.

Left behind deliberately (not used by `wdfUnitDSWorker`'s pipeline, not ported or audited):
`AdaptiveWhitening`, `Coloring`, `createsegmentsMinMax`, `CreateSegments`, `DownSamplingLF`,
`DownSampling`, `StateVectorSegments`, `wdf_reconstruct`, `wdfUnitBPDSWorker`, `wdfUnitWorker`
(the last two share the pre-fix `ExtraSize=0` whitening issue -- worth the same fix if/when
ported), `structures.ClusteredEvent` (an empty data holder), `structures.segment`, `utility.*`.

## Tutorials

`tutorials/` holds four runnable notebooks. Everything in them is built in the notebook itself --
no data set to download, no frame file to point at:

1. `01_the_statistic_and_the_parameters` -- one window of whitened data, the transform,
   thresholding, basis competition, and what the estimated parameters mean. Needs `pytsa`.
2. `02_the_wavegram_and_long_signals` -- reading parameters off the time-frequency tiles, and
   recovering a signal that spans several analysis windows. Needs `pytsa`.
3. `03_coincidence_and_significance` -- coincidence, time-slide background, false-alarm
   probability and ROC, using only `wdf.analysis`. **No `pytsa` required.**
4. `04_reconstruction_and_phase` -- the event's own coefficients inverted and stitched across
   windows, compared against the waveform that was injected, and the phase read sample by
   sample. Needs `pytsa`.

## Install

### Requirements

Python 3.10 or newer. `wdflow` is not on an index: it installs from a checkout
of this repository, and everything it depends on except the compiled core comes
from PyPI.

```bash
git clone https://github.com/elenacuoco/wdflow
cd wdflow
pip install -e ".[all]"
```

**Always installed** — the analysis layer runs on these alone:

| Package | Used for |
|---------|----------|
| `numpy`, `scipy` | arrays, signal processing |
| `pandas >= 2.0` | trigger and event tables |
| `pyarrow` | reading and writing the Parquet trigger files |
| `h5py` | the AR and lattice-filter coefficient files |
| `scikit-learn >= 1.2` | DBSCAN, in `TriggerClusterer` |
| `matplotlib >= 3.7` | the report figures |

**Optional groups**, each installed with `pip install -e ".[name]"`:

| Group | Packages | Needed for |
|-------|----------|------------|
| `gnn` | `torch >= 2.1`, `torch_geometric >= 2.5` | `wdf.analysis.gnn` — the learned cross-detector coincidence |
| `data` | `gwpy >= 3.0` | fetching public strain, e.g. from GWOSC |
| `pipeline` | `coloredlogs` | trigger-generation logging |
| `mock` | `pycbc` | `wdf.mock` — generating the simulated data sets: CBC injections, glitch morphologies, catalogue waveforms |
| `tutorials` | `jupyter`, `nbclient`, `ipykernel` | running `tutorials/` |
| `docs` | `sphinx >= 7`, `sphinx-rtd-theme >= 2`, `myst-nb >= 1` | building the documentation |
| `dev` | `pytest >= 7` | the test suite |
| `all` | all of the above | |

**Not from an index:**

| Requirement | Needed for | Where it comes from |
|-------------|------------|---------------------|
| `pytsa` (p4TSA) | `wdf.processes`, `wdf.observers` — trigger generation, and any wavelet transform | built from [p4TSA](https://github.com/elenacuoco/p4TSA) |
| a GWF backend for gwpy | reading frame files through gwpy | `pip install lalsuite`, or conda's `python-ldas-tools-framecpp` |

### The compiled core

Trigger generation needs p4TSA, imported as `pytsa`. It is deliberately **not**
declared as a dependency of any extra: p4TSA has no PyPI distribution — FrameL
has no wheel — so the declaration could not resolve, and **`pip install pytsa`
is a different project** (an unrelated Python decorator library). Build it from
[p4TSA](https://github.com/elenacuoco/p4TSA) instead, with its conda recipe or
`pip install .` from a checkout. p4TSA in turn needs GSL, FFTW3, FrameL, the
Boost.uBLAS headers and the Cereal headers, all on conda-forge.

If `import pytsa` behaves oddly, check what you actually have:

```bash
pip show p4tsa
python -c "import pytsa; print(pytsa.__file__)"   # must be a compiled .so, not a .py
```

### The legacy `wdf` package must not be installed alongside

`wdflow`'s importable package is `wdf`, the same top-level name the retired
`wdf` package used. Whichever of the two sits directly in `site-packages` wins,
and it shadows even an editable install of the other, silently:

```bash
python -c "import wdf; print(wdf.__file__)"   # must point at your wdflow checkout
pip uninstall wdf                             # if it points into site-packages
```

## What comes next

Parameter estimation by normalising flow, on the coefficient matrix an event
already carries. The coefficients a trigger keeps are a fixed-length description
of the transient on the time-frequency plane, already on the noise scale, so a
flow can be conditioned on them directly without a spectrogram in between. That
is where the question of what a transient *is* belongs -- after the selection,
never inside it -- and it is the reason the search stores the coefficients
rather than a summary of them.

## Contributing

Changes reach `master` through pull requests only, and a pull request merges
only once CI is green. See [`CONTRIBUTING.md`](https://github.com/elenacuoco/wdflow/blob/master/CONTRIBUTING.md).

## How to cite

**Use of this code in published work requires citation of the following.**

*The Wavelet Detection Filter:*

- E. Cuoco, *The Wavelet Detection Filter: a real-time un-modelled search for
  gravitational wave transients, ranking coincidences with a graph neural
  network*, (2026), in preparation. VIR-0605A-26

*WDFX:*

- E. Cuoco, M. Razzano, A. Utina, *Wavelet-based classification of transient
  signals for gravitational wave detectors*, 26th European Signal Processing
  Conference (EUSIPCO), 2648–2652 (2018).
  [10.23919/EUSIPCO.2018.8553393](https://doi.org/10.23919/EUSIPCO.2018.8553393)

*Time-domain whitening, which the conditioning stage implements:*

- E. Cuoco *et al.*, *On-line power spectra identification and whitening for the
  noise in interferometric gravitational wave detectors*, Class. Quantum Grav.
  **18**, 1727 (2001).
  [10.1088/0264-9381/18/9/309](https://doi.org/10.1088/0264-9381/18/9/309)
- E. Cuoco *et al.*, *Noise parametric identification and
  whitening for LIGO 40-m interferometer data*, Phys. Rev. D **64**, 122002
  (2001).
  [10.1103/PhysRevD.64.122002](https://doi.org/10.1103/PhysRevD.64.122002)

`CITATION.cff` in this repository carries the same list in machine-readable
form; GitHub's *Cite this repository* button reads it.

## Use of generative AI

The Wavelet Detection Filter, the `p4TSA` core it runs on and the design of this
pipeline are the author's own work. Claude (Opus 5, Anthropic), used through
Claude Code, wrote parts of the implementation, its tests and its documentation,
and produced the logo and stylesheet of the project page. Every such
contribution was reviewed and is covered by the test suite that runs in CI: the
golden-output fixture pins trigger generation end to end, so a generated change
that moves the numerics fails the build rather than passing silently.
Responsibility for the method and for everything published here rests with the
author.

## Status

`tests/` includes a golden-output regression fixture pinning trigger generation end to end on a
small synthetic frame (the legacy `wdf` package has none), plus the `wdf.analysis` test suite
(synthetic trigger data, no WDF run required):

```bash
pytest tests                    # needs pytsa for the golden fixture
pytest tests/test_clustering.py tests/test_coincidence.py tests/test_significance.py   # no pytsa
```

`tests/test_gnn.py` needs the `gnn` extra and `tests/test_mock_dataset.py` needs `pycbc`; both are
skipped by deselecting them if those are not installed.
