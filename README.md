# wdflow

A maintained, standalone package for the WDF (Wavelet Detection Filter) trigger-generation
pipeline for transient time-series signals, built on the C++ core
[p4TSA](https://github.com/elenacuoco/p4TSA) (exposed to Python as `pytsa`), plus the downstream
trigger analysis that turns raw per-window triggers into candidate events.

## Layout

- `wdf.config`, `wdf.processes`, `wdf.observers`, `wdf.structures` -- trigger generation. Needs
  the compiled `pytsa`/p4TSA core (`pip install wdflow[pipeline]`).
- `wdf.analysis` -- clustering, multi-detector coincidence (classical + GNN), background/
  false-alarm-probability, and ROC analysis. Operates on plain pandas DataFrames / saved trigger
  files, no `pytsa` dependency, so it works standalone (`pip install wdflow`, no extras needed).

## Relationship to the legacy `wdf` package

`wdflow` carries forward the parts of the legacy [`wdf`](https://gitlab.com/wdfpipe/wdf) package
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

- **Trigger SNR is the wavelet-coefficient energy statistic**
  (`wdf.processes.wavelet_energy.wavelet_energy_snr`): each trigger's own (already thresholded)
  wavelet coefficients, summed in energy above the Donoho-Johnstone universal threshold (Donoho &
  Johnstone, 1994) on the AR-whitening noise scale. This replaces the legacy
  reconstructed-waveform-based `snrMean`/`snrPeak`. `EnWDF` (WDF's own internal per-window,
  per-basis, locally-normalized detection statistic -- the value `par.threshold` gates trigger
  emission on) is still recorded per trigger for diagnostics, but `wdf.analysis`'s
  clustering/coincidence/GNN code ranks and characterizes triggers by the energy statistic instead.

- **Trigger output is Parquet, not CSV** (`wdf.observers.SingleEventPrintFileObserver`), written
  incrementally in row-group batches (`flush_every`, default 500 triggers) and finalized by a
  `close()` call at the end of `segmentProcess`. `wdf.analysis.io`'s loaders accept both
  `*.parquet` (default) and `*.csv` (for older runs), dispatching on file extension.

- **The downstream analysis layer (formerly the separate `wdfLib` package) is merged in** as the
  `wdf.analysis` subpackage: clustering, multi-detector coincidence (classical + GNN),
  background/false-alarm-probability, and ROC analysis. It has no `pytsa` dependency and operates
  on plain pandas DataFrames / saved trigger files, so it works standalone.

Left behind for now (not used by `wdfUnitDSWorker`'s pipeline, not ported/audited):
`AdaptiveWhitening`, `Coloring`, `createsegmentsMinMax`, `CreateSegments`, `DownSamplingLF`,
`DownSampling`, `StateVectorSegments`, `wdf_reconstruct`, `wdfUnitBPDSWorker`, `wdfUnitWorker`
(the last two share the pre-fix `ExtraSize=0` whitening issue -- worth the same fix if/when
ported), `structures.ClusteredEvent` (an empty data holder), `structures.segment`, `utility.*`.

## Install

```bash
pip install -e ".[pipeline,gnn,data,dev]"
```

- `pipeline` -- `p4tsa` (trigger generation).
- `gnn` -- `torch` (`wdf.analysis.gnn`).
- `data` -- `gwpy` (GWOSC data fetching, notebook use).
- `dev` -- `pytest`.

`p4tsa`/`pytsa` is not declared as a hard dependency; it's conda-installed / built from source in
the environments this package targets, same convention as `wdf` and `wdf-detection-pipeline`.

## Status

`tests/` includes a golden-output regression fixture pinning a real short GWOSC segment's trigger
output (the legacy `wdf` package has none), plus the `wdf.analysis` test suite (synthetic trigger
data, no WDF run required).
