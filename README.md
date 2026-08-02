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

Functional changes from the legacy `wdf` package:

- **Fixed, `par.len`-independent AR-whitening lookahead.** `wdfUnitDSWorker.segmentProcess` used
  to hardcode `DoubleWhitening`'s lookahead window (`ExtraSize=0`), which made the whitened signal
  -- and therefore trigger counts and per-trigger statistics -- depend on `par.len` (the streaming
  chunk size), a value that should only affect I/O batching/throughput. `WhiteningExtraSize`
  (default: 20 resampled-rate seconds) fixes the lookahead at a size large enough for
  `DoubleWhitening`'s backward pass to settle, independent of `par.len`. Set
  `parameters.WhiteningExtraSize = 0` to reproduce the old behavior exactly.

- **`gpsEnd - par.len` segment-coverage margin: understood, kept as-is.** The main detection loop
  checks the *previous* read's start before issuing the next, so its last read can extend up to
  `par.len` past the checked bound; the `- par.len` margin exists to keep that last read within
  legitimately available frame data. Attempted removing/shrinking this margin twice this session
  (once outright, once via an adaptive shorter last chunk) -- both broke real usage (the first
  crashed downstream on a degenerate all-zero window when `FrameIChannel` didn't raise cleanly
  near a segment's true end; the second silently dropped ~2.4% of triggers on the golden-output
  fixture via `DWhitening.SetOutputSize` interaction, not yet understood). Reverted both times.
  Net effect: a larger `par.len` does still analyze up to `par.len` fewer seconds at a segment's
  tail -- a real, currently-accepted tradeoff, not a bug to casually "fix" without a lot more
  care than a config-level change deserves.

- **Remaining `par.len` sensitivity, characterized (not a bug to fix further)**: on a real 450s
  H1 GW250114 segment (`ARorder=3000`), triggers matched by `gpsPeak` between `par.len=10` and
  `par.len=150` agree exactly (`snrPeak`/`EnWDF` bit-identical) for ~88% of triggers; the
  remaining ~12% -- concentrated in triggers whose statistic sits close to a threshold crossing
  -- differ, sometimes sharply (including flipping to/from exactly 0). Root cause: processing the
  same signal through a different number/size of `DWhitening.Process()` calls accumulates
  floating-point roundoff differently (summation/filtering isn't perfectly associative), at a
  level far below any physical significance -- but `WaveletThreshold`'s `dohonojohnston` mode
  recomputes its threshold fresh per window from that window's own coefficient median, so a
  trigger sitting right at that boundary can flip. This is an inherent sensitivity of any
  hard/soft-threshold statistic to input near its decision boundary, not an identifiable
  additional bug; total trigger counts agree to ~1% (912 vs 924 on the same test segment).

- **Trigger SNR is now the wavelet-coefficient energy statistic**
  (`wdf.processes.wavelet_energy.wavelet_energy_snr`), computed from each trigger's own (already
  thresholded) wavelet coefficients against the Donoho-Johnstone universal threshold (Donoho &
  Johnstone, 1994) on the AR-whitening noise scale, replacing the old
  reconstructed-waveform-based `snrMean`/`snrPeak`. `EnWDF` (WDF's own internal per-window,
  per-basis, locally-normalized detection statistic -- the value `par.threshold` gates trigger
  emission on) is still recorded per trigger for diagnostics, but is not the statistic
  `wdf.analysis`'s clustering/coincidence/GNN code ranks or characterizes triggers by.

- **Trigger output is Parquet, not CSV** (`wdf.observers.SingleEventPrintFileObserver`), written
  incrementally in row-group batches (`flush_every`, default 500 triggers) and finalized by a
  `close()` call at the end of `segmentProcess`. `wdf.analysis.io`'s loaders accept both
  `*.parquet` (default) and `*.csv` (for older runs), dispatching on file extension.

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
