# Changelog

Versions follow [semantic versioning](https://semver.org). A release records
what the software does differently, not how it came to.

## Unreleased

### Two paths for stages 2 to 4: the paper path and the track path

WDF runs in four stages: triggers, single-detector reconstruction
(`Reconstruction`), the graph between detectors (`Coincidence`), and the
ranking of its candidate pairs (`Scorer`). `Reconstruction` and `Coincidence`
are "wavegram" (the default, the paper path) or "track", both the same
(`wdf.config.Parameters.network_path` refuses a mix). `Scorer` is "classic" or
"gnn" (with `GNNModel`). With the defaults the triggers, detector events and
network table are unchanged (tested against a snapshot).

- `wdf.analysis.network_driver`: `run_network` runs stages 2 to 4 on the path
  the configuration selects, with the paper path's sequence of calls
  (`NetworkComparisonBin`, `NetworkRankColumn`) and, with `NetworkSlides`, the
  time-slide background (`NetworkSlideStep`). `NetworkStream` holds the
  detectors' outputs online (`NetworkMemory`).
- `wdf.analysis.track_reconstruction`: each trigger seeds a track through the
  ridge map, from `TrackLookback` before its window to `TrackHorizon` after
  it, through the cell of its largest coefficient (`TrackSeedStates`), and the
  waveform along the track is reconstructed in `TrackBasis` (`LocalCos<M>` or
  `CosinePackets<m>`, the cosine-packet best basis with `TrackPacketCost`).
  The event table extends `DETECTOR_EVENT_COLUMNS` with the track's columns.
- `wdf.analysis.track_coincidence.TrackCoincidenceFinder`: the classical
  finder's admitted pairs, with track compatibility (`TrackTimingJitter`,
  `TrackFrequencyTolerance`) and the lag, coherence and phase of the complex
  cross-correlation of the waveforms. A time slide's displacement is applied
  to the tracks and waveforms. A GNN reads the track features through
  `cross_edge_dim`; the shipped models were fitted without them.
- With `Reconstruction = "track"` (and `Ridge`), `wdfUnitOnline` builds the
  tracks as the map completes (`TrackObserver`), declares
  `TrackLatency_s = Latency_s + TrackHorizon + RidgeSlot + (RidgeWindow -
  RidgeOverlap / 2) / rate`, and writes `TrackEvents` and `TrackWaveforms`;
  `wdfUnitOffline` builds them after the segment's ridge.
  `wdfUnitDSWorker` gains the `_observers` hook.
- `configs/online_zerophase_track.toml`: the online zero-phase configuration
  on the track path, with the map and the waveform in LocalCos64. The three
  other configurations carry the new parameters at their defaults.

### The shipped configurations search the 10 wavelet bases

`Bases` in `configs/online_default.toml`, `online_zerophase_default.toml` and
`offline_default.toml` holds the 10 wavelet bases. LocalCos128 remains the
ridge map's basis (`RidgeBasis`). `LocalCos64`, or `LocalCos128` for a window
of at least 128 L, can still be added to `Bases`. Both workers check `Bases`
against every scheduled window when they are built
(`wdf.config.Parameters.check_bases`).

### Online glitch label from a causal census

`GlitchLabel = "census"`, the new default, labels a trigger "glitch" when its
window meets a transient of the causal census of the whitened stream
(`wdf.processes.causal_census`). The census uses a robust sigma per octave
from `GlitchCensusLow` over the past `GlitchCensusMemory` seconds, the peak
must reach `GlitchTrigger`, and the extent comes from the broadband stream and
the octaves from `GlitchExtentLow`. The transients are written as
`GlitchCensusOnline`. `GlitchLabel = "bases"` keeps the rule by winning basis
(`GlitchLabelBases`). A label removes nothing.

### Online fit stretch chosen by a quality rule

With `ARFitQuality` (default on), the online model is fitted on a stretch of
the past whose whitened samples hold no census transient reaching
`GlitchTrigger`. For a refit, the stretch's octave power through the current
model must also stay within `ARFitLevelRatio` of white. A refit tries
`ARFitCandidates` stretches `ARFitStep` apart and keeps the current model when
none passes (`accepted` False, `RefitRejected`). The first fit waits for a
passing stretch, up to the same number of steps, then takes the stretch with
the smallest census peak.

### M4 coincidence protection stated in every offline run

`GlitchProtection` is "on" (with `GlitchProtectionDetectors`) when the other
detectors' census is given through `offline_network`. It is "OFF (single
detector ...)" when it cannot be, which is logged as a warning, and "off" when
`GlitchProtectCoincident` is false.

### Offline: the searched span, and parametersUsed at the end

The offline worker records `gpsEndSearched`, `gpsEndSegment`,
`UnsearchedTail_s`, and the parts of the tail in `ReadAheadParts_s`: two read
blocks, the band-pass settling, the whitening look-ahead, and the whitened
output's lag. Each search's `parametersUsed` is written again when the segment
ends, with the values set during the run (ridge drift and X-drop, low-pass
order, glitch and track counts).

### A third flow: the online worker with the offline whitening, at a fixed latency

`wdfUnitOnline` takes `WhiteningMode`, which defaults to "causal" and gives
the same outputs as before. "zerophase" whitens with the offline
`TruncatedWhitening` of `ZeroPhaseDuration`, built on the AR model fitted on
the past only. `BandPassPhase` defaults to "zero" in this mode (the offline
band-pass), and "causal" is also accepted. The flow reads ahead by a fixed
latency, `Latency_s = BandPassLookahead_s + WhiteningLookahead_s`: the zero-phase
band-pass's held blocks plus the filter's half-support. The latency is recorded
in `parametersUsed` and logged; it is 10 s at 2048 Hz with the shipped file. Every
output carries `data_time = emitted_at - Latency_s`. Altering the input after t
changes no output whose data time is at or before `t - Latency_s` (tested).
Glitches are labelled and kept, and the causal ridge runs on the whitened
stream. The configuration is `configs/online_zerophase_default.toml`, which is
`online_default.toml` except for the whitening, the band-pass phase and the
names.

### An offline worker: `wdfUnitOffline`, and `offline_network`

`wdf.processes.wdfUnitOffline` is `wdfUnitDSWorker` with, each switched by a
parameter: M4 glitch subtraction (`GlitchSubtraction`), with the census and the
fit in a pre-pass, the fit stretch cleaned and refitted
(`GlitchCleanFitStretch`), and the seeds coincident with another detector's
census (`GlitchCoincidenceSeeds`) protected; and the ridge over the segment
(`Ridge`), whose candidates, cells and z are written beside the triggers.
`offline_network` runs several detectors: their census first, so that each
protects the seeds coincident with the others, then each segment, then the
network ridge on the minimum of their z maps. With M4 and the ridge off it
gives `wdfUnitDSWorker`'s triggers from the same parameters (tested).
`wdfUnitDSWorker` is split into `_conditioning` and hooks (`_after_fit`,
`_prepare_segment`, `_read_block`, `_on_whitened`, `_finish_segment`) that do
nothing there; its triggers are unchanged.

### An online worker, causal end to end: `wdfUnitOnline`

`wdf.processes.wdfUnitOnline` searches one detector's stream without using
information from the future, block by block of `len` seconds: the causal
Butterworth band-pass and decimation; the prediction-error filter of an
autoregressive model fitted on the past only (`learn` seconds ending
`ARFitGap` before it starts, after `ARFitContext` of band-pass settling),
refitted every `ARRefitInterval` from the past and warmed up on the last
`ARWarmup` samples; the WDF search with every choice from the parameters; and
the causal ridge on the running wavegram of the window coefficients. There is
no glitch subtraction and no gating: triggers are labelled "glitch" when they
won one of `GlitchLabelBases`, and kept. Every trigger, ridge candidate,
running-best slot and refit carries its emission time and latency; altering
the input after a time changes nothing emitted before it (tested). It reads
frames (`FrameSource`) or arrays (`ArraySource`).

Configuration files are TOML, read by `wdf.config.Parameters.load_config`,
with `[detectors.<IFO>]` tables overriding the top level for one detector;
JSON files are still read. `BandPassDownSampling.process_array` runs the
causal form on NumPy blocks.

### Ridge tracking: a path along the cells, offline and causal

`wdf.analysis.ridge_tracking` finds tracks; `wdf.analysis.ridge` still only
describes the track of an event already found. Ported from the catalogue's
ridge_stage.py and ridge_v2.py:

- cell maps from the dense coefficients of the analysis windows (`RidgeBasis`,
  default `LocalCos128`; optionally the dechirped local cosine at
  `RidgeDechirpAlphas`, each cell its largest value over them), slots of
  `RidgeSlot` by states of `RidgeOctaveStep` octaves from `RidgeLowFrequency`,
  only each window's central stretch (`stream_cells`; `StreamingCells`, the
  same block by block, with a ring of the recent windows' coefficients);
- a per-row z from a rolling background, sigma-clipped (`RidgeBackground`,
  `RidgeBackgroundUpdate`, `RidgeBackgroundClip`): centred offline, strictly
  past online (`background_z`, `CausalBackground`); the network cell is the
  minimum of the detectors' z (`combine_min`);
- a local alignment scoring `z - b` per cell with a symmetric penalty
  `RidgeLambda * max(0, |d log2 f| - RidgeAllowance)` per move; b and the
  X-drop are calibrated on the background (`calibrate_drift`) unless given;
- `offline_ridge`: the episodes of the alignment over a segment, each
  extended greedily backward and forward; `StreamingRidge`: the same slot by
  slot from the past only, with a running best path ending at the current
  slot, candidates emitted when their episode ends with their latency, and a
  backward extension into the held past;
- `ridge_waveform`: the waveform under a path's cells, inverted from the
  window coefficients, and its integrated energy; `box_scores` and
  `standardise` for the per-duration standardisation of ridge v2.

### The local cosine in NumPy, and the dechirped local cosine

- `wdf.analysis.local_cosine`: `local_cosine` and `inverse_local_cosine`
  (Coifman-Meyer bell, DCT-IV, on the circle), laid out as p4TSA's
  `LocalCos<M>` and equal to its forward transform to rounding, over many
  windows at once; `local_cosine_geometry` gives each coefficient's tile.
- `wdf.analysis.dechirp`: the analytic signal times
  `exp(-i pi alpha (t - t_mid)^2)`, real part, then the local cosine
  (`dechirped_local_cosine`); its inverse, the re-chirp
  (`inverse_dechirped_local_cosine`, `rechirp`); the parallelogram tiles with
  their bounding rectangles (`dechirped_geometry`); and a symmetric sweep grid
  whose step is set by the tile, `1 / (2 T_seg^2)`, over a range given as a
  parameter (`alpha_grid`). The analytic signal is computed on the window
  padded with zeros, so nothing wraps from one end to the other
  (`analytic_signal`). **It is not an orthonormal basis:** the real part folds
  the bands the sweep moves across 0 Hz or the Nyquist frequency, so it cannot
  enter WDF2Classify's competition; it is in no default configuration.

### Glitches are subtracted with whitened sine-Gaussians (M4), with every choice a parameter

`wdf.processes.glitch_subtraction` (ported from feature/glitch-subtraction
a1bff75, census fix included) fits loud transients with sine-Gaussian atoms
defined in the conditioned stream and seen through the whitening's zero-phase
taps, by greedy matching pursuit with the exact two-parameter least squares of
each atom, and subtracts them before the whitening. Its census lives in
`wdf.processes.census` (the census of feature/gated-search, without gates).

- Every choice is a run parameter, read and recorded by
  `M4Settings.from_parameters`: `GlitchQs` (1, 2, 4, 8, 16),
  `GlitchFrequencyCount` (40), `GlitchTopFraction` (860/1024 of the Nyquist
  frequency), the bank's bottom `LowFrequencyCut`, `GlitchAtomCutTau` (5),
  `GlitchTrigger` (13.64), `GlitchFlag` (6), `GlitchMergeGap` (0.25 s),
  `GlitchCensusLow` (the search's `MinFrequency`, else `LowFrequencyCut`),
  `GlitchExtentLow` (4 times the census's lowest frequency), `GlitchSeedPad`
  (0.1 s), `GlitchStopSNR` (6), `GlitchMaxAtoms` (100), `GlitchOctaveOrder`
  (8). The bank's frequencies follow the rate (`bank_frequencies`).
- **Protection:** a seed found in two or more detectors within their light
  travel time plus `GlitchCoincidenceJitter` (0.01 s) may be a gravitational
  wave and is not subtracted (`protect_coincident`; `GlitchProtectCoincident`,
  default on). A network run takes each detector's census
  (`whitened_census`), marks the coincident seeds and fits the rest
  (`fit_seeds`).

### The Butterworth band-pass: a corner relative to the rate, an order from the alias requirement, a causal form

- `BandPassFilter = "butterworth"`: a high-pass of `HighPassOrder` (default 4)
  at `LowFrequencyCut` cascaded with a low-pass at `LowPassCorner` Hz, or at
  `LowPassCornerFraction` of the decimated Nyquist frequency (default
  0.87890625, 900 Hz at 2048 Hz). The low-pass order is no longer fixed: unset,
  `LowPassOrder` is the smallest order whose response, as the data receive it,
  is `AliasAttenuationDB` (default 50 dB) down at and above resampling -
  corner, the lowest input frequency that folds onto the corner; given, it is
  checked against that requirement and refused below it. Order 8 at 900 Hz
  meets 50 dB from 4096 Hz (53 dB) but reaches only 35 dB from 16384 Hz, where
  the requirement asks for order 12.
- `BandPassPhase` "zero" (the default) is the zero-phase band-pass as before;
  "causal" filters forward only with the state carried across blocks
  (`sosfilt`), reads nothing ahead and emits each block as it is read. Its
  response is `|H|`, not `|H|^2`, so the same requirement asks for a higher
  order; `response_db`, `alias_attenuation_db` and `passband` read the form in
  force.
- `LowFrequencyCut` has one default, `DEFAULT_LOW_FREQUENCY_CUT_HZ` (4 Hz),
  read through `wdf.config.Parameters.low_frequency_cut` by the band-pass and
  the worker alike.

### The search reads its rule, its block rule and its name from the parameters

- `ThresholdRule` ("block", the default, "dohonojohnston" or "cuoco") chooses
  the rule for a window's coefficients (`threshold_rule`); `wdf` and
  `wdfUnitDSWorker.segmentProcess` read it when no rule is passed, and record it.
- `BlockLength` (coefficients, 0 for round(ln window)) and `BlockLambda` set
  the block rule's L and lambda through p4TSA's `WDF2Classify.SetBlock`
  (feature/cosine-packets a6e8d3e); the values in force are recorded. Unset,
  the search is the core's own bit for bit.
- `SearchName` names a search in its trigger and `parametersUsed` file names
  (`wdf.analysis.io.trigger_file_name`, `parameters_file_name`,
  `run_parameters`): two searches with the same window, overlap and threshold
  no longer overwrite each other's files. A window that the schedule repeats
  gets its position appended. Without a name the files are named as before.
- The golden-output test pins the competition its fixtures were written with,
  since the core's default list changed (p4TSA feature/long-wavelets).

### A run chooses its bases: packet levels, local cosine, and a minimum frequency

- `Bases` (or `bases`) names the candidates of the per-window competition, a
  list or one comma-separated string, passed to p4TSA's `SetBases`; without
  it the core's own default list competes. The list that competed is written
  back, so `parametersUsed` names it. A name is a mother (pyramid), a mother
  with `P` and a depth (a uniform wavelet-packet level, e.g. `Coif1P6`), or
  `LocalCos` with a segment length M (the local cosine basis, e.g.
  `LocalCos128`, which needs p4TSA feature/cosine-packets).
- Every reading of a trigger's coefficients places them by the basis it won
  (`wave`): `packet_depth`, `mother_wavelet`, `coeff_levels`,
  `coeff_freq_bands` and `coeff_time_bounds` take it, and the maps, graphs,
  scales, meta-features and the reconstruction pass it. `LocalCos<M>` reads
  as packet depth log2 M (rows of fs / 2M, tiles of M samples) and inverts
  through `WaveletTransform(n, LocalCos, log2 M)`; `P<digits>` names are
  read as before. Packet and LocalCos tiles are drawn on the octave ladder of
  the maps and keep their own band in the tiles.
- `MinFrequency` (Hz, default 0) leaves the coefficients whose tile's upper
  edge is at or below it out of sigma, the block rule and EnWDF
  (`use_min_frequency`, p4TSA `WDF2Classify.SetMinFrequency`, d335d33 or
  later); 0 makes no call, and the search is the core's own bit for bit.
- `BlockRemainder` ("legacy", the default, "merge" or "scaled") sets how the
  block rule judges a block shorter than round(ln window) (`use_block_remainder`,
  p4TSA `WDF2Classify.SetBlockRemainder`, 5aeba62 or later). "merge" joins a
  run's remainder to the block before it, so a LocalCos128 row of 8 at a
  window of 1024 is one block of 8 rather than 7 + 1; "legacy" makes no call.

### The truncated whitening is its own class

`ZeroPhaseFilter = "truncated"` runs `TruncatedWhitening`: the model's
inverse spectrum at the resolution of `ZeroPhaseDuration`, Hann-truncated to
it, gwpy's whitening with the Burg model's ASD. It was `MagnitudeWhitening`
given a duration, though it is not the `|A|` filter; `MagnitudeWhitening` is
now that filter alone. `MagnitudeWhitening(..., duration=T)` still works,
returning the same `TruncatedWhitening` with a `DeprecationWarning`. Nothing
the worker computes changes.

### The noise model is fitted on settled samples

The stretch the autoregressive model is fitted on is read with
`ARFitContext` seconds of real data on each side (default
`DEFAULT_AR_FIT_CONTEXT_S`), band-passed whole, and the context is dropped
(`BandPassDownSampling.discard_edges`). Conditioned alone, the stretch began
and ended on the band-pass's transient from an assumed boundary; a Burg fit
estimates mean power, so the model absorbed that transient near the band edge
and the whitening left that band below white. The context must be at least the
band-pass's settling length and must be present in the frames; the worker
refuses a fit stretch the reader did not return whole. The triggers change.

### The whitening can have the response |A|, or be causal

The default is unchanged: the square root fitted and run forward and backward
(`ZeroPhaseFilter = "root"`), at order `SqrtWhiteningOrder`, by default the
model's order or 256, whichever is more. Given the same model it whitens bit
for bit as before; the triggers of a default run move only by the fit on
settled samples above.

`ZeroPhaseFilter = "magnitude"` selects `MagnitudeWhitening`: the filter whose
response is the modulus of the prediction-error polynomial, real and so exactly
zero phase, with the causal whitening's spectrum bin by bin, where the root
does not follow narrow lines. Its impulse response is measured down to
`ZeroPhaseResponseFloor` of its peak and applied by FFT convolution with real
past and future context around each block, which is linear convolution: the
stream does not depend on the block grid. The whitening from a measured
spectrum builds the same filter on the frequencies of its own estimate.

`ZeroPhaseFilter = "causal"` selects `CausalWhitening`: the fitted lattice
filter `A(z)` run forward only, latency zero, the model's phase, output scale
`sigma` as the magnitude filter's. It needs `WhiteningModel = "burg"`.

The warm-up `preWhite` is lengthened when shorter than the filter's past -- its
latency for the zero-phase filters, the model's order plus the band-pass's
settling for the causal one -- and the lookahead defaults to the longer of 20 s
and the latency; the filter, the warm-up and the latency (`ZeroPhaseLatency`)
are recorded in the run parameters.

## 1.3.0 --- 2026-09-18

### The compiled core comes from PyPI

`pip install "wdflow[pipeline]"` installs `py4tsa` along with the rest: p4TSA
publishes manylinux wheels, with GSL, FFTW3 and FrameL inside them, so nothing
has to be built or fetched by hand for trigger generation. `wdf.analysis` is
unchanged and still installs on its own.

The CI job that built p4TSA from source through conda is now a plain pip
install, and is named `Full suite`; a repository with the old name among its
required status checks has to be updated. The Docker image drops conda for
`python:3.12-slim`, and `docker/environment.yml` with it. The conda recipe
requires `py4tsa >= 3.2`.

### The triggers do not depend on the installed scipy

The zero-phase filtering lives in `wdf.filtering`, which reproduces scipy
1.17's `sosfiltfilt` bit for bit and so gives the same conditioned data under
any scipy, 1.18 included. The triggers and the golden references are
unchanged. The mock data generators use the same function.

### The compiled core is imported as `py4tsa`, not `pytsa`

Every `from pytsa.tsa import ...` in `wdf.processes`, `wdf.observers` and
`wdf.structures` is now `from py4tsa.tsa import ...`; code that imports the
core directly has to follow. `wdf.analysis` is unaffected. Requires p4TSA
3.1.0 or later.

### The block rule is the default for the coefficients of a window

The universal threshold judges each wavelet coefficient of a window alone
against `sqrt(2 ln N) sigma`, and a chirp crossing the window, spread over
neighbouring coefficients each at two or three sigma, leaves the trigger only
its loudest few. p4TSA's `WaveletThreshold.block` (Cai 1999) judges contiguous
coefficients of one level together, and is now the default of `wdf` and of
`wdfUnitDSWorker.segmentProcess`; on the merger window of GW150914 it keeps
the low-frequency part of the inspiral that the universal threshold dropped,
and fires on fewer windows of noise at the same EnWDF threshold. The rule a
search ran with is recorded beside its triggers as `waveletThreshold`. Pass
`wavThresh=WaveletThreshold.dohonojohnston` for the runs of the method paper.
Requires p4TSA with the block rule. The golden-output test pins each rule's
numbers on its own fixture: the universal threshold's are unchanged, and the
block rule's, `golden_triggers_block.parquet`, hold 1638 triggers where the
universal threshold emits 2843 on the same noise.

### The wavegram comparison runs on the device the machine has

`correlation_profiles` walks two gathered blocks of maps once per displacement,
and a block is the largest object the network stage holds, so the cost of the
comparison is proportional to the displacements searched. The two map stores
are now sent to the device once and the pairs are gathered against them there,
so nothing the size of a pair set crosses the bus and every displacement is
taken against one copy. `gpu` selects the path --- a GPU when there is one by
default, `False` to stay in numpy --- and `block` bounds the pairs gathered at
once, so the memory held is independent of how many pairs a slid background
produces. The two paths ask for the same arithmetic in double precision and
differ in the order the products are summed.

### The shape of a wavegram is compared on magnitudes

A coefficient carries the phase as well as the energy. Two detectors resolve one
transient onto different basis functions, at an arrival-time difference finer
than a bin, and two nearly antialigned detectors respond to one source with
opposite sign besides, so their coefficients disagree cell by cell where the
morphologies are identical. A cosine between signed maps therefore measures that
disagreement and not the shape, and taking its magnitude afterwards does not
undo it: the cancellation is inside the sum, not on the total. `shapes`, and so
`wavegram_similarity`, and the correlation `wavegram_match` reports, are formed
on `log(1 + |W|)`. The node features are formed the same way: the polarity of a
coefficient says where the source sits with respect to the two detectors, not
what the transient is, and a model that reads it learns the antenna pattern.

The sign is not discarded from the analysis. The coherent energy behind
`network_morphology` remains a signed sum whose magnitude is taken once at the
end, which is where a polarity that is physical belongs.

Every model fitted before this reads a different representation and has to be
refitted, and the lag the match reports moves with the change, so a sky region
derived from it is remeasured. The feature widths do not change, so a loaded
model stays loadable and says nothing about which representation it saw: a
caller checks the model against the builder rather than against the shapes.

### A background can be reduced as it is formed

`TimeSlideFAR.background_distribution` takes `reduce`. Given one, each slide's
candidates are handed to it and released, and the frame returned carries the
run's `attrs` alone: what a rate needs is an order statistic, and an order
statistic does not need the sample. `wdf.analysis.background.BackgroundAccumulator`
is one such reducer --- it keeps the largest values of every ranking exactly,
bins the rest, and says of every threshold whether it came from the exact tail
or from the histogram --- so the memory a background costs stops growing with
the number of displacements. Without `reduce` nothing changes.

## 1.2.0 --- 2026-09-04

### The wavegram match is measured on a grid that can resolve the delay

The bin two detectors' maps are compared on is derived by the network stage
from what the comparison has to represent, instead of being inherited from the
column width of whichever rendering was passed to it. Two bounds fix it and
both are physical: a bin coarser than a tile adds one detector's signed
coefficients to each other before either detector meets the other, so an
oscillating transient cancels against itself and the bands whose tiles are
shortest lose most, which is a frequency prior the search does not carry; a bin
coarser than the light travel time cannot represent the delay the coincidence
exists to measure, so the lag axis holds one usable point whatever tolerance
the edge was admitted on. The bin is the shorter of the shortest tile the band
ladder holds and the network's light travel time.

### The comparison of two wavegrams is a baseline, and is optional

`TriggerGraphBuilder` takes `match_wavegrams`. The comparison of a coincident
pair's two renderings is bounded by one, carries no loudness and maximises over
displacements, so it is a morphological baseline and enters no ranking; it
costs a correlation per candidate, and a study that does not read it need not
pay for it. With it off `network_wavegram_match`, `network_correlation` and
`coherent_statistic` are not measured, and `network_wavegram_matched` says so
rather than a zero standing in for a measurement.

The displacements searched are the timing tolerance's. Each rendering is laid
on its own event's instant and the whole bins of the difference between the two
instants are applied as a shift, at no cost whatever that difference is, so the
alignment at an arrival-time difference the light travel time allows is already
reachable. Searching as far as the transient lasts would only add agreements at
displacements the geometry forbids, at a cost growing as the square of the
transient's own length.

### One coherent amplitude, not two names for it

`network_block_morphology` was a literal copy of `network_morphology` and is
gone; `CoherentRanking` reads the one that remains. `network_morphology` is the
root of the magnitude of the signed coherent energy, so it is the geometric
mean of the two events' amplitudes on their noise scales reduced by the root of
the agreement between their tiles --- a quantity that vanishes when either
detector is silent, which the sum in quadrature of the two does not.

### The grid a pair is compared on comes from that pair

An event is rendered once, on a window centred on its own instant and no wider
than the event reaches. What two events are compared on is then the longer of
the two, widened by the displacements to be searched, and is a property of that
pair alone. Rendering every event on the widest event of the run made a
transient lasting minutes fix the grid of every millisecond-long one; what a
run holds is not known in advance, and at a bin of the shortest tile that array
does not exist. Pairs are handled in groups sharing a grid, by powers of two,
so a grid is built once per scale and no loop runs over pairs.

Because each pair is searched over its own transient, two pairs' profiles are
of one length only on the part they share. What `TriggerGraph` carries per edge
is therefore the profile on the timing tolerance's own lags, which every pair
can be read on and which a model can take as a feature of fixed size, together
with the agreement, the displacement it was found at and whether the pair was
compared at all --- taken where the comparison took them, over the pair's whole
search, and not reduced a second time from the part that was kept.

### The displacement the match found is reported beside it

The displacement found is reported beside the agreement, from the same pass, so
it is not a second estimate to be reconciled with it. It is read on the
`gpsPeak` clock the two renderings are anchored on, which is not the clock
`dt_s` is measured on, so the two are not to be added to each other.
`candidate_table` now also carries whether a pair was compared at all, so that a
match of zero is distinguishable from a match that was never formed, and a pair
never compared reports no displacement rather than the first point of the lag
axis.

### The match is asked only of a pair already coincident in time

A pair is admitted on the events' stretches of time, not on the difference of
their instants, and that stays as it is: a transient longer than one analysis
window is assembled as several events and the two detectors need not keep the
same one, so gating admission on the instants would take a candidate away from
detectors that did assemble it. The wavegram match, though, is a comparison of
two morphologies at a displacement, and it is now formed only where the two
events' own instants are already within the tolerance the geometry and their
timing spreads allow. No displacement the tolerance permits brings the instants
of a pair further apart than that together, so a search over those
displacements would report the agreement between the tail of one event and the
head of the other, and pay the trials factor of the search for it. Such a pair
keeps its edge and every statistic that needs no displacement; its match is
reported as no agreement.

### The lags searched are the ones the pair admits

Each event's map is laid on that event's own instant, so a map lag `L` places
the pair at the absolute displacement `offset + L`, with `offset` the
difference of the two anchors. `correlation_profiles` now searches the lags
that reach the displacements the tolerance admits rather than the lags within
plus or minus the tolerance: a pair whose two anchors already differ by the
tolerance reaches zero displacement only at `L = -offset`, and the agreement
reported for it was the largest among displacements the coincidence does not
admit.

The anchor difference is split for that: a whole number of bins, applied as a
shift of the map, and what the rounding leaves over. Pairs sharing a whole part
share their shifts, so the axis searched is one axis --- the displacements the
tolerance admits, widened by the half bin the rounding can leave --- whatever
the anchors are, and the work stops growing with how far apart anchors happen
to be. The function returns that axis together with each pair's leftover, and
the absolute displacement of a pair at a lag is the sum of the two.

The reduction inside that search is taken in numpy rather than through
`paired_dot`. Sending a matrix to a device is the right trade when one call
reduces a whole pair set against it, and the wrong one here: a shift changes
the slice, so no device copy survives to the next lag, and each reduction
touches a handful of pairs of a matrix of gigabytes. Measured on this stage's
shape, the transfer is three orders of magnitude more than the arithmetic it
carries.

`TriggerGraph.candidate_table` reports that displacement as the difference of
the two events' `gpsPeak` plus the map lag. It previously added the map lag to
`dt_s`, which is measured on whichever instant column the coincidence prefers,
so the two clocks differed by each detector's envelope-to-peak shift --- an
error that does not cancel in the difference and goes straight into the
arrival-time difference and the sky position.

### The normalisation is the event's, not the edge's

The norm of a map is a property of the event. Taking it once per edge gathered
one map per pair before reducing it, which is the allocation this stage cannot
afford at a background's pair count.

### Removed

`network_coherent_shape`, a sum of `network_morphology` and the logarithm of
the wavegram match. The first term is a coherent amplitude and the second is
dimensionless, so the sum has no reading; the floor the logarithm was guarded
with turned a match of exactly zero into an exclusion of 708 nats in a column
whose signal variation is of order one. A shape term belongs in
`wdf.analysis.network_statistic.CoherentRanking`, where it enters as a measured
log density ratio in the units the coherent energy is already in.

### The slide step is stated, and checked against what it has to clear

`TimeSlideFAR` checks a step against the coincidence's own timing-tolerance cap
and against the length the trigger stream is correlated over, and refuses one
below either: under the first a real coincidence stays admissible, under the
second two lags re-use the same clusters. The step itself is a stated constant
--- `FARConfig.min_shift_s`, four seconds --- as published burst searches state
theirs; None derives it from those two measurements instead, which makes the
step a property of the run rather than of the analysis.

The admission rule itself is unchanged: a pair is admitted on the events'
stretches of time, and the difference of their own instants ranks it.

### The instant a coincidence reads is named once

`wdf.analysis.robust_events.INSTANT_COLUMNS` names the order a coincidence
prefers its instants in, ending at the energy centroid. A centroid is a moment
of the energy that survived threshold in that detector, so two detectors seeing
one source at different projected amplitudes place it differently and that
difference is indistinguishable from geometry once it enters an arrival-time
difference; the centre of the tile carrying the largest coefficient is what both
detectors measure on the same transient, and it is what a pair is timed on.

`TimeSlideFAR` displaces every instant column together, so an event's instant
and its tiles stay on one clock through a slide.

### The two arrival-time differences are separated

A network edge carries the difference of the two events' own instants. That is
a difference of two node quantities: nothing is measured per pair, on the
unshifted population or on a background of accidentals.

`wdf.analysis.timing.arrival_time_difference` measures the finer one, on the
pair: the lag that maximises the cross-correlation of the two stitched
reconstructions, with the width the correlation peak declares, forming only the
lags the network's geometry allows. A placement it cannot reach --- two series
further apart than the maximum lag --- raises rather than returning a bound. It
costs a correlation per pair and no stage calls it: a study that wants a sky
region applies it to its own candidates.

### A displaced event carries its tiles

The coherent statistic is summed over the tiles two events share, and a tile
carries an absolute time. The node-side arrays are prepared once and reused for
every slide, so a slide that moved an event and not its tiles compared the pair
at the place it used to occupy: no slid pair shared a tile, and the statistic
was zero on the whole accidental population by construction rather than by
measurement. `tile_coherence_many` now takes a `displacement` per event, and
`TriggerGraphBuilder.build_from_prepared` measures it from the events
themselves --- the difference between the instant an event holds now and the
one it held when its tiles were laid out --- so it needs no knowledge of what
moved them. A displacement of zero is the arithmetic that was there before, so
the unshifted population is unchanged to the bit.

### An event carries the tiles of the block it was selected on

`wdf.analysis.detector_graph.EventWavegram.block_tiles` holds the tiles of the
member with the largest per-window energy, beside the tiles of every block the
event spans. A signed sum has mean zero, but its variance grows with the number
of tile pairs, so a coherent statistic taken over a whole event is ranked partly
on how long the transient lasted; summing over the selecting block alone would
bound that count by the analysis window whatever the transient lasted. That is
how the statistic that ranks is to be separated from the one that measures. It
is a candidate definition: the tiles are carried, and no column is formed from
them.

### The candidate table

`network_morphology`, the root of the magnitude of the signed coherent energy
over the tiles a pair shares, is the deterministic ranking beside the learned
columns. `WavegramCoincidenceFinder` given a scorer returns the learned
reading on every set of pairs it describes, so a background and the candidates
it calibrates are scored by one model over one population; on that path the
table carries `gnn_logit` in single precision and not the sigmoid `gnn_score`,
which is a function of it.

### Cost

An event is inverted once and the samples cross the C++ boundary in one
narrowing rather than one per sample. `wdf.analysis.pairs.paired_dot` takes a
`resident` cache, so a matrix re-used across calls is sent to the device once.
The learned scorer normalises its node features in place on the difference it
already owns, holding one copy of that matrix and not two.

## 1.1.1 --- 2026-09-02

Documentation only; the code is unchanged from 1.1.0.

The README, the design note and the published page state what the pipeline
does: an event's energy is the norm of its stitched reconstruction and what
ranks it for detection is its loudest block, the significance conditions on the
number of tiles and is read out of sample, and the scale a block is ranked on
is the median of its neighbours' while the thresholding stays with the block
itself.

## 1.1.0 --- 2026-09-02

### The statistic that selects is not the statistic that measures

An event's energy is the norm over its tiles; what ranks it for detection is
its loudest block. A hard threshold admits every tile at a floor of `2 ln N` in
normalised energy, so a sum over tiles accumulates that floor in the noise as
well as in the signal and a tile earns its place only when its excess exceeds
half the event's mean excess per tile. Ranking on the loudest block maps each
cluster to a maximum over quantities the search already computed, so its
background is a subset of the one an ungrouped search produces and the grouping
reduces the trials factor without being able to lose a candidate.

- `network_graph` ranks its nodes on `EnWDF_window` where the node table
  carries it.

### Nothing is calibrated on its own time

- `event_significance.out_of_sample_significance` scores each fold of a
  background from the others. A calibration read on the sample it was fitted on
  scores the j-th largest event of a bin as `log((N + 1) / (j + 1))` whatever
  the data, so a rate read off it is lower than a fresh background of the same
  livetime gives.
- `event_significance.significance_off_source` scores any event from the
  background of every time fold but its own, which is what a single recorded
  stretch allows: there the background is a part of the same data and a
  candidate lives beside it.
- `EventCalibration` conditions on `n_pixels`, the tiles the statistic sums,
  instead of on the blocks the analysis grid cut. The tile count sets the
  statistic's scale under the null; the block count is a property of the grid
  and made an event's significance depend on where the grid started.
- The exponential tail divides the total excess by `k - 1` rather than `k`, one
  of the `k` values being the anchor itself. A tail of equal values measures no
  slope instead of one at the smallest positive float.
- A size column carrying a missing value is refused rather than forming a bin
  of its own below every real size. An infinite statistic is scored as the
  loudest event there can be rather than dropped. A size larger than any the
  background produced is pooled into the last bin and says so.

### A coherent statistic is a product of amplitudes

- `event_tiles` returns each tile's signed amplitude beside its energy, and
  `tile_coherence` takes the product on the amplitudes. A product of magnitudes
  is positive whatever the data, so it grew with the number of tile pairs that
  met and two long events overlapping by accident scored above two short ones
  describing one transient. The signed product has mean zero under the null,
  and the pair is ranked on its magnitude because both polarities are physical.

### A block's noise is read where the transient is not

- `scale.local_noise_scale` takes the median of the neighbouring blocks'
  scales, and `scale.on_local_scale` re-expresses a block's statistic on it
  exactly, without recomputing a coefficient. A block's own scale is measured
  on the data it holds, signal included, so a transient loud enough to matter
  is divided by a scale it inflated itself. A block whose neighbours say
  nothing keeps the scale it measured itself.

### Also

- `detector_events` carries `n_pixels`, the distinct tiles an event owns.
- `efficiency_at_far` returns the realised rate and the number of accidentals
  the threshold stands on: a requested rate is not the rate a finite background
  can realise.
- `match_injections` takes the argmax over finite statistics alone, so an
  injection whose only candidate carries no statistic is unmatched rather than
  matched to it.

## 1.0.0 --- 2026-08-20

First tagged release.

### A time slide pays once for what does not change with it

The node-side matrices stay on the device between reductions. `paired_dot`
takes a mapping in which the caller keeps the device copies, and the graph
builder now hands it one belonging to the preparation: the shapes and the raw
maps are the same objects at every displacement, while sending one of them
costs far more than reducing it.

A coherent energy each event carries with itself was formed for every pair and
never read; it is gone. The event order a prepared graph is checked against is
built once per call rather than twice. The admitted pairs are taken as the
array the enumeration forms, through `IndexedCoincidenceFinder`'s
`candidate_edge_array`, instead of being named one tuple at a time and read
back. The windows a pair is compared on are laid end to end once by
`flatten_windows`, so placing them on a shared grid is one assignment rather
than a loop over events. The correlation at a lag is reduced over the whole
group and selected afterwards, which is what keeps a copy of a map per pair
from being made at every lag.

None of this changes a number the stage produces.

### An injected waveform begins and ends at zero

A series written into a frame that does not start and stop at zero leaves a
step there, and a step is wideband: whitening, which amplifies where the noise
is small, then gives the injection an edge it does not have, and a search can
rank it on that edge rather than on the source. Three places did it.

`band_limit` brings the ends of a band-limited pulse to zero. The band limit is
what a detector would have done to the pulse, and it does not leave it inside
the support it was generated on: a one-signed pulse becomes bipolar under a
high pass and the tail of the bipolar one is longer than the array. Measured
over the instrumental classes, the larger of a pulse's two end samples was a
sixth of its peak in the median for the Gaussian class and a twentieth for the
chirp like one; it is zero for all of them now, and what that removes is the
signal-to-noise the step was carrying --- two per cent in the median for the
Gaussian class.

The detector floor is drawn from a stream of its own for each injection.
Whether an injection's amplitude has to be raised to meet the floor, and how
many draws that takes, depends on its own waveform; taken from the shared
stream, a change to one waveform then moved the amplitude of every injection
drawn after it. Two sets drawn from one seed and differing in one class now
differ in that class alone.

### A supernova waveform is written into the band a detector responds in

`ccsn_polarisations` high-passes the catalogue's polarisations and brings their
ends to zero. A supernova does not return to zero after the bounce: the strain
settles on the memory, a permanent offset. Written into a frame that then
stops, that offset is a step as large as the memory itself, and a step is
wideband --- whitening, which amplifies where the noise is small, then makes
the end of the injection the loudest feature of the span, and a search ranks
the injection on the edge rather than on the source.

Measured on the catalogue, the last sample of a waveform was a third of its
peak in the median and nearly all of it at worst, and above a tenth in fifty of
the sixty-two files; it is zero in all of them now.

What that costs inside the band is measured on windowed spectra, since the
memory's own step leaks across an unwindowed one and makes the loss look far
larger than it is: the taper alone changes no band, and the high pass leaves a
tenth of the amplitude between twelve and sixteen hertz, a hundredth between
sixteen and twenty, and nothing above. A corner at five hertz would leave even
those; the one in use is stated rather than tuned, and what it removes is
stated with it. An injection scaled to a signal-to-noise ratio carries more of
it in band than the same injection did before, so a recovered fraction quoted
against that scale is not comparable with one measured on a set generated
before this.
