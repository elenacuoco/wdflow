# Online and offline

wdflow runs three flows. Each one runs on a single configuration file. Every
analysis choice in that file is a parameter, and every value used is written to
the run's `parametersUsed` file.

| | online, causal | online, zero phase | offline |
|---|---|---|---|
| worker | `wdfUnitOnline` | `wdfUnitOnline`, `WhiteningMode = "zerophase"` | `wdfUnitOffline` |
| configuration | `configs/online_default.toml` | `configs/online_zerophase_default.toml` (track path: `configs/online_zerophase_track.toml`) | `configs/offline_default.toml` |
| band-pass | Butterworth, causal (`sosfilt`, state carried) | Butterworth, zero phase (`sosfiltfilt`), as offline; `BandPassPhase = "causal"` also allowed | Butterworth, zero phase (`sosfiltfilt`) |
| low-pass order | the smallest that meets `AliasAttenuationDB` on `\|H\|` | the same rule on `\|H\|^2` | the same rule on `\|H\|^2` |
| whitening | AR prediction-error filter, causal | `ZeroPhaseFilter`, "truncated" 16 s, as offline | `ZeroPhaseFilter`, "truncated" 16 s |
| noise model | fitted on the past only, on a stretch that passes the quality rule; refitted every `ARRefitInterval` | the same | fitted once on a chosen stretch of the segment |
| latency | none read ahead; an output is emitted up to `len` after its data | fixed, declared: `Latency_s`, 10 s at 2048 Hz (8 s whitening + 2 s band-pass) | the segment |
| glitches | labelled from a causal census (`GlitchLabel`), never removed | labelled, never removed | M4 subtraction; seeds coincident in two or more detectors are protected (`offline_network`) |
| ridge | streaming, past only; running best path | streaming, on the zero-phase whitened stream | whole segment; backward and forward extension |
| track path | tracks at `TrackLatency_s` | tracks at `TrackLatency_s` | tracks over the segment |
| outputs | each with `emitted_at` and `latency` | each with `emitted_at`, `latency` and `data_time` | triggers, M4 atoms and seeds, ridge cells and candidates, the searched span |

## The online worker never reads the future

Data are read in blocks of `len` seconds. Each output is emitted at the end of
the block that completes it, and its latency is the time from the end of the
data it describes to that emission:

- for a trigger, the end of its window;
- for a ridge candidate, the end of its last slot.

A test changes the input after a time `t` and checks that nothing emitted at or
before `t` changes. The test covers:

- triggers;
- ridge candidates;
- the running best path;
- the model refits.

The whitening is the AR model's prediction-error filter, `A(z) = 1 - sum a_k z^-k`.
It is the lattice of `CausalWhitening` written in direct form, and a test
checks that the two are the same filter. Each refit uses `learn` seconds of the
past chosen by the quality rule below. The new filter's state is then set
exactly by running it over the last `ARWarmup` samples, which must be at least
`ARorder`.

### The fit stretch is chosen by a quality rule

With `ARFitQuality` (on by default) a stretch of `learn` seconds is a valid fit
stretch when:

- its whitened samples hold no census transient whose peak reaches
  `GlitchTrigger`;
- for a refit, its power in each octave from `GlitchCensusLow`, whitened by the
  current model, is within `ARFitLevelRatio` of white (between
  `1 / ARFitLevelRatio` and `ARFitLevelRatio`).

A refit tries `ARFitCandidates` stretches, the first ending `ARFitGap` before
the switch and each next one `ARFitStep` seconds earlier, and takes the most
recent that passes. When none passes, the current model is kept; the attempt
is recorded in the refits with `accepted` False and counted in
`RefitRejected`. The first fit judges the stretch with its own model. When the
stretch fails, the search waits `ARFitStep` seconds and judges the stretch then
ending, up to `ARFitCandidates` stretches, and then takes the one with the
smallest census peak. Every refit records the candidate it took, its census
peak and its octave levels. A test checks that a stretch holding a glitch or a
louder noise level is skipped, and that the rule reads only the past.

### One model history for several runs

Every online run writes the noise models it applied to
`ARModels-<itf>-Search<name>.h5`, next to its outputs, and records the path as
`ARModelsFile`. The file holds one row per model, in switch order:

- `switch_gps`: the GPS time from which the model whitens. In the causal mode
  this is the read time; in the "zerophase" mode, the end of the conditioned
  stream;
- `fit_start`, `fit_end`: its fit stretch;
- `sigma`: its whitening's scale;
- `ar`: its scale and prediction coefficients, `(models, ARorder + 1)`.

The attributes are `ARorder`, `WhiteningMode`, `resampling`,
`ZeroPhaseDuration` (zero-phase mode) and `itf`.

With `ARModelsFrom` set to such a file, a run fits nothing and takes no
fit-quality census. It applies that run's first model when its switch time is
reached, and each later model at its own switch time. A file written with
another `ARorder`, `WhiteningMode`, rate or `ZeroPhaseDuration` is refused.
This is how an injection run whitens exactly as its background run does: the
dense injections would otherwise fail the fit-quality rule at every refit.
Each model was fitted on data before its switch time, so the run stays
causal, and its latency is unchanged. A test checks that the injection run's
whitened stream minus the background's equals the injections whitened by the
same models, sample for sample. The offline worker fits one model per segment
and does not refit, so it has no such parameter.

### Glitches are labelled from a causal census

With `GlitchLabel = "census"` (the default) the online worker keeps a census of
its whitened stream, block by block, from the past only
(`wdf.processes.causal_census.CausalCensus`):

- the broadband stream and its octaves from `GlitchCensusLow`, split by causal
  Butterworth filters of order `GlitchOctaveOrder`;
- each divided by its robust standard deviation over the `GlitchCensusMemory`
  seconds before the block, from `GlitchCensusMinimum` seconds of past on;
- samples above `GlitchFlag` closer than `GlitchMergeGap` form a transient,
  kept when its peak reaches `GlitchTrigger`;
- its extent comes from the broadband stream and the octaves from
  `GlitchExtentLow` (four times `GlitchCensusLow` by default), widened by
  `GlitchSeedPad`.

A trigger whose window meets a transient known when it is emitted is labelled
"glitch", any other "candidate". The census reads the whitened stream up to
the block the trigger is emitted with, so the label adds no latency. The
transients are written as `GlitchCensusOnline`. `GlitchLabel = "bases"` labels
"glitch" a trigger whose winning basis is one of `GlitchLabelBases`. A label
never removes a trigger.

The ridge cells come from the dense coefficients of the analysis windows
(`RidgeBasis`). The coefficients of recent windows are kept in a ring buffer.
Each slot's z comes from a background strictly in the past. The local alignment
works as follows:

- it uses the cells up to the current slot only;
- it keeps the best path that ends at the current slot;
- it emits a candidate when the candidate's episode drops `RidgeXDrop` below
  its maximum;
- the backward extension reaches into the held past only.

## The zero-phase online flow reads ahead by a fixed latency

`WhiteningMode = "zerophase"` runs the online worker with the offline
whitening. The noise model is fitted on the past only, as in the causal flow.
The flow is then not causal: it reads ahead by a fixed latency, declared before
the run starts. The latency is written to `parametersUsed` as `Latency_s` and
logged:

```
Latency_s = BandPassLookahead_s + WhiteningLookahead_s
```

- `WhiteningLookahead_s` is the half-support of `TruncatedWhitening`,
  `round(ZeroPhaseDuration * resampling / 2) / resampling`: 8 s for 16 s, and
  16 s for V1's 32 s.
- `BandPassLookahead_s` is zero with `BandPassPhase = "causal"`. With "zero",
  the offline band-pass holds each read block until its settling `padlen` has
  been read, which takes `ceil(padlen / (len * sampling)) * len`. From 4096 Hz
  `padlen` is 5142 samples, so with `len = 1` s this is 2 s.

For the shipped configuration at 2048 Hz, the latency is 10 s (18 s for V1).
Only `TruncatedWhitening` is accepted, because its support is set by the
duration. A filter whose support is measured on the model, such as "magnitude",
has no fixed latency and is refused.

At every read, the whitened stream is complete up to `now - Latency_s`. Each
output carries:

- `data_time`, which is `emitted_at - Latency_s`;
- `emitted_at`.

A trigger's `latency` lies between `Latency_s` and `Latency_s + len` after the
end of its window.

A model fitted when the conditioned stream reaches S uses the `learn` seconds
that end `ARFitGap` before S. It whitens the samples from S on, and the search's
sigma follows it from S.

A test changes the input after a time `t`. It then checks that no output whose
`data_time` is at or before `t - Latency_s` changes, and that no trigger whose
window ends at or before `t - Latency_s` changes either. A second test checks
that the filter is the offline worker's streamed `TruncatedWhitening`, sample
for sample.

## The offline worker is the segment worker plus two stages

With `GlitchSubtraction` and `Ridge` off, `wdfUnitOffline` is
`wdfUnitDSWorker`, and a test checks that it gives the same triggers. When M4
is on:

1. A pre-pass takes the census of the segment's whitened stream.
2. Seeds that coincide with another detector's census are protected.
3. The remaining seeds are fitted.
4. The fitted atoms are subtracted before the whitening.

`offline_network` runs the full network in this order:

1. every detector's census;
2. every detector's search;
3. the network ridge, on the minimum of the detectors' z maps.

Coincidence protection needs the other detectors' census, so it is effective
only when the detectors run together through `offline_network`. A run records
`GlitchProtection`: "on", with the detectors in `GlitchProtectionDetectors`,
when another detector's census was given; "OFF (single detector: ...)" when
`GlitchProtectCoincident` is set but no other census was given, which is also
logged as a warning; "off" when `GlitchProtectCoincident` is false.

### What a segment searched

The worker reads ahead of what it searches. It records the end of the last
whitened block searched as `gpsEndSearched`, the segment's end as
`gpsEndSegment`, and the difference as `UnsearchedTail_s`. The parts of the
tail are in `ReadAheadParts_s`:

- `blocks`: two read blocks of `len`, because reading stops before a block
  would pass the frames' end, which the reader would pad with zeros;
- `bandpass`: the band-pass's settling `padlen`;
- `whitening`: the whitening's look-ahead `WhiteningExtraSize`;
- `whitening_lag`: one block plus `WhiteningExtraSize`, by which the whitened
  output trails the conditioned input.

That is about `3 len + 2 WhiteningExtraSize + padlen`. `len` only batches the
reads and changes no trigger, so a shorter `len` shortens the tail.
`parametersUsed` is written when each search starts and again when the segment
ends. The second write adds the values set during the run: the ridge's
calibrated drift and X-drop, the low-pass order, the searched span, and the
glitch and track counts.

## The four stages and the two paths

WDF runs in four stages:

1. **Triggers.** Each detector's single-window triggers (`EnWDF`, the
   competition between the bases of `Bases`), from one of the three workers.
2. **Single-detector reconstruction** (`Reconstruction`).
3. **The graph between detectors** (`Coincidence`).
4. **The ranking of its candidate pairs** (`Scorer`): "classic", the column
   `NetworkRankColumn`, or "gnn", a `GNNCoincidenceScorer` loaded from
   `GNNModel`.

Stage 1 is the same on both paths. Stages 2 to 4 run through
`wdf.analysis.network_driver.run_network`, online through `NetworkStream`.
Two paths are available, and they do not mix: `Reconstruction` and
`Coincidence` must both be "wavegram" or both be "track"
(`wdf.config.Parameters.network_path`).

| | "wavegram" (default) | "track" |
|---|---|---|
| stage 2 | `build_detector_graph`, `detector_events`, `event_coefficients`; the comparison map at `NetworkComparisonBin` times the longest light travel time | each trigger seeds a track through the full ridge map (`wdf.analysis.track_reconstruction`), and the waveform along it |
| what stage 2 reads | the triggers with their coefficients | the worker's `TrackEvents` and `TrackWaveforms` |
| stage 3 | `WavegramCoincidenceFinder` over `IndexedCoincidenceFinder`'s admitted pairs | `TrackCoincidenceFinder`: of the pairs `IndexedCoincidenceFinder` forms, those whose tracks are compatible and coherent at a physical lag |
| edge features | `EDGE_FEATURES` | the classical columns and `TRACK_EDGE_FEATURES`: track compatibility, and the lag, coherence and phase of the complex cross-correlation of the waveforms |
| ranking (classic) | `network_morphology` (`network_enwdf` selectable) | `track_statistic` |
| `dt_s` | the difference of the events' instants | the coherence lag `track_lag_s` |
| GNN | `cross_edge_dim = N_EDGE_FEATURES` | `cross_edge_dim = N_EDGE_FEATURES + len(TRACK_EDGE_FEATURES)`; a model must be fitted with it |
| latency online | the stream's | the stream's plus `TrackHorizon` and the map's completion: `TrackLatency_s` |

With the defaults the paper path is run, and its triggers, detector events and
network table are the same as before the track path existed (tested against a
snapshot). With `NetworkSlides` above zero, `run_network` also draws the
time-slide background (`TimeSlideFAR`, step `NetworkSlideStep`) and gives every
candidate its false-alarm rate.

### Optional screens on the paper path

`NetworkLagCut` (default "none") measures each pair's lag on the two events'
stitched reconstructions (`event_waveform`, `arrival_time_difference`,
searched over `NetworkLagSearch`, 0.25 s), not on the loudest tiles'
instants. It reports `network_lag_s` (first detector minus the second) and
`network_lag_width_s`. The pair's window is its light travel time plus
`NetworkLagTolerance` (0.01 s, about the spread of that lag; set it from the
spread measured on a run's injections).

- "cut" keeps a pair only when `|network_lag_s|` is within the window. A pair
  whose lag cannot be measured within the search is dropped.
- "penalty" keeps every pair and multiplies the ranking column by
  `network_lag_factor = 1 / (1 + (max(0, |lag| - window) / NetworkLagScale)^2)`
  into `network_rank_statistic`, which the pairs are then ranked on. The
  factor is 1 within the window and falls with the excess, the same for
  either sign. It is 0 when the lag cannot be measured.

A single detector's census labels a loud merger as a glitch, because one
detector cannot tell the two apart. The network can. Given the detectors'
census (`run_network(..., census=)`: the online `GlitchCensusOnline` tables
or the offline census; online, `NetworkStream.add(..., census=)`), each pair
carries `glitch_i` and `glitch_j`. An event that overlaps a census transient
of its detector is a glitch only when the other detector has no transient
within the light travel time plus `NetworkGlitchTolerance` (0.01 s), so
coincident transients are not glitches. `glitch_single_ifo` is true when
either event is a glitch. `NetworkGlitchVeto` sets its use:

- "none" (the default) only reports it;
- "veto" drops the pair;
- "penalty" multiplies the ranking column by `NetworkGlitchFactor` (0.1)
  into `network_rank_statistic`.

Online, the decision needs the other detector's census at the same latency as
its triggers: run the network stage once both detectors' outputs reach the
same data time.

The screens run inside the finder (`wdf.analysis.pair_screen.ScreenedFinder`),
so the time slides measure each slid pair as the zero-lag pairs, moving each
detector's reconstruction and census with its events.

### The track path

With `Reconstruction = "track"` the worker (it requires `Ridge = true`)
registers a `TrackObserver` on each search, which keeps each trigger with its
coefficients as a seed:

- the seed's slice of the map runs from `TrackLookback` seconds before the
  trigger's window to `TrackHorizon` seconds after its end;
- its cell is that of the trigger's largest coefficient;
- `offline_ridge` runs on the slice, and its best candidate that passes the
  seed's cell within `TrackSeedStates` states is the track; where none does,
  the track is the anchored alignment through the cell, with the same scores
  and X-drop, followed backward and forward from it;
- seeds that overlap in time and band form one group, and tracks that share a
  cell are merged.

The track has its own penalty (`TrackRidgeSettings`). A move between two
states costs `TrackLambda * max(0, |d log2 f| - c)` per slot. The allowance
`c` is the largest sweep an inspiral makes in one slot from the lower of the
two states' frequencies, `df/dt = (96/5) pi^(8/3) (G Mc / c^3)^(5/3) f^(11/3)`
at the largest chirp mass of `TrackMcRange`, and at most `TrackAllowanceMax`.
The allowance depends on the pair of states and not on the direction of the
move, so a falling track costs what a rising one costs. Low frequencies, where
an inspiral sweeps slowly, get a small allowance, and a move beyond it costs
`TrackLambda` per octave. The track's drift and X-drop (`TrackDrift`,
`TrackXDrop`) are calibrated on the background with this penalty and with the
track's own alignment (`calibrate_track_drift`). The drift is the smallest for
which the `TrackCalibrationQuantile` percentile of the slots the alignment
keeps into noise is at most `RidgeCalibrationKept`, so a track ends where the
signal ends. The values used are recorded as `TrackDriftUsed` and
`TrackXDropUsed`.

The waveform is the inverse transform of the coefficients under the track's
cells, widened by `RidgeWaveformStates`, in `TrackBasis`:

- `LocalCos64` or `LocalCos128`: the map's own coefficients when it is the
  map's basis (`RidgeBasis`), recomputed on the whitened stream otherwise.
  LocalCos64 tiles are 31.25 ms by 16 Hz at 2048 Hz, LocalCos128 tiles
  62.5 ms by 8 Hz;
- `CosinePackets<m>`: the cosine-packet best basis (`py4tsa.tsa.CosinePackets`,
  segments from `m` samples, cost `TrackPacketCost`) of the whitened stream over
  the track's span.

Online, a track is built once the map is complete `TrackHorizon` past its
seeds' last window. Its declared latency is

```
TrackLatency_s = Latency_s + TrackHorizon + RidgeSlot + (RidgeWindow - RidgeOverlap / 2) / rate
```

which is 11.47 s at 2048 Hz with `configs/online_zerophase_track.toml` (19.47 s
for V1). A test changes the input after a time `t` and checks that no track
whose seeds end at or before `t - TrackLatency_s` changes. The worker refuses a
slice longer than the map's ring (`RidgeMaxSlots` slots). Offline, the tracks
are built after the segment's ridge, on its map and its whitened stream. Both
write `TrackEvents` (CSV) and `TrackWaveforms` (npz), read back with
`load_tracks`.

`TrackCoincidenceFinder` describes each pair the classical rule forms with:

- `track_time_overlap`, `track_f_distance`, `track_f_overlap`,
  `track_common_s` and `track_compatible`: the two tracks' spans and f(t)
  compared within the light travel time plus `TrackTimingJitter`, with
  tolerance `TrackFrequencyTolerance` octaves. `track_common_s` is the time
  over which they agree within it, `track_common_fraction` that time over the
  shorter track and `track_common_cycles` the cycles in it;
- `track_lag_s`, `track_coherence` and `track_phase`: the lag of the largest
  modulus `|Gamma|` of the complex cross-correlation of the two waveforms'
  analytic signals, searched over `TrackLagSearch`, that modulus (0 to 1) and
  its phase.

It admits a pair when:

1. the tracks are compatible: the time their f(t) agree, `track_common_s`, is
   at least `TrackMinCommonFraction` (0.5) of the shorter track, holds at least
   `TrackMinCommonCycles` (2) cycles at the tracks' frequency, and is at least
   `TrackMinCommonDuration` (an optional floor, 0 by default). The rule is
   relative because a track is as long as its signal: a merger lasts a few
   tenths of a second, an inspiral seconds. Coherence over fewer than about two
   cycles says nothing about the phase;
2. the lag is physical: `|track_lag_s|` is at most the light travel time plus
   `TrackTimingJitter`.

The coherence is not cut on by default (`TrackCoherenceMin = "none"`). It
enters the ranking statistic, and the false-alarm rate of the time slides
decides. Two options set a third condition, `track_coherence >=
TrackCoherenceMin`:

- a number;
- "calibrate": the `TrackCoherencePercentile` percentile of `|Gamma|` over the
  run's time-slid pairs that pass 1 and 2 (`calibrate_coherence_min`, with
  `TrackCalibrationSlides` slides of `NetworkSlideStep`). When fewer than
  `TrackCalibrationMinimum` slid pairs pass, every slid pair the classical
  rule forms is used.

The value applied is recorded as `TrackCoherenceMinUsed` (0 for none), and a
calibration's population as `TrackCoherenceCalibration`.

The pair's statistic, and the track path's default `NetworkRankColumn`, is

```
track_statistic = min(rho_1, rho_2) * |Gamma|,    rho = sqrt(track_energy)
```

`rho` is the SNR integrated along each detector's track, the energy of its
reconstructed waveform in noise units. The minimum asks both detectors to carry
energy along their tracks, so a loud glitch in one detector paired with a weak
track in the other scores as the weak one. `|Gamma|` is the fraction of that
energy the two waveforms share at a physical lag. `network_enwdf`, the seed
triggers' statistic, stays a column. On the track path `dt_s` is the coherence
lag `track_lag_s`, and the difference of the events' instants is
`dt_instant_s`. Every lag is the first detector of `ifos_involved` minus the
second, `t_H1 - t_L1` for "H1,L1". This holds for `dt_s` on both paths,
`network_wavegram_match_dt` and `arrival_time_difference`, and the lag is
positive when L1 leads.

A time slide moves an event's times. The finder reads each event's
displacement from its `gpsPeak` and applies it to the event's track and
waveform, so `TimeSlideFAR` runs on it.

### How to choose

- Run the paper path to reproduce the published method; it is the default.
- Run the track path to describe each event by its track and its waveform
  across windows, and to compare detectors by the coherence of their
  waveforms. Online, it costs `TrackHorizon` plus the map's completion in
  latency.
- `TrackBasis`: `LocalCos64` follows a fast sweep in time, `LocalCos128`
  resolves frequency more finely, and `CosinePackets<m>` adapts the segment
  length to each track's span.
- The GNN of the paper path cannot read the track path's edges. A model for
  the track path must be fitted with its `cross_edge_dim`.

### Parameters of stages 2 to 4

| parameter | default | meaning |
|---|---|---|
| `Reconstruction` | "wavegram" | stage 2: "wavegram" or "track" |
| `Coincidence` | "wavegram" | stage 3: the same as `Reconstruction` |
| `Scorer` | "classic" | stage 4: "classic" or "gnn" |
| `GNNModel` | none | saved `GNNCoincidenceScorer`, required by "gnn" |
| `NetworkRankColumn` | "network_morphology"; "track_statistic" on the track path | the column the classic scorer ranks on; "network_enwdf" selectable |
| `NetworkComparisonBin` | 2.0 | comparison map bin over the longest light travel time; 0 for none |
| `NetworkSlides` | 0 | time slides of the background |
| `NetworkSlideStep` | 4.0 s | step between two slides |
| `NetworkMemory` | 64 s | what `NetworkStream` holds |
| `NetworkLagCut` | "none" | wavegram path: "none", "cut" or "penalty" on the reconstructions' lag |
| `NetworkLagTolerance` | 0.01 s | added to the light travel time |
| `NetworkLagScale` | 0.01 s | scale of the lag penalty |
| `NetworkLagSearch` | 0.25 s | half-width of the lag search |
| `NetworkGlitchVeto` | "none" | wavegram path: "none", "veto" or "penalty" on `glitch_single_ifo` |
| `NetworkGlitchTolerance` | 0.01 s | added to the light travel time for coincident transients |
| `NetworkGlitchFactor` | 0.1 | rank factor of a single-detector glitch |
| `TrackLookback` | 4.0 s | map searched before a seed's window |
| `TrackHorizon` | 1.0 s | map searched after it |
| `TrackBasis` | `RidgeBasis` | `LocalCos<M>` or `CosinePackets<m>` |
| `TrackSeedStates` | 1 | states either side of the seed's cell |
| `TrackPacketCost` | "entropy" | cosine-packet cost: l1, entropy, wdfUniversal, wdfBlock |
| `TrackTimingJitter` | 0.0035 s | added to the light travel time |
| `TrackFrequencyTolerance` | 0.5 octave | f(t) tolerance between two tracks |
| `TrackMinCommonFraction` | 0.5 | admission 1: least fraction of the shorter track the two f(t) agree over |
| `TrackMinCommonCycles` | 2 | admission 1: least cycles over that time |
| `TrackMinCommonDuration` | 0 s | admission 1: optional absolute floor on that time |
| `TrackLagSearch` | 0.1 s | half-width of the coherence's lag search |
| `TrackCoherenceMin` | "none" | optional coherence cut: "none", a number, or "calibrate" |
| `TrackCoherencePercentile` | 99 | noise percentile of "calibrate" |
| `TrackCalibrationSlides` | 100 | slides of the calibration |
| `TrackCalibrationMinimum` | 20 | slid pairs passing 1 and 2 needed |
| `TrackLambda` | 3.7 per octave | penalty beyond the allowance |
| `TrackMcRange` | [1, 30] solar masses | chirp masses whose largest sweep is the allowance |
| `TrackAllowanceMax` | 0.27 octave | largest allowance per slot |
| `TrackDrift`, `TrackXDrop` | "calibrate" | the track's drift and X-drop |
| `TrackCalibrationQuantile` | 90 | percentile of the slots kept into noise held at `RidgeCalibrationKept` |
| `TrackLightTravelTime` | from the detectors | overrides the light travel time |
| `TrackGraphBasis` | "LocalCos64" | basis of the node maps the GNN reads on the track path |

## Units

Some parameters are in samples of the decimated stream:

- `window`, `overlap`;
- `ARorder`, `ARWarmup`;
- `RidgeWindow`, `RidgeOverlap`;
- the local cosine segment M of `LocalCos<M>`;
- `nperseg`.

Durations are in seconds and frequencies in Hz. Where a default depends on
the rate, it is stated relative to the rate:

- the low-pass corner is 0.879 of the output Nyquist frequency;
- the top of the M4 bank is 0.84 of the output Nyquist frequency;
- the ridge's top state is at the Nyquist frequency.

## Not in either default

The dechirped local cosine (`wdf.analysis.dechirp`) is not an orthonormal basis.
It folds the bands that its sweep moves across 0 Hz or the Nyquist frequency.
Because of this it cannot enter WDF2Classify's competition. The ridge can use
it through `RidgeDechirpAlphas`, which is empty by default.
