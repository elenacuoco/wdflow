# Online and offline

wdflow runs three flows. Each one runs on a single configuration file. Every
analysis choice in that file is a parameter, and every value used is written to
the run's `parametersUsed` file.

| | online, causal | online, zero phase | offline |
|---|---|---|---|
| worker | `wdfUnitOnline` | `wdfUnitOnline`, `WhiteningMode = "zerophase"` | `wdfUnitOffline` |
| configuration | `configs/online_default.toml` | `configs/online_zerophase_default.toml` | `configs/offline_default.toml` |
| band-pass | Butterworth, causal (`sosfilt`, state carried) | Butterworth, zero phase (`sosfiltfilt`), as offline; `BandPassPhase = "causal"` also allowed | Butterworth, zero phase (`sosfiltfilt`) |
| low-pass order | the smallest that meets `AliasAttenuationDB` on `\|H\|` | the same rule on `\|H\|^2` | the same rule on `\|H\|^2` |
| whitening | AR prediction-error filter, causal | `ZeroPhaseFilter`, "truncated" 16 s, as offline | `ZeroPhaseFilter`, "truncated" 16 s |
| noise model | fitted on the past only, refitted every `ARRefitInterval` | fitted on the past only, refitted every `ARRefitInterval` | fitted once on a chosen stretch of the segment |
| latency | none read ahead; an output is emitted up to `len` after its data | fixed, declared: `Latency_s`, 10 s at 2048 Hz (8 s whitening + 2 s band-pass) | the segment |
| glitches | labelled (`GlitchLabelBases`), never removed | labelled, never removed | M4 subtraction; seeds coincident in two or more detectors are protected |
| ridge | streaming, past only; running best path | streaming, on the zero-phase whitened stream | whole segment; backward and forward extension |
| outputs | each with `emitted_at` and `latency` | each with `emitted_at`, `latency` and `data_time` | triggers, M4 atoms and seeds, ridge cells and candidates |

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
checks that the two are the same filter. Each refit uses the `learn` seconds
that end `ARFitGap` before the switch. The new filter's state is then set
exactly by running it over the last `ARWarmup` samples, which must be at least
`ARorder`.

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
