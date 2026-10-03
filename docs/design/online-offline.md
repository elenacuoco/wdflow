# Online and offline

wdflow has two workers. Each one runs on a single configuration file. Every analysis
choice in that file is a parameter, and every value used is written to the run's
`parametersUsed` file.

| | `wdfUnitOnline` | `wdfUnitOffline` |
|---|---|---|
| configuration | `configs/online_default.toml` | `configs/offline_default.toml` |
| band-pass | Butterworth, causal (`sosfilt`, state carried) | Butterworth, zero phase (`sosfiltfilt`) |
| low-pass order | the smallest that meets `AliasAttenuationDB` on `\|H\|` | the same rule on `\|H\|^2` |
| whitening | AR prediction-error filter, causal | `ZeroPhaseFilter`, "truncated" 16 s |
| noise model | fitted on the past only, refitted every `ARRefitInterval` | fitted once on a chosen stretch of the segment |
| glitches | labelled (`GlitchLabelBases`), never removed | M4 subtraction; seeds coincident in two or more detectors are protected |
| ridge | streaming, past only; running best path | whole segment; backward and forward extension |
| outputs | each with `emitted_at` and `latency` | triggers, M4 atoms and seeds, ridge cells and candidates |

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
