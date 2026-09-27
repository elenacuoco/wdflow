# Whitening

WDF whitens in the time domain, so the search runs on a stream. The filter has to
satisfy three things at once, and the third constrains the second.

1. **Flat spectrum, unit variance.** The detection statistic is
   `EnWDF = ‖c‖₂ / σ`, the norm of the surviving wavelet coefficients on the noise
   scale. Because the wavelet transform is orthonormal, that equals the
   matched-filter signal-to-noise ratio of the reconstructed transient — but only
   if the noise the coefficients are measured against is white.
2. **No phase distortion.** The same coefficients reconstruct the waveform and feed
   parameter estimation. A filter that moves the signal relative to the data makes
   the reconstruction describe something that did not happen.
3. **Streaming.** Sample-by-sample filtering, no spectrum re-estimation and no
   transform inside the loop.

## The filter

The noise is an autoregressive process: white noise of scale `σ` through the
all-pole filter `1/A(z)`, with

    A(z) = 1 - Σₖ aₖ z⁻ᵏ,     S(f) = σ² / |A(f)|²

**Running the lattice filter forward** gives `y = A(z)x`: the magnitude is right and
the phase is `arg A`, which varies with frequency, so a transient comes out smeared.

**Running any filter B forward and then backward** multiplies the spectrum by
`B·B* = |B|²` — real and non-negative, so the phase is identically zero. With
`B = A` this is the classic double whitening, and it divides by `S(f)` rather than
by `√S(f)`: where the front-end band-pass has emptied the spectrum, `|A|²` is
enormous and the output is dominated by a band the search does not analyse.

**The filter that satisfies both** has the response `|A(f)|` itself: real and
non-negative, so zero phase, and of the same modulus as the causal whitening, so
the whitened spectrum is the causal one bin by bin. Two constructions reach it.

### The magnitude filter (default)

`MagnitudeWhitening` applies `|A|` directly. Its impulse response

    h[n] = IFFT{ |A(f)| }[n]

is real and even. `|A|` is not a polynomial, so `h` is not finite: near a zero of
`A` close to the unit circle -- a narrow line -- `|A|` has a corner rather than a
smooth minimum, and the coefficients of a corner fall as the inverse square of
the lag. The support `K` is therefore measured on `h`, as the settling of the
band-pass is measured on its impulse response: the last lag at which `|h|` is
above `ZeroPhaseResponseFloor` of its peak. A truncation at a few times the
model's order, which a smooth spectrum would allow, leaves a narrow line far
above white. The taps kept, `-K … K`, are tapered at their ends and are exactly
even.

The filter is applied by FFT convolution over each output block together with
`K` samples of the real stream before it and `K` after it. That is linear
convolution with a fixed filter, so a stream whitened block by block is the
stream whitened at once, and no output sample depends on where a block began.
Its output has standard deviation `σ`, the causal whitening's.

With `WhiteningModel = "spectrum"` the response is `1/√S` of the measured
spectrum on the frequencies of its own estimate, which resolves `fs/nperseg`
and nothing finer; the filter is then `nperseg` taps long.

### The square root (`ZeroPhaseFilter = "root"`)

**Running any filter B forward and then backward** gives `|B|²` at zero phase,
so the filter that whitens by `|A|` both ways is the one with

    |B(f)|² = |A(f)|

the spectral square root of `A`, written `A₁ᐟ₂`. It is fitted by Levinson on the
autocorrelation of the pseudo-spectrum `1/|A(f)|`:

    r[m] = IFFT{ 1 / |A(f)| }[m]
    (A₁ᐟ₂, e, k) = Levinson(r, q)

because an AR model fitted to a spectrum `P` returns `P ≈ e/|A₁ᐟ₂|²`, so `P = 1/|A|`
gives `|A₁ᐟ₂|² ≈ e|A|`. Forward-backward with `A₁ᐟ₂` then yields

    y = e · |A| · x

flat, zero phase, with standard deviation `e·σ`. The fit is an approximation of
`|A|` at order `q`, and where `A` has deep narrow zeros it does not follow them:
the error is paid twice, since the response is the square of the fitted
magnitude.

## Latency

Zero phase and strict causality are incompatible -- a zero-phase filter has a
symmetric impulse response. What both constructions give instead is a **fixed
latency**, known before the filter runs:

- the magnitude filter reads exactly `K` future samples, its measured support;
- the square root is an FIR polynomial of order `q`, so its backward output at
  sample `i` is `z[i] = Σₖ₌₀..q aₖ · y₁[i+k]`, a finite sum over `q` future
  samples.

The same length is needed in the past, which is why the warm-up `preWhite` is
lengthened to the filter's latency when it is shorter, and why the lookahead
`WhiteningExtraSize` defaults to the longer of twenty seconds and the latency.
For the magnitude filter the latency is set by the narrowest line the model
holds, and the floor trades the residual left at that line against it.

## What runs where

| Step | When | Cost |
|---|---|---|
| Burg fit of `A` | once per segment | seconds |
| FFT of `A`, IFFT of `|A|`, support measured | once per segment | negligible beside the fit |
| FFT convolution of each block with its context (magnitude) | per block, streaming | one transform of block plus twice the support |
| Levinson for `A₁ᐟ₂` (root) | once per segment | negligible beside the fit |
| Lattice recursion, both directions (root) | per sample, streaming | — |

## Storage conventions

Two off-by-one conventions in the p4TSA containers, both easy to get wrong:

- `ArBurgEstimator`'s array holds **σ in `ar[0]`**, not a coefficient; the
  polynomial is `A(z) = 1 - Σₖ ar[k] z⁻ᵏ`.
- `LatticeView` stores `parcorF[j] = -k[j-1]`, with **slot 0 unused**.
  `ErrorForward`/`ErrorBackward` are metadata: the filter output does not depend on
  them.

## Using it

`wdf.processes.zero_phase_whitening.MagnitudeWhitening` is what `wdfUnitDSWorker`
uses by default; `ZeroPhaseResponseFloor` sets the floor its support is measured
at. `ZeroPhaseFilter = "root"` selects
`wdf.processes.zero_phase_whitening.ZeroPhaseWhitening` instead, with
`SqrtWhiteningOrder` setting `q`. The two share one interface. What ran, and its
latency (`ZeroPhaseLatency`), is recorded in the run parameters. See
`examples/zero_phase_whitening_example.py` for a standalone run of the square
root.

## Verifying a change

Any change to the conditioning should be checked against all five:

- `std / σ ≈ 1`
- kurtosis ≈ 3
- spectral flatness ≈ 1 across the analysis band
- zero lag between an injection and its reconstruction
- the whitened spectrum against the causal whitening's, bin by bin at the
  resolution of the narrowest line, not only its average over a band
