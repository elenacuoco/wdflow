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

**The filter that satisfies both** is therefore the one with

    |B(f)|² = |A(f)|

the spectral square root of `A`, written `A₁ᐟ₂`. It is fitted by Levinson on the
autocorrelation of the pseudo-spectrum `1/|A(f)|`:

    r[m] = IFFT{ 1 / |A(f)| }[m]
    (A₁ᐟ₂, e, k) = Levinson(r, q)

because an AR model fitted to a spectrum `P` returns `P ≈ e/|A₁ᐟ₂|²`, so `P = 1/|A|`
gives `|A₁ᐟ₂|² ≈ e|A|`. Forward-backward with `A₁ᐟ₂` then yields

    y = e · |A| · x

flat, zero phase, with standard deviation `e·σ`.

## Latency

Zero phase and strict causality are incompatible — a zero-phase filter has a
symmetric impulse response. What the construction gives instead is a **fixed
latency**, and it is exact rather than approximate: `A₁ᐟ₂` is an FIR polynomial of
order `q`, so the backward output at sample `i` is

    z[i] = Σₖ₌₀..q aₖ · y₁[i+k]

a finite sum over exactly `q` future samples. Nothing beyond `q` changes the
answer, so the latency is `q / fs` seconds and is known before the filter runs.

The order can be small because `|A|` is far smoother than `|A|²`: an order-256
square root matches the whitening quality of the AR(3000) model it comes from.

## What runs where

| Step | When | Cost |
|---|---|---|
| Burg fit of `A` | once per segment | seconds |
| FFT/IFFT + Levinson for `A₁ᐟ₂` | once per segment | ~2 ms |
| Lattice recursion, both directions | per sample, streaming | — |

No transform ever runs inside the detection loop.

## Storage conventions

Two off-by-one conventions in the p4TSA containers, both easy to get wrong:

- `ArBurgEstimator`'s array holds **σ in `ar[0]`**, not a coefficient; the
  polynomial is `A(z) = 1 - Σₖ ar[k] z⁻ᵏ`.
- `LatticeView` stores `parcorF[j] = -k[j-1]`, with **slot 0 unused**.
  `ErrorForward`/`ErrorBackward` are metadata: the filter output does not depend on
  them.

## Using it

`wdf.processes.zero_phase_whitening.ZeroPhaseWhitening` wraps the whole thing and is
what `wdfUnitDSWorker` uses; `SqrtWhiteningOrder` sets `q`. See
`examples/zero_phase_whitening_example.py` for a standalone run.

## Verifying a change

Any change to the conditioning should be checked against all four:

- `std / σ ≈ 1`
- kurtosis ≈ 3
- spectral flatness ≈ 1 across the analysis band
- zero lag between an injection and its reconstruction
