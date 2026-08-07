# Warm-up: the seconds at the start of a segment that are not analysed

A WDF run discards roughly the first four to six seconds of every segment. This
page says what those seconds are for, why they cannot simply be kept, and what
would be needed to recover them.

## Why anything has to be discarded

Two stages of the chain carry memory, and neither gives a correct answer until
it has seen enough data.

**The conditioning band-pass.** `BandPassDownSampling` filters the stream
forward and then backward, which is what makes the pass zero phase: a filter
with frequency-dependent phase displaces a transient in time, and every
parameter the search reports -- peak time, duration, the reconstructed waveform
itself -- is read off that transient. Running the filter backward means it has
to start somewhere, and starting it at an arbitrary point injects a step. The
chain therefore runs the backward pass through `padlen` samples of *real future
data* first, discards that stretch, and keeps only what follows, by which point
the filter state has settled.

**The whitening.** The autoregressive model is fitted on a separate learning
stretch, but the lattice filter that applies it also carries state, and
`DoubleWhitening` additionally needs a buffer of future data (`ExtraSize`)
before its own backward pass can produce a good estimate.

## Why the discarded stretch is not a fixed number

`padlen` is measured, not chosen. `settling_length` drives an impulse through
the designed filter and finds where the response has decayed below a fraction of
its peak. This matters because a filter's ringing is not read off its order: a
Chebyshev type II of order 12 with a steep transition near Nyquist settles in
about 1.2 s, while a Butterworth of order 5 settles in about 0.34 s.

That difference is what makes the constraint real. The settling stretch has to
fit inside a single read, because it is taken from the block that follows the
one being emitted, and the warm-up reads the stream one second at a time. A
filter that is asked to settle in less than it needs does not fail -- it emits
the unsettled transient as if it were data, at the start of every block, where
it looks like a short broadband burst and lands in the finest wavelet scales.
This is what a fixed `padlen` of one second, inherited from a Butterworth that
settled in a third of that, did to a steeper filter that did not.

The default order is therefore chosen so that the filter settles inside one
read: at order 10 it settles in 0.85 s, with 80 dB of rejection at the frequency
that folds.

If `FilterOrder` is raised past what the read length can supply, `Process`
raises rather than producing quietly wrong data.

## What is discarded, in order

1. `preWhite` warm-up reads, so that both the band-pass and the whitening
   lattice have settled.
2. `WhiteningExtraSize` samples buffered ahead of the detection loop, so the
   whitening's backward pass has its lookahead before the first output block.

With the defaults this comes to a few seconds at the head of each segment.

## Could they be analysed?

Offline, yes, and nothing about the data itself is bad -- it is discarded
because the filters have not settled *going forward*, not because the strain is
unusable. Reading the segment backward, or filtering it as one array with
`sosfiltfilt`, would condition those seconds correctly; this is what the
`estimation=True` path already does for the learning stretch, which is complete
in itself and needs no warm-up.

Recovering them is not currently done, for two reasons worth stating plainly.
The seconds recovered are a negligible fraction of any real observing segment,
and a stretch conditioned by a different path is not guaranteed to have the same
noise properties as the rest, so triggers from it would not be directly
comparable to the rest of the run. A search whose background estimate has to
hold across the whole segment is better served by a uniform treatment than by a
few extra seconds.

In a low-latency setting the same seconds are a real startup cost and cannot be
recovered at all, since the future data they need has not arrived yet.
