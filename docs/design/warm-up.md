# Warm-up: the seconds at the start of a segment that are not analysed

A WDF run discards the first stretch of every segment. This page says what that
stretch is for, why it cannot simply be kept, and what would be needed to
recover it.

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
steep transition close to Nyquist rings far longer than a gentle one of higher
order, and the stretch that has to be discarded follows the impulse response
rather than the parameter that produced it.

A filter that is asked to settle in less than it needs does not fail -- it emits
the unsettled transient as if it were data, at the start of every block, where
it looks like a short broadband burst and lands in the finest wavelet scales.
Measuring the settling is what prevents that: `padlen` is whatever the designed
filter turns out to need, so raising `FilterOrder` or steepening the band
lengthens the discarded stretch instead of corrupting the emitted one.

The settling stretch does not have to fit inside a single read. `Process`
buffers what it has read and emits a block only once `padlen` samples of real
future data have arrived, returning `None` meanwhile, however many reads that
takes. The cost is therefore latency rather than a constraint on the filter, and
it is stated in `latency_s` and carried by the timestamps.

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
