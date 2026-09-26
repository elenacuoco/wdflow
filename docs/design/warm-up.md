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
itself -- is read off that transient. Each pass has to start somewhere, and
starting it at an arbitrary point injects a step. Within the stream the chain
runs the forward pass through `padlen` samples of *real past data* and the
backward pass through `padlen` samples of *real future data*, and keeps only
what lies between, by which point the filter state has settled in both
directions. At the first sample of a segment there is no past, and that is what
the warm-up below is for.

**The whitening.** The autoregressive model is fitted on a separate learning
stretch, but the lattice filter that applies it also carries state, and
`DoubleWhitening` additionally needs a buffer of future data (`ExtraSize`)
before its own backward pass can produce a good estimate.

## Why the discarded stretch is not a fixed number

`padlen` is measured, not chosen. `settling_length` drives an impulse through
the designed filter and finds where the response has decayed below a fraction of
its peak. This matters because a filter's ringing is not read off its order: a
steep transition close to Nyquist rings far longer than a gentle one of higher
order, a narrow notch longer still, and the stretch that has to be discarded
follows the impulse response rather than the parameter that produced it. The
response is followed until it has stayed below the floor for as long again as
it took to get there, however long that is, up to a stated limit; a filter that
rings past the limit is refused rather than reported at it.

A filter that is asked to settle in less than it needs does not fail -- it emits
the unsettled transient as if it were data, at the start of every block, where
it looks like a short broadband burst and lands in the finest wavelet scales.
Measuring the settling is what prevents that: `padlen` is whatever the designed
filter turns out to need, so raising `FilterOrder` or steepening the band
lengthens the discarded stretch instead of corrupting the emitted one.

The settling stretch does not have to fit inside a single read. `Process`
buffers what it has read and emits a block only once `padlen` samples of real
future data have arrived, returning `None` meanwhile, however many reads that
takes; a block whose future is already in is emitted without reading more
(`emit`, which `read_conditioned` asks first), so what the front end holds is
its settling and one read, whatever the sizes of the reads before. The cost is
therefore latency rather than a constraint on the filter, and it is stated in
`latency_s` and carried by the timestamps.

## What is discarded, in order

1. The warm-up: one-second reads that are conditioned and whitened but not
   searched. There are at least as many as the conditioning's settling plus the
   whitening's order, in seconds, since the first searched sample needs both
   filters to have settled on real data: the band-pass over `padlen` samples of
   it, and the whitening's forward pass, which is FIR, over its order. `preWhite`
   asks for more when it is larger, and the number used is what the run records.
2. `WhiteningExtraSize` samples buffered ahead of the detection loop, so the
   whitening's backward pass has its lookahead before the first output block.

The discarded stretch therefore grows with the filters: a few seconds for the
band-pass alone, and the settling of the narrowest notch when lines are
notched.

## Could they be analysed?

Offline, yes, and nothing about the data itself is bad -- it is discarded
because the filters have not settled *going forward*, not because the strain is
unusable. Conditioning them correctly needs real data before them as well as
after, which the segment does not hold. The stretch the noise model is fitted
on is in the opposite situation: it lies inside the segment, so it is read
together with `padlen` samples of real data on each side and conditioned by
`condition_stretch`, which keeps only what lies between. Filtered alone, as a
block complete in itself, its edges would carry the filter's start and the model
would be fitted on it as though it were the noise.

Recovering them is not done, for two reasons worth stating plainly.
The seconds recovered are a negligible fraction of any real observing segment,
and a stretch conditioned by a different path is not guaranteed to have the same
noise properties as the rest, so triggers from it would not be directly
comparable to the rest of the run. A search whose background estimate has to
hold across the whole segment is better served by a uniform treatment than by a
few extra seconds.

In a low-latency setting the same seconds are a real startup cost and cannot be
recovered at all, since the future data they need has not arrived yet.
