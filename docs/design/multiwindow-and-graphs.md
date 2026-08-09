# Several window lengths, and two levels of graph

The search runs at more than one analysis window length, and what it produces is
read through two graphs: one inside a detector, one across the network.

```
h_d(t) → multi-window WDF
       → level one: the detector's triggers as a graph over (t, f, scale)
       → single-detector events
       → level two: those events as an inter-detector network graph
       → a ranking statistic
       → a false-alarm rate from time slides
```

## Why more than one window length

A transient longer than the analysis window cannot have its extent measured
inside one, and only a longer window reaches the bands below. The window is
therefore a list in the configuration (`wdf.config.Parameters.window_schedule`),
and every length is a search of its own — its own stride, its own coefficient
grid, its own trigger file — over a single conditioning pass. The record already
distinguishes them: `n_coeff` and `fs` are stored per trigger.

Doubling the window does **not** insert bands between the octaves. It extends the
dyadic ladder one octave downward and makes every tile twice as long. So a longer
window buys duration and low frequency directly, and frequency only indirectly.

## Making the lengths comparable

`EnWDF` is a norm over a window, and the Donoho–Johnstone threshold deciding
which coefficients survive depends on how many the window holds, so it is not the
same quantity at two lengths. Every length also reads the same strain, so adding
the results counts one transient's energy several times.

`wdf.analysis.scale` maps each length through its own measured background
instead. For window length `L` and octave band `b`,

```
S = -log P(E' ≥ E | H0, L, b)
```

which is exponential with unit rate whatever produced it. The band belongs in the
key because the bands are shared exactly between lengths wherever both reach
them, so this is a lookup and not a binning choice.

`scale_maximum` forms the cross-scale maximum together with the scale signature —
how many lengths saw a tile, and how evenly. That maximum is taken over
correlated searches and carries a look-elsewhere effect whose size is a property
of the data, so it is measured on the background rather than assumed to be the
number of lengths.

## Level one: `wdf.analysis.detector_graph`

A node is one trigger. An edge joins two triggers close in time relative to their
window span and overlapping in band, which covers both neighbours at one length
and triggers of different lengths covering the same region of the plane. The
connected components are the detector's events.

The trigger is the node rather than the tile because the labels that train the
model exist only there — an injection is stated to belong to a trigger, and
nothing states which tiles are one transient — and because the statistic is a
property of a window. The graph is also rebuilt once per time slide, so the node
count is paid a hundred times over.

Connected components over every admissible edge is not the answer: triggers close
in time and overlapping in band are common in noise, so what percolates is the
noise. `components(keep)` takes a mask, and deciding that mask is what the
level-one model does.

A node carries its trigger's wavegram on a band-by-time grid, so what reaches the
model is how the trigger looks in the plane. The rows are indexed by absolute
frequency band rather than by octave level: since a longer window extends the
ladder downward instead of subdividing it, the same physical band is the same row
at every length.

## Level two: `wdf.analysis.network_graph`

The level-one events become the nodes. An edge exists only where a signal could
have produced the pair: the two events must cover the same stretch of time once
one of them is allowed to shift by the light travel time plus their own timing
spreads, and they must overlap in band. These are the candidates
`wdf.analysis.robust_events.IndexedCoincidenceFinder.candidate_edges` admits, so
the learned and the classical statistic rank one population.

The test is on the events' extents and not on any instant of them. An extended
transient has no arrival time: which instant a detector calls its centroid or its
peak depends on its own noise, its antenna response and which coefficients
survived threshold. On the simulated set two detectors seeing one compact binary
put their centroids a median of 77 ms apart and their peaks 14 ms apart, against
a light travel time of 10 ms — so gating on either admitted 19 of 108 coincident
injections while gating on the extents admitted 71. For a transient shorter than
the light travel time the two statements coincide, which is why the extent test
is the general one. The arrival-time difference is still measured and still ranks
the survivors; it no longer decides which pairs exist.

An edge carries the arrival-time difference, the shared fraction of band and of
time support, the log ratio of the two energies, and the agreement between the
two wavegrams. No alignment is searched for: each event's map is centred on its
own energy, so the arrival-time difference is not in the maps at all. The energy
ratio is a feature and not a penalty: the antenna responses make unequal
amplitudes between detectors physical.

The maps are rendered twice, because two questions want opposite resolutions.
The assembly map joins windows seconds apart into one transient and needs wide
columns and a long span; the comparison map is what two detectors are matched on
and needs columns of the order of the light travel time. `event_coefficients`
takes both the column width and the number of columns, so the same function
produces either.

## Significance

A model's output is not a probability of astrophysical origin.
`wdf.analysis.robust_events.TimeSlideFAR` shifts the detectors against each other
by much more than the light travel time, which leaves each detector's noise intact
while destroying any real coincidence, and the false-alarm rate follows from the
resulting distribution.

The zero-lag candidates and the slid candidates come from the same finder, so they
are one population differently ordered. `wdf.analysis.baseline` provides a
deterministic ranking through the same background, because a learned statistic
that does not beat it at fixed false-alarm rate is not worth introducing.

`wdf.analysis.anomaly.BackgroundAnomalyScorer` is the learned one, and it is
fitted on accidental coincidences alone. Candidates built from time-slid data, or
from a stretch of noise holding no injection, are accidental by construction and
available in quantity; a graph autoencoder fitted to reproduce that population
scores a candidate by how badly it fails to reproduce it. Its `fit` takes graphs
and nothing else — no label, no injection — so the selection cannot depend on the
waveform family the model was shown. It is the un-modelled counterpart of a
likelihood ratio: the denominator is measured and the numerator, which needs a
signal model, is never formed.

Efficiency is read as the excess over an accidental floor. Injections placed in
one detector only cannot be recovered in coincidence, so whatever fraction of
them a statistic appears to recover measures the rate at which some candidate
happens to fall inside the matching window. An efficiency at or below that floor
is not evidence of recovery.

## Scale

Nothing here forms a dense pairwise matrix. `wdf.analysis.pairs` finds the pairs
inside a tolerance by searching a sorted time axis, which is what makes a search
carrying no per-detector threshold tractable: at these event rates an `n × n`
array is tens of gigabytes, and the graphs are rebuilt once per slide.
