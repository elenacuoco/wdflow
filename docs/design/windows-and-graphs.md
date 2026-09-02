# The analysis window, and two levels of graph

The search reads the strain in blocks of one length, and what it produces is
read through two graphs: one inside a detector, one across the network.

```
h_d(t) → WDF, one analysis window
       → detector stage: the detector's triggers as a graph over (t, f, scale)
       → single-detector events
       → network stage: those events as an inter-detector graph
       → a ranking statistic
       → a false-alarm rate from time slides
```

## Why one window length, and an overlap

A dyadic transform is already multi-resolution: the level of a coefficient and
the band it occupies are tied to each other, so a tile of a given band lasts the
same however long the block is. Doubling the window does **not** insert bands
between the octaves — it extends the dyadic ladder one octave downward — so
searching the same strain at several lengths repeats one tiling on shifted grids
of blocks rather than adding resolution, and pays a trials factor for the
repetition.

The transient longer than the block is therefore not the window's problem but
the grouping's. The block is a unit of computation; the event is the physical
object, assembled afterwards from the coefficients of every block it touched,
and its statistic is measured on the reconstruction stitched across them.

The overlap is what the window length is not. A transient shorter than the block
that falls astride a boundary is divided between two blocks and has a fraction
of its energy in each; an overlap of half the window puts every sample in the
body of two blocks, so such a transient is seen whole at least once. It also
gives the reconstruction two estimates of every overlapped sample, which is what
lets the two be crossed over continuously instead of switched between.

The window remains a list in the configuration
(`wdf.config.Parameters.window_schedule`), because the length is a parameter of
the search and not a property of it. A run configured at more than one length
searches each as a search of its own — its own stride, its own coefficient grid,
its own trigger file — over a single conditioning pass, and the record
distinguishes them, `n_coeff` and `fs` being stored per trigger.

## Making several lengths comparable, when there are several

`EnWDF` is a norm over a block, and the Donoho–Johnstone threshold deciding
which coefficients survive depends on how many the block holds, so it is not the
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
number of lengths. Under a run at one length it is degenerate and the
significance reduces to the single-scale case.

## The detector stage: `wdf.analysis.detector_graph`

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
detector-stage model does.

A node carries its trigger's wavegram on a band-by-time grid, so what reaches the
model is how the trigger looks in the plane. The rows are indexed by absolute
frequency band rather than by octave level: since a longer window extends the
ladder downward instead of subdividing it, the same physical band is the same row
at every length.

## The file between the two stages: `wdf.analysis.event_store`

The detector stage and the network stage are separated by a file. The first
reads frames and writes, per detector and frame kind, the events it assembled
together with the triggers they were built from, each trigger carrying the
label of the event it belongs to. The second reads that pair and needs nothing
else: an event's wavegram and its reconstruction are functions of the
coefficients it kept, which the triggers carry, so the store is what the graph
builder is given.

The order is part of the contract. The network stage prepares its node-side
arrays once and indexes them by position, so a store that reordered the events
would hand the second stage a different catalogue.

What this buys is that a change to the coincidence, to the ranking or to the
learned stage costs the network stage's time and not the search's.

## The network stage: `wdf.analysis.network_graph`

The detector stage's events become the nodes. An edge exists only where a signal could
have produced the pair: the two events must cover the same stretch of time once
one of them is allowed to shift by the light travel time plus their own timing
spreads, and they must overlap in band. These are the candidates
`wdf.analysis.robust_events.IndexedCoincidenceFinder.candidate_edges` admits, so
the learned and the classical statistic rank one population.

The arrival-time difference an edge carries is read on the two events'
stitched reconstructions, at the lag that aligns them, with the uncertainty the
correlation peak declares. The instant an event reports is the centre of the
tile carrying its largest coefficient, and in a dyadic transform a tile's
length is tied to its band, so two detectors whose loudest tile falls on
different rungs of the ladder report instants displaced by the difference of
two tile lengths; the reconstruction carries the waveform at the sample. What
the correlation measures is a property of the two waveforms and not of a
slide's shift, so it is measured once per pair of events and reused by every
slide.

The test is on the events' extents and not on any instant of them. An extended
transient has no arrival time: which moment a detector calls its centroid or its
peak depends on its own noise, its antenna response and which coefficients
survived threshold, so the two detectors do not agree on it. For a transient
shorter than the light travel time the two statements coincide, which is why the
extent test is the general one. The arrival-time difference is still measured
and still ranks the survivors; it no longer decides which pairs exist.

The map two detectors are compared on is anchored the same way: on the centre of
the tile carrying the loudest member's largest coefficient, not on the event's
energy centroid. An event kept as one block and its counterpart kept as five have
centroids far apart, so maps centred on them sit a large part of their own width
from each other and their agreement measures the difference in extent rather than
the difference in morphology. The anchor is an instant both detectors measure on
the same transient; the centroid is a property of what each of them recovered.

Where that difference is measured, an event is timed on `gpsPeak`, the centre of
the tile carrying its largest coefficient, and not on `gpsCentroid`. The two are
different quantities and only one of them is a clock. A centroid is a moment of
the energy that survived threshold in that detector; two detectors seeing one
source at different projected amplitudes keep different portions of it and place
their centroids differently, and that difference is indistinguishable from
geometry once it enters the arrival-time difference. The peak tile tracks the
loudest instant instead, which both detectors share. The remaining uncertainty
is then the duration of that tile, since the time assigned is its centre rather
than an instant within it, and `tSpread` is what declares it.

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
`wdf.analysis.robust_events.TimeSlideFAR` displaces every detector but the first
by an amount of its own, each far larger than the light travel time, which leaves
each detector's noise intact while destroying any real coincidence; the
false-alarm rate follows from the resulting distribution. With more than two
detectors the pair that excludes the reference is decorrelated only by the
*difference* of two shifts, so the draw is repeated until every difference, and
not merely every shift, clears the minimum: two detectors moved by nearly the
same amount would stay in step and their own pair would keep its real
coincidences.

An event's own statistic is made comparable across sizes by
`wdf.analysis.event_significance`, which maps it through the background
distribution of events holding the same number of tiles. The tiles are what the
statistic sums, so their count is what sets its scale under the null; the number
of blocks is a property of the analysis grid, and conditioning on it would make
an event's significance depend on where the grid started. Inside the measured
range the mapping is the empirical survival, so a calibrated background is
exponential with unit rate by construction --- for an event the calibration was
not fitted on. An event the calibration contains counts itself, and the j-th
largest of a bin then scores `log((N + 1) / (j + 1))` whatever the data: a
ladder fixed by the bin sizes, which no member can leave for the extrapolated
branch a candidate reaches. `out_of_sample_significance` scores a background
fold by fold from the others, and `significance_off_source` scores any event
from the background of every time fold but its own, which is what a single
recorded stretch allows. Beyond the largest value a bin
measured it continues along an exponential fitted to that bin's own upper tail.
The continuation is what keeps the statistic usable: an empirical survival cannot
fall below one count in its bin, so on its own it caps the significance at the
logarithm of the bin's size, and a threshold beyond that cap silently rejects
every event of that extent however loud it is --- which is exactly the class the
grouping exists to build. The extrapolation is a stated model of the tail and the
edge of the sample records where the measurement ends.

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
