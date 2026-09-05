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

The detector stage's events become the nodes. An edge exists only where a signal
could have produced the pair: the two events must cover the same stretch of time
once one of them is allowed to shift by the light travel time widened by what
each event declares its instant is worth, and they must overlap in band. These
are the candidates
`wdf.analysis.robust_events.IndexedCoincidenceFinder.candidate_edges` admits, so
the learned and the classical statistic rank one population. The difference of
the two events' own instants is measured on every admitted pair and ranks it; it
does not decide which pairs exist.

The arrival-time difference an edge carries is the difference of the two
events' own instants: the centre of the tile carrying each event's largest
coefficient, found on the event's own wavegram rather than on one block's.
`wdf.analysis.robust_events.INSTANT_COLUMNS` names where that is read from and
in what order it is preferred, the energy centroid being the last resort.

What such a difference can resolve is bounded by the tiling. A tile lasts one
over the upper edge of its own band --- a few milliseconds high in the band,
tens of milliseconds low in it --- so where an event's loudest coefficient sits
low the tile is longer than the light travel time of the network, and the
difference of two tile centres is then a statement about the two tilings and
not about the geometry. That is a bound on what an edge's timing is worth, and
`tSpread` is what declares it; a candidate whose direction is wanted is
measured again, below the tile, on its reconstruction.

It is a property of one event and of nothing else. A time slide moves the
event's times and carries its instant with them, so a difference of two of them
is a difference of two node quantities and nothing is measured per pair --- on
the zero-lag population or on a background of tens of millions of accidentals.
That is what an edge needs: how much of its timing tolerance the pair consumed,
on every pair the admission rule left it.

What a candidate deserves is finer, and it is measured outside these stages.
`wdf.analysis.timing.arrival_time_difference` reads the difference at the lag
that maximises the cross-correlation of the two events' stitched
reconstructions, with the uncertainty the correlation peak declares, which is
what a sky region wants. It costs a correlation per pair, so it belongs to the
candidates a map is drawn for and not to the graph, and no stage here calls it:
a study that wants a direction applies it to its own candidates.

Admission is on the events' stretches of time and not on the difference of
their instants. A transient longer than one analysis window is assembled as
several events and the two detectors need not keep the same one, so their
instants can differ by far more than the light travel time although both belong
to one signal; gating on that difference would take evidence away from a
candidate the detectors did assemble. The difference ranks the pair instead,
which is answerable only because each event has an instant of its own, taken on
the tile it was ranked on rather than on where its energy happened to sit --- a
centroid is a property of what that detector recovered, and the two detectors
do not agree on it.

The map two detectors are compared on is anchored the same way: on the centre of
the tile carrying the loudest member's largest coefficient, not on the event's
energy centroid. An event kept as one block and its counterpart kept as five have
centroids far apart, so maps centred on them sit a large part of their own width
from each other and their agreement measures the difference in extent rather than
the difference in morphology. The anchor is an instant both detectors measure on
the same transient; the centroid is a property of what each of them recovered.

`gpsCentroid` is last in `INSTANT_COLUMNS` because it is not a clock. A
centroid is a moment of the energy that survived threshold in that detector;
two detectors seeing one source at different projected amplitudes keep
different portions of it and place their centroids differently, and that
difference is indistinguishable from geometry once it enters the arrival-time
difference. The peak tile tracks the loudest instant, which both detectors
share. Where an event is timed on the tile centre, the
uncertainty is the duration of that tile, since the time assigned is its centre
rather than an instant within it, and `tSpread` is what declares it.

An edge carries the arrival-time difference, the shared fraction of band and of
time support, the log ratio of the two energies, the coherent amplitude over the
tiles the pair shares, and the agreement between the two wavegrams. The energy
ratio is a feature and not a penalty: the antenna responses make unequal
amplitudes between detectors physical.

The maps are rendered twice, because two questions want opposite resolutions.
The assembly map joins windows seconds apart into one transient and needs wide
columns and a long span; the comparison map carries the tiles two detectors are
matched on. `event_coefficients` takes both the column width and the number of
columns, so the same function produces either.

The grid the match itself is measured on is not either rendering's column: it is
derived by the network stage from the two things the comparison has to resolve.
A bin coarser than a tile adds the signed coefficients of one detector to each
other before either detector meets the other, so an oscillating transient
cancels against itself and the highest bands, whose tiles are shortest, lose
most --- a frequency prior the search does not carry. A bin coarser than the
light travel time cannot represent the delay the coincidence exists to measure,
so the lag axis collapses onto a single point whatever tolerance the edge was
admitted on. The bin is therefore the shorter of the shortest tile the band
ladder holds and the network's light travel time.

The two maps are compared as magnitudes. A coefficient carries the phase as
well as the energy: two detectors resolve one transient onto different basis
functions, at an arrival-time difference finer than a bin, and the two LIGO
detectors respond to one source with opposite sign besides, so their
coefficients disagree cell by cell where the morphologies are identical. A
cosine between signed maps measures that disagreement rather than the shape,
and taking its magnitude afterwards does not undo it, because the cancellation
is inside the sum and not on the total. The sign is not discarded from the
analysis --- the coherent energy is a signed sum whose magnitude is taken once
at the end --- only from the question of whether two maps have the same shape.

Each event's map is laid on that event's own instant, and the difference between
the two anchors is carried alongside, so a map lag `L` places the pair at the
absolute displacement `offset + L`. The lags searched are the ones that reach
the displacements the tolerance admits, which for a pair whose anchors already
differ by the tolerance are not the lags near zero: an axis of plus or minus the
tolerance would report the largest agreement among displacements the coincidence
does not admit.

The match is formed only where the two events' own instants are already within
that tolerance. Admission is on the events' stretches of time and stays so, for
the reason above, but no displacement the tolerance permits brings the instants
of a pair further apart than it together: what a search over those
displacements would find is the agreement between the tail of one event and the
head of the other, at the price of the trials factor of the search. Such a pair
keeps its edge and every statistic that needs no displacement, and its match is
no agreement.

An event is rendered once, on a window centred on its own instant and no wider
than the event reaches, and the grid two events are compared on is the longer of
those two widened by the displacements to be searched. It is a property of the
pair and not of the run: what a run holds is not known in advance, a binary
neutron star lasts minutes where a black-hole merger lasts a fraction of a
second, and rendering every event on the widest would let the long transient fix
the grid of every short pair --- at a bin of the shortest tile, an array that
does not exist. Pairs sharing a grid are compared together, by powers of two, so
a grid is built once per scale and never once per pair.

The displacement found is reported beside the agreement, out of the same pass,
so it is not a second estimate to be reconciled with it. It is read on the
`gpsPeak` clock the two maps are anchored on, which is not the clock `dt_s` is
measured on, and a pair compared at no displacement reports none rather than the
first point of an axis.

The anchor difference is split to search them on one axis: a whole number of
bins, applied as a shift of the map, and what the rounding leaves over. Pairs
sharing a whole part share their shifts, so what is searched is always the
tolerance's own span, widened by the half bin the rounding can leave, and never
the span between two anchors. The maximum is therefore taken over the pair's
admitted arrival-time differences and over nothing else, whatever those two
anchors are; a pair is compared only where the light travel time allows it to
be one signal, and the match is not formed at all for a pair the coincidence
did not admit.

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
recorded stretch allows. Beyond the largest value a bin measured it continues
along an exponential fitted to that bin's own upper tail.
The continuation is what keeps the statistic usable: an empirical survival cannot
fall below one count in its bin, so on its own it caps the significance at the
logarithm of the bin's size, and a threshold beyond that cap silently rejects
every event of that extent however loud it is --- which is exactly the class the
grouping exists to build. The extrapolation is a stated model of the tail and the
edge of the sample records where the measurement ends.

The zero-lag candidates and the slid candidates come from the same finder, so they
are one population differently ordered. A learned statistic that does not beat a
deterministic one at fixed false-alarm rate is not worth introducing, so the
candidate table carries the deterministic ranking beside every learned column.

That ranking is `network_morphology`, the root of the magnitude of the coherent
energy: the product of the two events' coefficient amplitudes on their noise
scales, summed over the tiles that cover the same place on the plane, and
carrying the coefficients' signs. The column is a coherent *amplitude*, on the
noise scale, and a ranking that reads it as an energy has to square it back. A
product of magnitudes is positive whatever the data, so its mean under the null
grows with the number of tile pairs that happen to meet, and two long events
overlapping by accident then outrank two short ones describing one transient.
The signed product has mean zero under the null, which is what makes summing
over many tiles pay for agreement rather than for extent. Both polarities are
physical --- two detectors can respond to one source with opposite sign --- so
the magnitude is what ranks. A tile carries an absolute time, so where a stage
displaces an event --- a time slide does --- the tiles are carried with it and
the pair is compared where it stands rather than where it stood.

The table also carries loudness-only readings (`network_enwdf`,
`network_min_enwdf`, and `network_min_enwdf_timed`, which discounts a pair by
the fraction of its timing tolerance it consumed), which ask that both
detectors were loud and never whether they agree.

Two learned rankings read the same graph.
`wdf.analysis.gnn.GNNCoincidenceScorer` is supervised: its `fit` takes graphs
with a 0/1 label per cross edge, and it adds `gnn_logit` to the candidate table.
`wdf.analysis.anomaly.BackgroundAnomalyScorer` is not: it is fitted on
accidental coincidences alone. Candidates built from time-slid data, or from a
stretch of noise holding no injection, are accidental by construction and
available in quantity; a graph autoencoder fitted to reproduce that population
scores a candidate by how badly it fails to reproduce it. Its `fit` takes graphs
and nothing else — no label, no injection — so the selection cannot depend on the
waveform family the model was shown. It is the un-modelled counterpart of a
likelihood ratio: the denominator is measured and the numerator, which needs a
signal model, is never formed.

`WavegramCoincidenceFinder` given a scorer returns the learned column on every
set of pairs it describes, so a background and the candidates it calibrates are
scored by one model over one population. On that path the table carries
`gnn_logit` in single precision and not the sigmoid `gnn_score`, which is a
function of it: a slid background is many millions of rows per shift.

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
