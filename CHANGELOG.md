# Changelog

Versions follow [semantic versioning](https://semver.org). A release records
what the software does differently, not how it came to.

## 1.2.0 --- 2026-09-04

### The wavegram match is measured on a grid that can resolve the delay

The bin two detectors' maps are compared on is derived by the network stage
from what the comparison has to represent, instead of being inherited from the
column width of whichever rendering was passed to it. Two bounds fix it and
both are physical: a bin coarser than a tile adds one detector's signed
coefficients to each other before either detector meets the other, so an
oscillating transient cancels against itself and the bands whose tiles are
shortest lose most, which is a frequency prior the search does not carry; a bin
coarser than the light travel time cannot represent the delay the coincidence
exists to measure, so the lag axis holds one usable point whatever tolerance
the edge was admitted on. The bin is the shorter of the shortest tile the band
ladder holds and the network's light travel time.

### The comparison of two wavegrams is a baseline, and is optional

`TriggerGraphBuilder` takes `match_wavegrams`. The comparison of a coincident
pair's two renderings is bounded by one, carries no loudness and maximises over
displacements, so it is a morphological baseline and enters no ranking; it
costs a correlation per candidate, and a study that does not read it need not
pay for it. With it off `network_wavegram_match`, `network_correlation` and
`coherent_statistic` are not measured, and `network_wavegram_matched` says so
rather than a zero standing in for a measurement.

The displacements searched are the timing tolerance's. Each rendering is laid
on its own event's instant and the whole bins of the difference between the two
instants are applied as a shift, at no cost whatever that difference is, so the
alignment at an arrival-time difference the light travel time allows is already
reachable. Searching as far as the transient lasts would only add agreements at
displacements the geometry forbids, at a cost growing as the square of the
transient's own length.

### One coherent amplitude, not two names for it

`network_block_morphology` was a literal copy of `network_morphology` and is
gone; `CoherentRanking` reads the one that remains. `network_morphology` is the
root of the magnitude of the signed coherent energy, so it is the geometric
mean of the two events' amplitudes on their noise scales reduced by the root of
the agreement between their tiles --- a quantity that vanishes when either
detector is silent, which the sum in quadrature of the two does not.

### The grid a pair is compared on comes from that pair

An event is rendered once, on a window centred on its own instant and no wider
than the event reaches. What two events are compared on is then the longer of
the two, widened by the displacements to be searched, and is a property of that
pair alone. Rendering every event on the widest event of the run made a
transient lasting minutes fix the grid of every millisecond-long one; what a
run holds is not known in advance, and at a bin of the shortest tile that array
does not exist. Pairs are handled in groups sharing a grid, by powers of two,
so a grid is built once per scale and no loop runs over pairs.

Because each pair is searched over its own transient, two pairs' profiles are
of one length only on the part they share. What `TriggerGraph` carries per edge
is therefore the profile on the timing tolerance's own lags, which every pair
can be read on and which a model can take as a feature of fixed size, together
with the agreement, the displacement it was found at and whether the pair was
compared at all --- taken where the comparison took them, over the pair's whole
search, and not reduced a second time from the part that was kept.

### The match is slid over the transient, not over the light travel time

Each event's rendering is laid on that event's own instant, the centre of the
tile carrying its largest coefficient. Two detectors seeing one transient at
different amplitudes need not make that the same part of it, so their two
renderings can be displaced by as much as the transient lasts while their
arrival times differ by no more than the light travel time. The displacements
searched are therefore the longer of the two events' own extents --- measured
on the tiles the event is made of --- widened by the timing tolerance, and not
the tolerance alone. Searching the tolerance alone cannot reach the alignment
of a pair whose two detectors anchored on different parts of one transient, and
what it reports for such a pair is the largest agreement among displacements
that do not align it.

The displacement found is reported beside the agreement, from the same pass, so
it is not a second estimate to be reconciled with it. It is not constrained to
the light travel time: a displacement larger than the geometry allows is the
match saying the two renderings do not align physically, which is a statement
about the pair rather than a truncation of the search. `candidate_table` now
also carries whether a pair was compared at all, so that a match of zero is
distinguishable from a match that was never formed, and a pair never compared
reports no displacement rather than the first point of the lag axis.

### The match is asked only of a pair already coincident in time

A pair is admitted on the events' stretches of time, not on the difference of
their instants, and that stays as it is: a transient longer than one analysis
window is assembled as several events and the two detectors need not keep the
same one, so gating admission on the instants would take a candidate away from
detectors that did assemble it. The wavegram match, though, is a comparison of
two morphologies at a displacement, and it is now formed only where the two
events' own instants are already within the tolerance the geometry and their
timing spreads allow. No displacement the tolerance permits brings the instants
of a pair further apart than that together, so a search over those
displacements would report the agreement between the tail of one event and the
head of the other, and pay the trials factor of the search for it. Such a pair
keeps its edge and every statistic that needs no displacement; its match is
reported as no agreement.

### The lags searched are the ones the pair admits

Each event's map is laid on that event's own instant, so a map lag `L` places
the pair at the absolute displacement `offset + L`, with `offset` the
difference of the two anchors. `correlation_profiles` now searches the lags
that reach the displacements the tolerance admits rather than the lags within
plus or minus the tolerance: a pair whose two anchors already differ by the
tolerance reaches zero displacement only at `L = -offset`, and the agreement
reported for it was the largest among displacements the coincidence does not
admit.

The anchor difference is split for that: a whole number of bins, applied as a
shift of the map, and what the rounding leaves over. Pairs sharing a whole part
share their shifts, so the axis searched is one axis --- the displacements the
tolerance admits, widened by the half bin the rounding can leave --- whatever
the anchors are, and the work stops growing with how far apart anchors happen
to be. The function returns that axis together with each pair's leftover, and
the absolute displacement of a pair at a lag is the sum of the two.

The reduction inside that search is taken in numpy rather than through
`paired_dot`. Sending a matrix to a device is the right trade when one call
reduces a whole pair set against it, and the wrong one here: a shift changes
the slice, so no device copy survives to the next lag, and each reduction
touches a handful of pairs of a matrix of gigabytes. Measured on this stage's
shape, the transfer is three orders of magnitude more than the arithmetic it
carries.

`TriggerGraph.candidate_table` reports that displacement as the difference of
the two events' `gpsPeak` plus the map lag. It previously added the map lag to
`dt_s`, which is measured on whichever instant column the coincidence prefers,
so the two clocks differed by each detector's envelope-to-peak shift --- an
error that does not cancel in the difference and goes straight into the
arrival-time difference and the sky position.

### The normalisation is the event's, not the edge's

The norm of a map is a property of the event. Taking it once per edge gathered
one map per pair before reducing it, which is the allocation this stage cannot
afford at a background's pair count.

### Removed

`network_coherent_shape`, a sum of `network_morphology` and the logarithm of
the wavegram match. The first term is a coherent amplitude and the second is
dimensionless, so the sum has no reading; the floor the logarithm was guarded
with turned a match of exactly zero into an exclusion of 708 nats in a column
whose signal variation is of order one. A shape term belongs in
`wdf.analysis.network_statistic.CoherentRanking`, where it enters as a measured
log density ratio in the units the coherent energy is already in.

## 1.2.0 --- 2026-09-03

### A pair is admitted on the arrival-time difference

The network stage admits a pair when the difference of the two events' own
instants does not exceed the light travel time, widened by what each event
declares its instant is worth --- the tolerance
`CoincidenceConfig.timing_tolerance` already computed and which, until now,
only widened an overlap of extents. An extent says how long a transient lasted
and not when it arrived, so two long events overlapping for seconds were
admitted as causally compatible when they are not; the accidental population
was that of pairs which happen to be simultaneous.

The window a pair is admitted in is therefore the coincidence's own tolerance
cap, tens of milliseconds rather than the seconds an extent reaches.
`TimeSlideFAR` checks a step against it, and against the length the trigger
stream is correlated over, and refuses one below either: under the first a
real coincidence stays admissible, under the second two lags re-use the same
clusters. The step itself is a stated constant --- `FARConfig.min_shift_s`,
four seconds --- as published burst searches state theirs; None derives it
from those two measurements instead, which makes the step a property of the
run rather than of the analysis.

### An event carries its own instant

`wdf.analysis.timing.envelope_instant` reads one event's arrival time on its
own stitched reconstruction: the peak of the analytic envelope, sought within a
stated width of the instant the event was ranked on. The detector stage writes
it as `gpsEnvelope`, and `wdf.analysis.robust_events.INSTANT_COLUMNS` names the
order a coincidence prefers its instants in --- that one, then the centre of the
tile carrying the largest coefficient, then the energy centroid. A tile lasts
one over the upper edge of its own band, so where an event's largest
coefficient sits low the tile is longer than the light travel time of the
network and a difference of two tile centres carries no geometry; the
reconstruction carries the waveform at the sample. Where the envelope cannot be
read the tile centre answers, so the column always holds the best instant an
event has.

`TimeSlideFAR` displaces every instant column together, so an event's instant
and its tiles stay on one clock through a slide.

### The two arrival-time differences are separated

A network edge carries the difference of the two events' own instants. That is
a difference of two node quantities: nothing is measured per pair, on the
unshifted population or on a background of accidentals.

A candidate's arrival-time difference is measured on the pair.
`arrival_time_difference` returns the lag that maximises the cross-correlation
of the two stitched reconstructions, with the width the correlation peak
declares, and forms only the lags the network's geometry allows. A placement it
cannot reach --- two series further apart than the maximum lag --- raises rather
than returning a bound. This is what a sky region is built from, and it is
applied to candidates and not to the graph.

### A displaced event carries its tiles

The coherent statistic is summed over the tiles two events share, and a tile
carries an absolute time. The node-side arrays are prepared once and reused for
every slide, so a slide that moved an event and not its tiles compared the pair
at the place it used to occupy: no slid pair shared a tile, and the statistic
was zero on the whole accidental population by construction rather than by
measurement. `tile_coherence_many` now takes a `displacement` per event, and
`TriggerGraphBuilder.build_from_prepared` measures it from the events
themselves --- the difference between the instant an event holds now and the
one it held when its tiles were laid out --- so it needs no knowledge of what
moved them. A displacement of zero is the arithmetic that was there before, so
the unshifted population is unchanged to the bit.

### The coherent statistic obeys the rule the detector stage obeys

An event carries the tiles of every block it spans and, beside them, the tiles
of the one block it was selected on
(`wdf.analysis.detector_graph.EventWavegram.block_tiles`, the member with the
largest per-window energy). The candidate table carries the coherent energy
over both: `network_morphology` over the whole event, and
`network_block_morphology` over the selecting blocks. A statistic summed over
tiles carries the threshold's floor once per tile, and a signed sum has mean
zero but a variance that grows with the number of tile pairs, so a sum over the
whole event is ranked partly on how long the transient lasted; one block bounds
that count. `block_coherence` joins `EDGE_FEATURES`, so the learned ranking
reads it too --- a model fitted before this carries a different number of edge
inputs and cannot be loaded against it.

### The candidate table

`network_morphology`, the magnitude of the signed coherent energy over the
tiles a pair shares, is the deterministic ranking beside the learned columns.
`WavegramCoincidenceFinder` given a scorer returns the learned reading on every
set of pairs it describes, so a background and the candidates it calibrates are
scored by one model over one population; on that path the table carries
`gnn_logit` in single precision and not the sigmoid `gnn_score`, which is a
function of it.

### Cost

`rescore_on_reconstruction` takes `keep_series`, so a caller that reads the
statistic alone does not hold a waveform per event. An event is inverted once
and the samples cross the C++ boundary in one narrowing rather than one per
sample. `wdf.analysis.pairs.paired_dot` takes a `resident` cache, so a matrix
re-used across calls is sent to the device once. The learned scorer normalises
its node features in place on the difference it already owns, holding one copy
of that matrix and not two.

## 1.1.1 --- 2026-09-02

Documentation only; the code is unchanged from 1.1.0.

The README, the design note and the published page state what the pipeline
does: an event's energy is the norm of its stitched reconstruction and what
ranks it for detection is its loudest block, the significance conditions on the
number of tiles and is read out of sample, and the scale a block is ranked on
is the median of its neighbours' while the thresholding stays with the block
itself.

## 1.1.0 --- 2026-09-02

### The statistic that selects is not the statistic that measures

An event's energy is the norm over its tiles; what ranks it for detection is
its loudest block. A hard threshold admits every tile at a floor of `2 ln N` in
normalised energy, so a sum over tiles accumulates that floor in the noise as
well as in the signal and a tile earns its place only when its excess exceeds
half the event's mean excess per tile. Ranking on the loudest block maps each
cluster to a maximum over quantities the search already computed, so its
background is a subset of the one an ungrouped search produces and the grouping
reduces the trials factor without being able to lose a candidate.

- `network_graph` ranks its nodes on `EnWDF_window` where the node table
  carries it.

### Nothing is calibrated on its own time

- `event_significance.out_of_sample_significance` scores each fold of a
  background from the others. A calibration read on the sample it was fitted on
  scores the j-th largest event of a bin as `log((N + 1) / (j + 1))` whatever
  the data, so a rate read off it is lower than a fresh background of the same
  livetime gives.
- `event_significance.significance_off_source` scores any event from the
  background of every time fold but its own, which is what a single recorded
  stretch allows: there the background is a part of the same data and a
  candidate lives beside it.
- `EventCalibration` conditions on `n_pixels`, the tiles the statistic sums,
  instead of on the blocks the analysis grid cut. The tile count sets the
  statistic's scale under the null; the block count is a property of the grid
  and made an event's significance depend on where the grid started.
- The exponential tail divides the total excess by `k - 1` rather than `k`, one
  of the `k` values being the anchor itself. A tail of equal values measures no
  slope instead of one at the smallest positive float.
- A size column carrying a missing value is refused rather than forming a bin
  of its own below every real size. An infinite statistic is scored as the
  loudest event there can be rather than dropped. A size larger than any the
  background produced is pooled into the last bin and says so.

### A coherent statistic is a product of amplitudes

- `event_tiles` returns each tile's signed amplitude beside its energy, and
  `tile_coherence` takes the product on the amplitudes. A product of magnitudes
  is positive whatever the data, so it grew with the number of tile pairs that
  met and two long events overlapping by accident scored above two short ones
  describing one transient. The signed product has mean zero under the null,
  and the pair is ranked on its magnitude because both polarities are physical.

### A block's noise is read where the transient is not

- `scale.local_noise_scale` takes the median of the neighbouring blocks'
  scales, and `scale.on_local_scale` re-expresses a block's statistic on it
  exactly, without recomputing a coefficient. A block's own scale is measured
  on the data it holds, signal included, so a transient loud enough to matter
  is divided by a scale it inflated itself. A block whose neighbours say
  nothing keeps the scale it measured itself.

### Also

- `detector_events` carries `n_pixels`, the distinct tiles an event owns.
- `efficiency_at_far` returns the realised rate and the number of accidentals
  the threshold stands on: a requested rate is not the rate a finite background
  can realise.
- `match_injections` takes the argmax over finite statistics alone, so an
  injection whose only candidate carries no statistic is unmatched rather than
  matched to it.

## 1.0.0 --- 2026-08-20

First tagged release.
