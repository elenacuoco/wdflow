# Where on the sky, and what the map is made of

`wdf.analysis.skymap` turns a network coincidence into a region of the sky. It
needs two things per detector and nothing else: an arrival time, and the
uncertainty on it. Everything that distinguishes this map from one drawn by any
other pipeline is therefore in where those two numbers come from, and they come
from the same machinery that assembled the event in the first place — the
wavegram, the two levels of graph, and, for which candidates are worth a map at
all, the learned score.

## The time is the event's, not the block's

A block is a computational unit. A transient that lasts longer than one is
analysed in several, and the instant any single block reports is a property of
where the analysis grid happened to start. An arrival time read off a block
would therefore carry the grid into the arrival-time difference, and from there
straight into the sky position, where nothing downstream could separate the two.

The time a map is built on is the assembled event's: the detector stage
(`wdf.analysis.detector_graph`) joins every block the transient touched on
geometry alone, and the event is then timed on its own reconstruction. Every
block it touched is inverted and stitched, and
`wdf.analysis.timing.envelope_instant` takes the peak of that waveform's
analytic envelope. The catalogue carries it as `gpsEnvelope`.

The peak is sought within one block of `gpsPeak`, the centre of the tile
carrying the event's largest coefficient, which is found on the event's own
wavegram rather than one block's. The bound is what keeps the instant on the
feature the event was selected for: an event assembled from many blocks spreads
its energy over its whole extent, and the envelope of a long transient peaks
where that energy happened to concentrate. Where the envelope cannot be read
the tile centre answers, so the column always carries the best instant the
event has.

The tile centre alone is not enough, because a tile lasts one over the upper
edge of its own band. Where an event's largest coefficient sits low in the band
the tile is longer than the light travel time of the network, and a difference
of two such instants then carries no geometry at all.

It is deliberately not `gpsCentroid`. A centroid is a moment of the energy that
survived threshold *in that detector*: two detectors seeing one source at
different projected amplitudes keep different portions of it and place their
centroids differently, and that difference is indistinguishable from geometry
once it enters an arrival-time difference. The envelope peak and the peak tile
both track the loudest instant, which both detectors share. The same reasoning
fixes the anchor of the comparison map in
[the network-stage note](windows-and-graphs.md); the sky map inherits it rather
than choosing again.

## Two arrival-time differences, and which one a map uses

`localise` is given a difference of arrival times, and the library measures that
difference in two ways for two purposes.

On a network edge it is the difference of the two events' own instants. That is
a difference of two node quantities: a time slide moves an event's times and
carries its instant with them, so nothing is measured per pair, on the
unshifted population or on a background of accidentals. What an edge needs is
whether the pair is causally possible and how much of its timing tolerance it
consumed, and the event instant resolves that.

A sky region needs the finer one, and it is a property of the pair rather than
of either event. `wdf.analysis.timing.arrival_time_difference` takes the lag
that maximises the cross-correlation of the two stitched reconstructions, over
the lags the network's geometry allows, and returns with it the width the
correlation peak declares. It measures the difference on the morphology the two
detectors share, at a cost of one correlation per pair, so it is applied to the
candidates a map is drawn for and not to the graph.

## The uncertainty is declared, not chosen

`localise` takes a spread per detector and never invents one. The spread an
event declares is `tSpread`, the spread of its energy in time about its
centroid, with each tile's own width folded in as the variance of energy
distributed uniformly across it. Over an event assembled from several blocks it
is the members' spreads combined about the event's centroid.

It is a property of how far the event's energy reaches in time, so an event
confined to one octave and a few tiles declares a tighter time than one smeared
across octaves and blocks, and the map widens or narrows accordingly without
anything being tuned.

This matters more than the region's shape. A region drawn with a spread chosen
to make it look small is a picture and not a measurement, and the only check
that distinguishes the two is coverage: over many injections of known position,
the true direction must fall inside the region of stated credibility about that
often. A region that is too narrow fails it; one that is too wide passes it
while saying nothing. `credible_area` and `contains` exist so that the check can
be run rather than asserted.

## Which coincidences get a map

The network stage (`wdf.analysis.network_graph`) decides which pairs of events could
have come from one source: the difference of the two events' own instants
must not exceed the light travel time, widened by what each declares its
instant is worth.

The learned score on the network graph ranks those pairs; it does not change
their geometry. A graph neural network can order candidates better than a
threshold on one statistic, and so decides which are worth following, but the
region a coincidence occupies on the sky is fixed by its arrival times and
spreads whatever score it carries. Keeping the two apart is what allows the
ranking to be replaced without the localisation changing.

## What two detectors can and cannot say

A network of `n` detectors measures `n - 1` independent arrival-time
differences, and each fixes the angle between the source and the baseline
joining a pair of sites. Two detectors therefore give a ring and not a point:
one number cannot say more than an angle. A third crosses that ring with another
and leaves a pair of patches reflected through the plane of the network; each
further site cuts further. `localise` reports the whole weighted grid rather
than a best point, because for two detectors there is no best point to report
and quoting one would be inventing information the geometry does not contain.

Only differences enter, so a timing error common to every detector moves
nothing. One that differs between them goes straight into the arrival-time
difference and hence into the position, which is why absolute time is
established once, from what each reader actually returned, and never assumed.
