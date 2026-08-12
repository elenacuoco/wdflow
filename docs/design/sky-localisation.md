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

The time a map is built on is the assembled event's: level one
(`wdf.analysis.detector_graph`) joins every block the transient touched on
geometry alone, and the event is timed on `gpsPeak`, the centre of the tile
carrying its largest coefficient. That tile is found on the event's own
wavegram, over all of its coefficients rather than one block's.

It is deliberately not `gpsCentroid`. A centroid is a moment of the energy that
survived threshold *in that detector*: two detectors seeing one source at
different projected amplitudes keep different portions of it and place their
centroids differently, and that difference is indistinguishable from geometry
once it enters an arrival-time difference. The peak tile tracks the loudest
instant, which both detectors share. The same reasoning fixes the anchor of the
comparison map in [the level-two note](windows-and-graphs.md); the sky map
inherits it rather than choosing again.

## The uncertainty is declared, not chosen

`localise` takes a spread per detector and never invents one. The spread an
event declares is `tSpread`, the duration of the tile whose centre was taken as
the time: the time assigned is that centre, so the residual uncertainty is the
extent of the tile it was read from. Because the tile's duration follows from
its octave, a high-frequency event declares a tighter time than a
low-frequency one, and the map widens or narrows accordingly without anything
being tuned.

This matters more than the region's shape. A region drawn with a spread chosen
to make it look small is a picture and not a measurement, and the only check
that distinguishes the two is coverage: over many injections of known position,
the true direction must fall inside the region of stated credibility about that
often. A region that is too narrow fails it; one that is too wide passes it
while saying nothing. `credible_area` and `contains` exist so that the check can
be run rather than asserted.

## Which coincidences get a map

Level two (`wdf.analysis.network_graph`) decides which pairs of events could
have come from one source: the two must cover the same stretch of time once one
is allowed to shift by the light travel time plus their spreads, and must
overlap in band. Only what that admits reaches the sky stage, and the constraint
it imposes is the same physics the map then solves — a pair the light travel
time excludes has no sky position to find.

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
