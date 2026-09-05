# Event candidates: architecture and current coverage

## Detection is single-detector and blind

WDF's own trigger generation (`wdf.processes`/`wdf.observers`, backed by p4TSA's
C++ core) is entirely single-detector: it has no knowledge of any other
detector's data, no sky position, and no external GPS-time hint while
searching. Every trigger carries a detection statistic (`EnWDF`, and the
derived `snrMean`/`snrPeak`), timing (`gps`, `gpsPeak`), and time-frequency
metadata, computed purely from that one detector's own whitened data.

`wdf.analysis.clustering.TriggerClusterer` groups a burst of raw per-window
triggers from one detector into candidate events (`clustered_events`). This
step also stays single-detector -- cross-detector combination is a separate,
later stage, by design: a burst of raw triggers in one detector says nothing
on its own about whether a real astrophysical signal is present, so nothing
about the trigger-generation or per-detector-clustering stages should depend
on what any other detector saw.

## Two independent, comparable cross-detector combination methods

Once each detector has its own clustered event list, wdflow combines them
across detectors via two separate methods, meant to be run side by side and
compared, not to replace one another:

- **Deterministic coincidence** (`wdf.analysis.CoincidenceFinder`, which is
  `robust_events.IndexedCoincidenceFinder`): matches clustered events across
  detectors whose covered stretches of time meet once one may shift by the
  light travel time plus the events' own timing spreads, and whose bands
  overlap. The pairing is one-to-one, by assignment rather than by nearest
  neighbour, so one event cannot enter two candidates.
  `wdf.analysis.BackgroundEstimator` (`robust_events.TimeSlideFAR`) estimates
  the accidental rate by non-physical time shifts, the standard time-slide
  technique, with the rate's denominator the background **livetime** — the
  slides times the span — and not the number of candidates the slides produced.

  Two earlier implementations remain importable for comparison and are not what
  the canonical names give you: `wdf.analysis.LegacyCoincidenceFinder` pairs
  each event with its nearest neighbour without consuming it, so one event can
  appear in several candidates, and `wdf.analysis.significance.BackgroundEstimator`
  divides by the number of background candidates, which is a tail percentile of
  the candidate distribution rather than a rate.

- **Learned combination** (`wdf.analysis.gnn.GNNCoincidenceScorer`): a graph
  neural network over the same per-detector clustered events (nodes),
  intra-detector temporal-neighborhood edges, and cross-detector candidate
  edges, trained to score how likely a cross-detector edge is a real
  coincidence versus accidental. Built on `torch_geometric`, so many
  segments' graphs can be batched into one sparse forward/backward pass
  (`GNNCoincidenceScorer.fit`) rather than one Python-level call per segment.

- **Morphological combination** (`network_morphology` on
  `TriggerGraph.candidate_table`): the coherent amplitude the pair carries,
  formed on the coefficients themselves.
  `wdf.analysis.detector_graph.tile_coherence` multiplies the two events'
  coefficient amplitudes on their noise scales and sums over the tiles whose
  rectangles cover the same place on the plane, keeping the coefficients'
  signs; `network_morphology` is the root of its magnitude, the root because
  the sum is an energy and the column is reported on the noise scale, the
  magnitude because two detectors can respond to one source with opposite
  polarity. It is an inner product taken at the resolution the transform has,
  with no grid imposed in between.

The node's own description and the shape summary an edge carries are formed on
the coefficients' magnitudes. The polarity of a coefficient says where the
source sits with respect to the two detectors and how far apart they saw it,
not what the transient is, so a description that keeps it teaches a model the
antenna pattern rather than the morphology, and a shape compared with it
measures the delay rather than the shape.

This is a deliberately different architecture from a coherent multi-detector
pipeline (e.g. Coherent WaveBurst). The coherent quantity here is formed on the
two detectors' surviving coefficients rather than on a joint likelihood over
the network response: no antenna patterns enter it, and no signal model is
projected onto the data. Where a pair's arrival-time difference is wanted at
better resolution than the tiles give ---  a sky region needs it ---
`wdf.analysis.timing.arrival_time_difference` cross-correlates the two stitched
reconstructions, which is a consistency check between reconstructed waveforms
applied to candidates by a caller that wants one, rather than by a stage to
every admitted pair.

## The simulated set the efficiencies are read on

`wdf.mock.dataset` places each compact binary on the sky and projects it onto
every detector through that detector's own antenna response, so the two strains
differ by amplitude and by arrival time exactly as a real source would make them
differ. `min_detector_snr`, when given, redraws the sky and orientation until
every detector receives at least that signal-to-noise ratio, raising the network
target inside the requested range where the geometry demands it.

The floor exists because a projection can starve one detector entirely. An
injection only one detector receives cannot be recovered in coincidence at any
amplitude, and counting it dilutes every aggregate with a signal no network
could find. With the floor the population is the one the whole network sees,
which is the population an efficiency should be quoted on; without it the
aggregate mixes a statement about the search with a statement about how the sky
was drawn.

Glitches are single-detector by construction and are never projected. They exist
to measure the accidental floor of the coincidence, and a glitch placed in both
detectors would measure something else.

## Status

Timing coincidence + time-slide FAR is the production-track method: fast,
deterministic, no training data required. The GNN scorer is comparatively
early-stage -- real positive examples are limited to catalogued events in
whatever segment(s) are being analyzed, so `fit` is closer to a
proof-of-concept calibration than a large-scale trained model; extending its
training set via synthetic signal injection (e.g. bilby/pycbc) rather than
relying only on catalogued real events is the natural next step, independent
of the `torch_geometric` batching infrastructure already in place for it.

## One noise scale, and an orthonormal candidate list

Every amplitude a trigger carries is expressed on the same noise scale: the
per-window sigma of the basis that won the window, which p4TSA exposes as
`EventFullFeatured.mSigma`. `EnWDF` is normalised by it, and
`ParameterEstimationObserver` reads the same value for `snrMean`/`snrPeak`, so
the statistics are comparable to each other and a single one of them can rank
foreground against background. A sigma estimated once for a run would answer a
different question in every window, since detector noise is not stationary over
one.

The candidate list is orthonormal throughout. Biorthogonal wavelets do not
preserve L2 energy, so a sigma estimator that assumes orthonormal coefficients
misestimates their noise floor and lets them win basis selection on noise
alone; the same identity is what makes the score a matched-filter
signal-to-noise ratio at all, so the list cannot hold a basis that breaks it.

The energy statistic sums the coefficients as `WaveletThreshold` emits them and
does not threshold them a second time. The zeros already carry the significance
decision, and under soft thresholding a survivor has been shrunk by the
threshold, so re-testing it against that same threshold would discard it for
having passed.
