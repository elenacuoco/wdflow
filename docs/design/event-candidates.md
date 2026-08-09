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

This is a deliberately different architecture from a coherent multi-detector
pipeline (e.g. Coherent WaveBurst): there is no combined-likelihood statistic
built directly from multiple detectors' *waveforms* (a ρ_c/η_c-style coherent
statistic, or a network correlation coefficient consistency check between
reconstructed waveforms) -- both of wdflow's cross-detector methods above
work from each detector's already-independent trigger/cluster list, not from
a joint reconstruction. `reconstruct_cluster_waveform`/`Coloring` (waveform
reconstruction back to strain units, per detector) already produce the kind
of per-detector reconstructed waveform data a coherence check would need;
using it for cross-detector waveform consistency (rather than timing/learned
scoring alone) is a natural extension, not yet implemented.

## Status

Timing coincidence + time-slide FAR is the production-track method: fast,
deterministic, no training data required. The GNN scorer is comparatively
early-stage -- real positive examples are limited to catalogued events in
whatever segment(s) are being analyzed, so `fit` is closer to a
proof-of-concept calibration than a large-scale trained model; extending its
training set via synthetic signal injection (e.g. bilby/pycbc) rather than
relying only on catalogued real events is the natural next step, independent
of the `torch_geometric` batching infrastructure already in place for it.

## SNR statistic: two root causes fixed at the p4TSA/wdflow boundary

`EnWDF` (WDF's own internal per-window detection statistic) and
`snrMean`/`snrPeak` (recomputed in `wdf.processes.wavelet_energy` from the
same wavelet coefficients) used to disagree by orders of magnitude on real,
non-stationary data. Two causes, both now fixed:

1. `EnWDF` is normalized by a sigma estimated fresh, per window, from the
   winning wavelet basis's own coefficients -- appropriate for real
   non-stationary detector noise. `snrMean`/`snrPeak` were instead normalized
   by a single sigma estimated once from AR-whitening residuals at the start
   of a run, drifting further out of sync with `EnWDF`'s convention as real
   noise conditions moved away from what that one-time estimate saw. p4TSA
   now exposes the winning basis's own per-window sigma
   (`EventFullFeatured.mSigma`), and `ParameterEstimationObserver` uses it
   for `snrMean`/`snrPeak` instead -- both statistics now share one noise
   convention.
2. p4TSA's basis-selection candidate list included biorthogonal B-spline
   wavelets alongside orthonormal ones (Haar, Daubechies). Biorthogonal
   wavelets don't preserve L2 energy (Parseval's theorem), so a sigma
   estimator that assumes homoscedastic, orthonormal coefficients
   systematically misestimates their noise floor -- letting them win basis
   selection even on pure noise, unrelated to any real signal content. The
   candidate list is now the full orthonormal GSL wavelet family only.

A further, separate finding: `wavelet_energy_snr` re-applies a
Donoho-Johnstone threshold to coefficients p4TSA's C++ engine has already
(soft-)thresholded once -- a survivor shrunk by soft-thresholding often falls
back under the same threshold when re-tested, zeroing a large fraction of
triggers' `snrPeak`/`snrMean` a second time. This is independent of both
causes above (present regardless of which sigma convention is used) and is
not yet fixed; noted here as a known limitation of the current
`snrMean`/`snrPeak` statistic.
