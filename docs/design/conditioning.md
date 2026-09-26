# Conditioning: what the search is handed, and how it is checked

Before the search reads a sample, each detector's strain passes through a chain
that restricts it to the analysed band and makes its noise white, flat and
Gaussian, since the threshold means the same thing at every frequency only on
such a stream. This page describes the chain in the order it runs and says why
each step is placed where it is.

    strain
      -> the detector's line notches          (per detector)
      -> the band-pass and the decimation     (the same for every detector)
      -> the zero-phase whitening             (per detector, its own model)
      -> the search

## The lines, cut down to the floor

A spectral line is a feature far narrower than the broadband noise around it:
the power mains and their harmonics, calibration lines, the violin modes of
the suspensions. Each detector has its own, at its own frequencies and heights.

They are removed because of the noise model. The whitening filter is built from
a model of the noise, and a model that has lines to represent spends its order
on them and fits the floor between them worse; a model is only as good as its
fit of that floor. What it should be handed is the floor itself.

`wdf.processes.lines` finds the lines on the stretch the model is fitted on, in
the detector's unconditioned strain. The spectrum is the median of the
periodograms, so a transient inside the stretch does not lift it, and the floor
is its running median over a band much wider than any line, so a line does not
lift its own floor. A line is a run of bins above twice the floor, reported when
its height reaches the configured threshold, and it is searched for across every
frequency whose content can reach the analysed stream: from the band-pass's low
edge up to the frequency that folds onto its high edge when the stream is
decimated. The upper stop band of the band-pass lies inside the analysed band
and the whitening lifts what remains there back up, so a line there, or folded
there, reaches the search.

Each line is cut by a peaking section whose depth is matched to the line's
height: run forward and backward, it brings the line to the floor and no
further. A notch with its zeros on the unit circle would remove the floor under
the line as well, and the hole it leaves is one more feature for the model to
follow; a matched cut leaves the smooth floor the model is meant to fit.

The sections are stacked in front of the band-pass and applied with it, in every
pass of the filter: on the stream, with real data before and after each block,
and on the stretch the model is fitted on, which is read with the same real
context. A narrow notch rings far longer than the band-pass, and the settling
the chain measures, the warm-up at the start of a segment and the context of the
fit stretch all follow it (see [Warm-up](warm-up.md)).

The configuration can name the lines instead (`LineNotches`), in which case they
are notched in every segment as given; otherwise each segment's lines are found
on its own fit stretch, above `LineThreshold` times the floor.

## The band, shared, and the detector's own low cut

Every detector is conditioned by the same band-pass: a Chebyshev type II filter
whose `LowFrequencyCut` and upper edge are the edges of its stop bands, run
forward and backward, then decimated to the analysed rate. Its pass band starts
above `LowFrequencyCut`, where the transition ends.

The search reads the analysed stream in octaves, the bands of its wavelet
levels, and by default a detector is searched from the lower edge of the lowest
octave that lies wholly inside the pass band, where the two passes take no more
than `PASS_BAND_LOSS` of the power (`BandPassDownSampling.search_low_frequency`).
The octave below that one lies partly in the transition; it is still searched,
and no check reads it.

A detector whose noise is not to be searched that low is given its own
`SearchLowFrequency`. The band-pass stays the one every detector shares, and a
high-pass of the same order and attenuation, flat from `SearchLowFrequency` up
(`highpass_stop_edge`), is stacked after it on the stream the search reads. It
is not applied to the stretch the noise model is fitted on, and that is what
makes it a cut rather than a distortion. A model fitted on a stream that
includes the cut has to represent the cut: a model able to do so undoes it,
lifting the stop band back up, and one that is not spends its order on the
cliff and misfits the octave above it. Fitted without the cut, the model is the
one the shared conditioning gives; applied to the cut stream it whitens
everything above the cut exactly as it whitens the uncut stream, since there
the cut's response is one, and below the cut nothing is left for it to lift.

## Gates on the whitened stream

A detector's noise holds transients no noise model describes and no search is
meant to report: instrumental glitches hundreds of noise standard deviations
tall, often at low frequency and not flagged by the detector's data quality.
Left in, one of them fills every wavelet level it touches with coefficients far
above the threshold.

They are gated on the whitened stream, the stream the search reads, and not on
the strain. A gate on the strain is an edge in the data that the band-pass and
the whitening then filter: both ring on it, and what they leave on either side
of the gated stretch is a transient of its own in the whitened stream. On the
whitened stream nothing downstream spreads the gate, and its raised-cosine
taper is the only edge there is.

`wdf.processes.gating` takes the census. At each sample the level is the
largest ratio of the stream to its robust standard deviation (a median absolute
deviation, which the transients do not move) over the broadband stream and
every octave the search reads, so that a glitch confined to one octave is
measured there and not diluted by the others. A transient is a run of samples
above the flag level, runs closer than a quarter of a second joined; its extent
therefore holds its whole excursion in every band. A transient whose peak
reaches `GateThreshold` is gated: its extent is zeroed and `GateTaper` seconds
on each side are tapered. `Gates` in the configuration declares stretches to
gate as well, and declared and found gates are merged.

The census needs the whole segment, since a transient's extent and the scale it
is measured against are known only from the stream around it. The worker
therefore conditions and whitens the segment once before searching it, finds
the gates, and applies them to each whitened block of the search pass by
absolute time; the two passes run the same front end and the same filter, so
the stream the gates were found on is the stream they are applied to.

The threshold is an empirical choice, not a physical constraint: it must stand
above the loudest excursion an astrophysical signal the search is meant to
report can produce in one octave of the whitened stream, and below the
instrumental transients it is meant to remove.

## The check before the search

The threshold means the same thing at every frequency and at every time only on
a stream that is white, Gaussian and stationary, on the scale the search divides
by. The conditioning is meant to give the stream those properties, and it can
fail to, one band at a time: a noise model that misfits an octave, a line left
in, a glitch the gates did not catch, noise that is not Gaussian in one band
whatever the model. A single number over the whole band hides all of them, so
the check is read octave by octave.

`wdf.processes.validation.validate` reads the whitened stream the search is
about to read -- gated, divided by the search's scale, over the whole stretch it
will search -- in every octave from the detector's search low frequency to the
Nyquist frequency:

- **White**: the octave's power, as the median of one-second periodograms,
  within a tenth of the power of unit white noise. The median, so that
  transients do not lift it; they are the third criterion's.
- **Gaussian**: the median over eight-second windows of the kurtosis of the
  octave, within a tenth of what the same estimator gives on Gaussian white
  noise of the same length through the same filter. Comparing with the
  estimator's own value on Gaussian noise rather than with three removes its
  bias on a finite, band-limited window by construction, since the bias is the
  same in both. The band above the band-pass's high edge is read as well: the
  band-pass empties it and the whitening lifts what is left back up, so it is
  where a rounding floor, or any other residue the conditioning leaves, shows.
  Windows a gate touches are left out, since they are not searched there.
- **Clean**: the transients the census flags, each widened by one analysis
  window, and the gates with their tapers, cover at most one percent of the
  stretch.
- **Stationary**: in each third of the stretch every octave's power is within a
  tenth of its power over the whole stretch.

The report also carries the detector's angle-averaged range for a binary
neutron star, read off the spectrum of the strain on its fit stretch. It is
reported and not tested: a detector's sensitivity is not a property of its
conditioning.

`wdfUnitDSWorker.validate(segment)` runs everything the search runs before its
first block -- lines, model, whitened stream, gates -- checks the stream and
returns the report without searching. `segmentProcess` does the same before it
searches and stops with `ConditioningRejected` when a criterion fails, naming
the detector, the band and the criterion; the segment is then neither searched
nor marked done. Either way the report is written beside the segment's triggers
as `conditioning-check.json`. `ValidateConditioning = False` turns the check off.

A stretch of data holds several detectors and, for each, one or more science
segments. `wdf.processes.network_search.check` checks every segment of every
detector before any of them is searched, and stops the stretch with every
detector, band and criterion that failed. The check of a segment is filed in a
directory of its own, and each search of that segment --- one per rule for the
coefficients of a window --- is handed the check's model, lines and gates, the
last as declared stretches with no census of its own: the stream searched is
the stream the check read, whatever rule then reads it.

A science segment that fails its check can be checked again from later starts
(`SearchConfig.trim_step_s`): a grid of starts from the segment's own start, as
long as what is left holds the shortest segment searched and the fit stretch.
A segment searched from a later start keeps its fit stretch, its lines and its
model, and both filters settle within the warm-up and forget where they were
started, so the stream it searches is the whole segment's stream from that
start's warm-up on. `wdfUnitDSWorker.validate_starts` therefore whitens the
segment once and checks, start after start, the tail a segment beginning there
would search, with the census, the gates and every criterion read on that tail
alone: the check the shorter segment would get. The earliest start that passes
is kept, its check is made and filed like any other, and the stretch before it
is neither searched nor counted. The tolerances are the check's own; what
moves is where the segment begins, and the grid is fixed by the segment and the
configuration, so the start kept depends on the data's check and on nothing
else.

`wdfUnitDSWorker.whitened_stretch` rebuilds that stream between two instants
without whitening the whole segment again. The chain is started a whole number
of seconds before the first instant, far enough for both filters to have
settled there, and since the band-pass runs with real data on both sides of
every block and the whitening is a finite filter with a look-ahead of its own
order, what it returns is the segment's stream at those instants, gated and on
the search's scale.

The check does not read the octave below the detector's search low frequency,
which the search still reads where the band-pass has not emptied it. Whether
that octave is to be checked, or cut, is an open decision.
