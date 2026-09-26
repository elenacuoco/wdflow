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
