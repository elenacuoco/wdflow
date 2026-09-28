# The benchmark: one set every new idea is compared on

A change to the search --- a wavelet basis, a thresholding rule, a window, a
clustering --- is judged here on one data set whose truth is known entirely:
three detectors of white Gaussian noise with low signal-to-noise compact
binaries injected at recorded times, and the same noise without them. This page
says what the set is, why it is built this way, and how it is checked.
`wdf.mock.benchmark` writes it; everything it contains is produced by
`wdf.mock.generate_dataset`.

## Why white noise

On coloured or recorded data a change to the search is seen through the
conditioning. A basis that looks better may be one the whitening happened to
suit, and a threshold that looks worse may be one a residual line defeated. In
unit-variance white noise at the analysis rate there is nothing to condition, so
the difference between two configurations is a difference in the search.

The worker still runs its chain on the set: it band-passes the stream and fits
an autoregressive whitening filter to it. On white noise that filter is the
identity to within its estimation error, which is the same for every
configuration compared, and the band-pass removes the share of each injection's
amplitude that lies outside it --- recorded in the set's `validation.json`.

## What it holds

- **Noise.** `wdf.mock.noise.white_noise`, one seed per detector, 2048 Hz, in
  H1, L1 and V1. Foreground and background are the same realisation, so their
  difference is exactly the injections.
- **Signals.** IMRPhenomD binaries drawn by chirp mass, uniform in its logarithm
  over 5--30 solar masses in the detector frame, mass ratio 1--4, aligned spins
  within 0.5. Network SNR from 6 to 20, with 60 per cent of draws inside 7--12
  and the rest uniform over the whole range (`snr_core`).
- **Geometry.** Every source is placed on the sky and projected
  (`project_cbc`): each detector has its own antenna response and its own
  arrival time.
- **Virgo.** Receives 0.32 of the amplitude a LIGO detector with the same
  antenna response would (`relative_sensitivity`). In whitened data a detector
  whose strain noise is higher by `1/r` sees the same waveform times `r`, and
  that is all the difference is.
- **Truth.** `injections.parquet` --- per-detector GPS and support, optimal SNR
  per detector and for the network, masses, chirp mass, mass ratio, sky, the
  band the track spans --- and `tracks.parquet`, each injection's
  time-frequency track (`wdf.mock.waveforms.cbc_track`).

## How it is checked

On the frames as written, read back the way a search reads them
(`wdf.mock.validation`):

- The spectrum at 1/16 Hz, octave by octave, against the flat density, in units
  of the scatter an average of `K` segments allows. Not the variance, which a
  spectrum can have right while being wrong everywhere.
- Gaussianity band by band: excess kurtosis and the share of samples beyond
  three deviations, against what a Gaussian of `2 B T` independent samples
  allows. Skewness is not used: inside an octave it is zero whatever the noise.
- Every injection in every detector filtered with its true template, rebuilt
  from the truth table. The injection alone (foreground less background) must
  return the recorded SNR to the precision of the arithmetic, at zero lag. In
  noise, the recovered value is the recorded one plus a standard normal: at
  SNR 8 the noise alone moves it by an eighth, so a per-injection agreement of a
  few per cent is a test of the first kind and not of the second.
- The worker's own reader (`FrameIChannel`) returns the samples gwpy does.

## The unmodelled principle

The set holds one morphology because its question is recovery near threshold.
A configuration that wins on it by preferring rising frequency has learnt the
population and not the noise; that is not a better unmodelled search.
