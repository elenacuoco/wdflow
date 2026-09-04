"""What an arrival-time difference is worth, measured on the two populations.

A pair is admitted on the stretches of time its two events cover, because a
transient longer than one analysis window is assembled as several events and
the two detectors do not keep the same one. The difference of the two events'
instants is then not a gate --- it would discard those pairs --- but it is not
nothing either: a pair that used a tenth of the tolerance it was allowed is
more likely to have come from one source than one that used all of it.

How much more is a measurement and not a choice of shape. Under a signal the
difference follows the projected baseline smeared by each event's timing
error; under an accidental it is flat across whatever window the pair was
allowed. Both distributions are produced by the analysis already: the slid
background is the accidental one and the injections some candidate matched are
the signal one. This module reads them and returns the log ratio.

The difference is taken relative to each pair's own tolerance, so one
distribution serves pairs whose events declare different timing spreads: a
long pair allowed a wide window and a short one allowed a narrow window are
compared on the fraction of it they consumed.
"""
from __future__ import annotations

import numpy as np

#: How many bins the two densities are estimated on, over the full range of
#: the fraction of tolerance a pair may consume. The number is a resolution
#: and not a tuning: a pair's timing error is a continuum, and the bins only
#: have to be finer than the feature they must resolve --- the peak the signal
#: population makes at zero --- and coarser than the count supports.
BINS = 41

#: The smallest density either population is credited with, as a fraction of a
#: uniform one. A bin no injection landed in is a bin the measurement did not
#: reach, not a bin a signal cannot occupy, and a zero there would send the
#: ratio to minus infinity on the evidence of one absent count.
FLOOR = 1e-3


class TimingPrior:
    """The log ratio of the signal and accidental densities of `dt/tolerance`.

    :ivar edges: the bin edges the two densities were estimated on.
    :ivar log_ratio: one value per bin, the log of the signal density over the
        accidental one.
    """

    def __init__(self, edges: np.ndarray, log_ratio: np.ndarray):
        self.edges = np.asarray(edges, dtype=float)
        self.log_ratio = np.asarray(log_ratio, dtype=float)
        #: The range the densities were estimated on, in units of whatever the
        #: caller normalised by. A value outside it is scored by the nearest
        #: bin, which is the edge of the window a pair could be admitted in.
        self.span = float(max(abs(self.edges[0]), abs(self.edges[-1])))

    @classmethod
    def fit(cls, accidental, signal, bins: int = BINS, floor: float = FLOOR,
            span: float = 1.0):
        """Measure the two densities and take their log ratio.

        :param accidental: `dt/tolerance` of the slid population, the
            distribution under the hypothesis that the pair is a coincidence
            of unrelated noise.
        :param signal: `dt/tolerance` of the candidates an injection matched,
            the distribution under the hypothesis that one source produced the
            pair.
        :type bins: int
        :param bins: how many bins to estimate on.
        :type floor: float
        :param floor: the smallest density either population is credited with,
            as a fraction of a uniform one.
        :type span: float
        :param span: half-range the densities are estimated on, in the units
            the caller normalised the difference by. One is right where the
            quantity is a fraction of a tolerance a pair cannot exceed; a
            quantity that can exceed its own scale --- the lag of two events
            admitted on their extents, in units of the light travel time ---
            needs the range its own population occupies, or every pair beyond
            it lands in one bin together with the accidentals just outside.
        :return: TimingPrior
        :raises ValueError: if either population is empty, since a ratio of
            two densities needs both of them.
        """
        acc = np.asarray(accidental, dtype=float)
        sig = np.asarray(signal, dtype=float)
        acc = acc[np.isfinite(acc)]
        sig = sig[np.isfinite(sig)]
        if not acc.size or not sig.size:
            raise ValueError(
                f"the ratio needs both populations, and one is empty: "
                f"{acc.size} accidental, {sig.size} signal")

        # Fixed rather than taken from the data: two runs then estimate on the
        # same bins and their priors compare.
        edges = np.linspace(-float(span), float(span), int(bins) + 1)
        width = edges[1] - edges[0]

        def density(values):
            counts, _ = np.histogram(np.clip(values, -span, span), bins=edges)
            total = counts.sum()
            if not total:
                return np.full(len(edges) - 1, 1.0 / (len(edges) - 1) / width)
            d = counts / (total * width)
            # A uniform density over the range is 1/(2 span); the floor is
            # that, scaled, so a bin nothing landed in still admits a finite
            # ratio.
            return np.maximum(d, floor * 0.5 / float(span))

        return cls(edges, np.log(density(sig) / density(acc)))

    def score(self, fraction) -> np.ndarray:
        """What each pair's arrival-time difference is worth, in log units.

        :param fraction: `dt/tolerance` per pair. A value outside the range the
            prior was estimated on is scored by the nearest bin, which is the
            edge of the window a pair could have been admitted in.
        :return: numpy.ndarray -- the log ratio per pair, zero where the
            fraction is not a number.
        """
        u = np.asarray(fraction, dtype=float)
        place = np.clip(np.searchsorted(self.edges,
                                        np.clip(u, -self.span, self.span),
                                        side="right") - 1,
                        0, len(self.log_ratio) - 1)
        out = self.log_ratio[place]
        return np.where(np.isfinite(u), out, 0.0)
