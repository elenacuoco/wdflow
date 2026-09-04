"""The pair's deterministic ranking: coherent energy, discounted by its lag.

The network stage measures two things about a candidate pair and they are not
in the same units. The first is how much coherent energy the two detectors
carry over the tiles they share, in units of the noise scale squared. The
second is how far apart the structure they share arrived, in seconds, which is
evidence only through how much likelier that lag is under a signal than under
an accidental coincidence.

They are put on one scale by measuring the second rather than weighting it: the
log ratio of the two densities of the lag is a log likelihood ratio in nats,
and the coherent energy is twice a log likelihood ratio in Gaussian noise on
whitened coefficients, so the conversion is a factor of two and not a number
chosen to make a table look better. What is a choice is whether to use the
timing at all, and that is read at a stated false-alarm rate against the
background of each ranking separately: a ranking that adds a term changes its
own background and earns no credit for candidates it merely reorders.

The lag this reads is the one measured on the tiles the pair shares, not the
difference of the two events' instants. A transient longer than one analysis
window is assembled as several events and the two detectors do not keep the
same one, so the difference of their instants is a statement about which
fragment each of them kept; the shared tiles are the piece of the plane both
of them kept, and their lag is a statement about the source.

Nothing here selects: it produces a column, and the threshold on that column is
read where every threshold in this analysis is read, on the population the
slides produced over the livetime they produced it in.
"""
from __future__ import annotations

import numpy as np

from wdf.analysis.timing_prior import BINS, FLOOR, TimingPrior

#: How many light travel times the lag is estimated over. A pair is admitted on
#: its events' extents, so the shared tiles may sit up to the travel time apart
#: plus the duration of the tiles themselves, which low in the band is tens of
#: milliseconds. The range has to cover what the population occupies, or every
#: pair beyond it is scored in one bin together with the accidentals just
#: outside; it is a range and not a tolerance, and nothing is refused for
#: falling outside it.
LAG_SPAN_TRAVEL_TIMES = 5.0

#: What one nat of timing evidence is worth in units of coherent energy. The
#: coherent energy is twice a log likelihood ratio for whitened coefficients in
#: Gaussian noise, and the timing term is a log likelihood ratio, so two is the
#: conversion between the two conventions. Zero recovers the coherent energy
#: alone, which is the ranking this reduces to when the timing says nothing.
TIMING_WEIGHT = 2.0


class CoherentRanking:
    """Coherent energy over the shared tiles, discounted by their lag.

    :ivar prior: the measured log ratio of the two densities of the lag.
    :ivar travel_time_s: the pair's light travel time, seconds, which the lag
        is expressed in units of.
    :ivar timing_weight: what one nat of timing evidence is worth in units of
        coherent energy.
    :ivar amplitude: column carrying the coherent amplitude, the square root of
        the magnitude of the coherent energy, in units of the noise scale.
    :ivar fraction: column carrying the signed fraction of the shared energy
        that is coherent.
    :ivar lag: column carrying the lag of the shared tiles, seconds.
    """

    def __init__(self, prior: TimingPrior, travel_time_s: float,
                 timing_weight: float = TIMING_WEIGHT,
                 amplitude: str = "network_morphology",
                 fraction: str = "block_coherent_fraction",
                 lag: str = "block_coherent_dt"):
        self.prior = prior
        self.travel_time_s = float(travel_time_s)
        self.timing_weight = float(timing_weight)
        self.amplitude = amplitude
        self.fraction = fraction
        self.lag = lag

    @classmethod
    def fit(cls, accidental, signal, travel_time_s: float,
            timing_weight: float = TIMING_WEIGHT,
            span: float = LAG_SPAN_TRAVEL_TIMES, bins: int = BINS,
            floor: float = FLOOR, amplitude: str = "network_morphology",
            fraction: str = "block_coherent_fraction",
            lag: str = "block_coherent_dt"):
        """Measure the two lag densities and build the ranking from them.

        The two populations must be the ones the ranking will be read on and
        must not be the ones its efficiency is then quoted on: a prior fitted
        on the candidates it is later scored against reports its own memory.
        The analysis already has the split it needs --- one set fits, the
        held-out set and the recorded strain are scored.

        :param accidental: the slid pairs, one row each, carrying `lag` and
            `fraction`. The distribution under the hypothesis that the pair is
            a coincidence of unrelated noise.
        :param signal: the zero-lag pairs an injection matched, carrying the
            same columns. The distribution under the hypothesis that one source
            produced the pair.
        :type travel_time_s: float
        :param travel_time_s: the pair's light travel time, seconds, strictly
            positive. It is the physical scale the lag is expressed in, so a
            prior fitted for one baseline is not one for another.
        :type timing_weight: float
        :param timing_weight: nats to coherent-energy units; zero recovers the
            coherent energy alone.
        :type span: float
        :param span: half-range of the estimate, in light travel times.
        :type bins: int
        :param bins: how many bins the densities are estimated on.
        :type floor: float
        :param floor: smallest density either population is credited with, as a
            fraction of a uniform one.
        :param amplitude: column carrying the coherent amplitude.
        :param fraction: column carrying the signed coherent fraction.
        :param lag: column carrying the lag of the shared tiles, seconds.
        :return: CoherentRanking
        :raises ValueError: if the travel time is not positive, if a column is
            missing from either population, or if either population carries no
            pair with a measured lag.
        """
        if not travel_time_s > 0.0:
            raise ValueError(
                f"the lag is expressed in light travel times and this pair's "
                f"is {travel_time_s!r}")
        built = cls(TimingPrior(np.array([-1.0, 1.0]), np.zeros(1)),
                    travel_time_s, timing_weight, amplitude, fraction, lag)
        prior = TimingPrior.fit(
            built._measured_fraction(accidental, "accidental"),
            built._measured_fraction(signal, "signal"),
            bins=bins, floor=floor, span=span)
        built.prior = prior
        return built

    def _measured_fraction(self, table, label):
        """The lag of the pairs that measured one, in light travel times.

        :param table: the candidate pairs.
        :param label: what the population is called, for the failure message.
        :return: numpy.ndarray -- the lag of the pairs sharing coherent energy.
        :raises ValueError: if a column is missing or nothing was measured.
        """
        for column in (self.lag, self.fraction):
            if column not in table:
                raise ValueError(
                    f"the {label} pairs carry no {column!r}, so the ranking "
                    f"cannot be read on them; rebuild the network graph")
        u = self.lag_fraction(table)
        u = u[np.isfinite(u)]
        if not u.size:
            raise ValueError(
                f"no {label} pair shares a tile, so no lag was measured on "
                f"that population")
        return u

    def lag_fraction(self, table) -> np.ndarray:
        """The shared tiles' lag, in units of the light travel time.

        :param table: the candidate pairs.
        :return: numpy.ndarray -- one value per pair, not a number where the
            pair shares no coherent energy and so measured no lag.
        """
        lag = table[self.lag].to_numpy(dtype=float)
        measured = table[self.fraction].to_numpy(dtype=float) != 0.0
        return np.where(measured, lag / self.travel_time_s, np.nan)

    def score(self, table) -> np.ndarray:
        """The ranking of each pair, in units of the noise scale squared.

        A pair that shares no tile carries no coherent energy and no lag: it is
        scored at zero energy and at the most penalised value the prior
        measured, so it ranks below every pair that measured either. That is a
        statement about evidence and not a cut --- the pair stays in the
        population and in the background it contributes to.

        :param table: the candidate pairs, carrying the amplitude, the fraction
            and the lag columns this ranking was built for.
        :return: numpy.ndarray -- the ranking, higher meaning more signal-like.
        :raises ValueError: if a column is missing.
        """
        for column in (self.amplitude, self.lag, self.fraction):
            if column not in table:
                raise ValueError(
                    f"the pairs carry no {column!r}, so this ranking cannot be "
                    f"read on them; rebuild the network graph")
        # The amplitude is the root of the magnitude of the coherent energy, so
        # the energy is its square: the ranking is in energy units because that
        # is what a log likelihood ratio adds to.
        energy = table[self.amplitude].to_numpy(dtype=float) ** 2
        u = self.lag_fraction(table)
        timing = np.where(np.isfinite(u), self.prior.score(u),
                          float(self.prior.log_ratio.min()))
        return energy + self.timing_weight * timing

    def attach(self, table, column: str = "network_coherent_timed"):
        """Write the ranking into the table, in place.

        Side effect: `table` gains one single-precision column. A slid
        background is tens of millions of rows and a copy of it is the memory
        wall of the run; single precision resolves a threshold that is read as
        an order statistic, which is the same reason the learned logit is
        carried that way.

        :param table: the candidate pairs, modified in place.
        :type column: str
        :param column: the name to write under.
        :return: pandas.DataFrame -- the same table.
        :raises ValueError: if a column the ranking reads is missing.
        """
        table[column] = self.score(table).astype("float32")
        return table
