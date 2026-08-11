"""The trigger, as the search hands it on.

One analysis block that passed threshold: the coefficients that survived, the
noise scale they were measured against, the basis that won, and the parameters
those coefficients imply. Everything a later stage does --- grouping,
reconstruction, coincidence, ranking --- is a function of what this record
carries, so the record is what has to be complete rather than the stage that
produced it.
"""
__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = ["http://www.giantflyingsaucer.com/"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"
__status__ = "Development"

from dataclasses import dataclass, field

import numpy as np


@dataclass
class eventPE:
    """One trigger: the coefficients that survived, and the parameters they imply.

    :param gps: GPS time of the analysis window's first sample, which the
        coefficient tiles are placed against.
    :param gpsStart: GPS time the transient starts at, the earliest surviving tile.
    :param gpsCentroid: energy centroid of the transient in time.
    :param tSpread: spread of the energy in time about the centroid, seconds.
    :param gpsPeak: centre of the tile carrying the largest coefficient.
    :param duration: extent of the surviving tiles, seconds.
    :param duration90: interval holding the central 90 per cent of the energy,
        seconds; unlike `duration` one marginal tile cannot stretch it.
    :param EnWDF: the search's statistic for this window.
    :param sigma: noise scale the search measured on this window.
    :param snrPeak: largest coefficient on the noise scale.
    :param freqMin: lower edge of the surviving tiles, Hz.
    :param freqMean: energy-weighted frequency of the surviving tiles, Hz.
    :param freqMax: upper edge of the surviving tiles, Hz.
    :param freqQ05: frequency below which 5 per cent of the energy lies, Hz.
    :param freqQ95: frequency below which 95 per cent of the energy lies, Hz.
    :param wave: name of the basis that produced the coefficients.
    :param n_coeff: length of the window's coefficient vector.
    :param fs: sampling frequency the coefficients were computed at, Hz.
    :param index: coefficient indices of the survivors.
    :param value: coefficient values, in the same order as `index`.
    """

    gps: float
    gpsStart: float
    gpsCentroid: float
    tSpread: float
    gpsPeak: float
    duration: float
    duration90: float
    EnWDF: float
    sigma: float
    snrPeak: float
    freqMin: float
    freqMean: float
    freqMax: float
    freqQ05: float
    freqQ95: float
    wave: str
    n_coeff: int
    fs: float
    index: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint16))
    value: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    def record(self) -> dict:
        """The trigger as one row of a trigger file.

        :return: dict -- the fields of `wdf.analysis.coefficients.TRIGGER_SCHEMA`.
        """
        return dict(
            gps=float(self.gps),
            gpsStart=float(self.gpsStart),
            gpsCentroid=float(self.gpsCentroid),
            tSpread=float(self.tSpread),
            gpsPeak=float(self.gpsPeak),
            duration=float(self.duration),
            duration90=float(self.duration90),
            EnWDF=float(self.EnWDF),
            sigma=float(self.sigma),
            snrPeak=float(self.snrPeak),
            freqMin=float(self.freqMin),
            freqMean=float(self.freqMean),
            freqMax=float(self.freqMax),
            freqQ05=float(self.freqQ05),
            freqQ95=float(self.freqQ95),
            wave=str(self.wave),
            n_coeff=int(self.n_coeff),
            fs=float(self.fs),
            wt_index=[int(i) for i in self.index],
            wt_value=[float(v) for v in self.value],
        )
