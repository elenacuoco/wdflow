"""Turns a WDF event into a trigger record.

The search hands over the statistic, the winning basis, the local noise scale
and the window's coefficients. The coefficients that survived thresholding are
the record; the event's extent, band and time are moments of the energy their
tiles carry, so nothing here inverts a transform or takes a spectrum.
"""
import logging

import numpy as np

from wdf.analysis.coefficients import from_dense
from wdf.analysis.metaparameters import meta_features
from wdf.observers.observable import Observable
from wdf.observers.observer import Observer
from wdf.structures.eventPE import eventPE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterEstimation(Observer, Observable):
    """Derives each trigger's parameters from the coefficients WDF found."""

    def __init__(self, parameters):
        """
        :type parameters: wdf.config.Parameters.Parameters
        :param parameters: run configuration, read for `Ncoeff` and the
            analysis sampling frequency.
        """
        Observable.__init__(self)
        Observer.__init__(self)

        if parameters.ResamplingFactor is not None:
            self.sampling = parameters.sampling / parameters.ResamplingFactor
        else:
            self.sampling = parameters.sampling
        self.Ncoeff = parameters.Ncoeff

    def update(self, event):
        """Builds one trigger record and passes it to the registered observers.

        :type event: pytsa.tsa.EventFullFeatured
        :param event: the window WDF triggered on.
        :return: None
        """
        coefficients = np.zeros(self.Ncoeff)
        for i in range(self.Ncoeff):
            coefficients[i] = event.GetCoeff(i)

        index, value = from_dense(coefficients)
        gps = float(event.mTime)
        sigma = float(event.mSigma)
        features = meta_features(index, value, self.Ncoeff, self.sampling,
                                 sigma, gps=gps)

        self.update_observers(eventPE(
            gps=gps,
            EnWDF=float(event.mSNR),
            sigma=sigma,
            wave=str(event.mWave),
            n_coeff=int(self.Ncoeff),
            fs=float(self.sampling),
            index=index,
            value=value,
            **features,
        ))
