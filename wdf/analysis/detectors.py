"""Where the detectors are, and what that implies for a coincidence.

The largest arrival-time difference a real signal can have between two
detectors is set by how far apart they are. Writing that difference into the
analysis as a number fixes the code to one pair of detectors; the positions are
published constants of the instruments, and the time follows from them.

Coordinates are the vertex positions in the Earth-fixed frame, in metres, as
given in the LIGO/Virgo/KAGRA instrument descriptions.
"""
from __future__ import annotations

import numpy as np

SPEED_OF_LIGHT = 299792458.0

DETECTOR_VERTEX = {
    "H1": (-2161414.93, -3834695.36, 4600350.22),
    "L1": (-74276.05, -5496283.72, 3224257.02),
    "V1": (4546374.10, 842989.70, 4378576.96),
    "K1": (-3777336.02, 3484898.41, 3765313.68),
    "G1": (3856309.94, 666598.96, 5019641.42),
}


def light_travel_time(ifo_a: str, ifo_b: str) -> float:
    """The largest arrival-time difference a signal can have between two sites.

    :type ifo_a: str
    :param ifo_a: one detector's name, as `DETECTOR_VERTEX` keys it.
    :type ifo_b: str
    :param ifo_b: the other detector's name.
    :return: float -- seconds.
    :raises KeyError: if either detector's position is unknown.
    """
    a = np.asarray(DETECTOR_VERTEX[str(ifo_a).upper()], dtype=float)
    b = np.asarray(DETECTOR_VERTEX[str(ifo_b).upper()], dtype=float)
    return float(np.linalg.norm(a - b) / SPEED_OF_LIGHT)


def network_light_travel_time(ifos) -> float:
    """The largest light travel time within a set of detectors.

    :param ifos: iterable of detector names.
    :return: float -- seconds; zero for fewer than two known detectors.
    """
    known = [name for name in dict.fromkeys(str(i).upper() for i in ifos)
             if name in DETECTOR_VERTEX]
    if len(known) < 2:
        return 0.0
    return max(light_travel_time(a, b)
               for i, a in enumerate(known) for b in known[i + 1:])
