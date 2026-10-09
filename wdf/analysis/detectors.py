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

# The speed of light and the detectors' vertices are constants of nature and of
# the instruments (`wdf.config.constants`); they are re-exported here.
from wdf.config.constants import DETECTOR_VERTEX, SPEED_OF_LIGHT  # noqa: F401


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
