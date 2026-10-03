"""The run configuration, read from JSON and shared by every stage.

One object carries what the search is told: which channel of which frames over
which segments, the conditioning parameters, the analysis window and its
overlap, and where the output goes. Stages read it rather than being given
arguments one at a time, so a run is described by one file and reproduced from
it.

Nothing here decides anything. A value that is not in the configuration is not
a parameter of the search, and a parameter of the search is not written
anywhere else.
"""
__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"
__status__ = "Development"
__project__ = "wdf"

import json
import copy
import pprint 
pp = pprint.PrettyPrinter(indent=4)

# class to handle the parameters
class Parameters(object):
    """
    This class stores set of parameters required by WDF to work
    """

    def __init__(self, **kwargs):
        """
        The constructor
        """

        self.__dict__ = kwargs

        def __getattr__(self, attr):
            return self.__dict__[attr]

    def dump(self, filename):
        """

        :param filename: name of file where saving the parameters
        :type filename: basestring
        """
        self.filename = filename
        with open(self.filename, mode="w", encoding="utf-8") as f:
            json.dump(self.__dict__, f)

    def print(self):
        
        """
        print parameters
        """  
        
        pp.pprint(self.__dict__)   

    def load(self, filename):
        """

                :param filename: name of file where loading the parameters
                :type filename: basestring
                """
        self.filename = filename
        with open(self.filename) as data_file:
            data = json.load(data_file)
        self.__dict__ = data
        return self.__dict__

    def copy(self, param):
        """
                :param param: parameters

              """
        self.__dict__ = copy.deepcopy(param.__dict__)
        return self.__dict__


def window_schedule(par):
    """The analysis windows a run searches at, as (window, overlap) pairs.

    A run searches at one window length. The schedule accepts several, each
    with its own overlap, because the length is a parameter and not a property
    of the search; searching at more than one repeats the same tiling on a
    shifted grid of blocks, and pays a trials factor for it. Each of `window`
    and `overlap` is either a single value, applying to every window length, or
    a list.

    :param par: run configuration carrying `window` and `overlap`, in samples.
    :return: list[tuple[int, int]] -- (window, overlap), in the order given.
    :raises ValueError: if `window` and `overlap` are lists of different
        lengths, or if an overlap does not fit inside its window.
    """
    windows = par.window if isinstance(par.window, (list, tuple)) else [par.window]
    overlaps = (par.overlap if isinstance(par.overlap, (list, tuple))
                else [par.overlap] * len(windows))
    if len(overlaps) != len(windows):
        raise ValueError(
            f"{len(windows)} window lengths but {len(overlaps)} overlaps; give one "
            "overlap per window length, or a single overlap for all of them"
        )
    schedule = [(int(w), int(o)) for w, o in zip(windows, overlaps)]
    for window, overlap in schedule:
        if not 0 <= overlap < window:
            raise ValueError(
                f"overlap {overlap} does not fit in a window of {window} samples; "
                "the search advances by window - overlap and would not move"
            )
    return schedule


#: The high-pass corner and the lowest frequency of the conditioning, Hz, when
#: a run does not give `LowFrequencyCut`. It is the one default every stage
#: reads, through `low_frequency_cut`.
DEFAULT_LOW_FREQUENCY_CUT_HZ = 4.0


def low_frequency_cut(par):
    """The run's `LowFrequencyCut`, Hz, or `DEFAULT_LOW_FREQUENCY_CUT_HZ`.

    Every stage that needs the low edge of the conditioning reads it here, so
    that a run without one gets the same value everywhere.

    :param par: run configuration.
    :return: float -- the low-frequency cut, Hz.
    """
    value = getattr(par, "LowFrequencyCut", None)
    return DEFAULT_LOW_FREQUENCY_CUT_HZ if value is None else float(value)


#: The rules for the coefficients of a window `ThresholdRule` may name, as
#: p4TSA's `WaveletThreshold` spells them.
THRESHOLD_RULES = ("block", "dohonojohnston", "cuoco")
#: The rule of a run that does not name one.
DEFAULT_THRESHOLD_RULE = "block"


def threshold_rule(par):
    """The run's `ThresholdRule` as a `py4tsa.tsa.WaveletThreshold` value.

    :param par: run configuration; `ThresholdRule` is one of
        `THRESHOLD_RULES`, `DEFAULT_THRESHOLD_RULE` when unset.
    :return: py4tsa.tsa.WaveletThreshold.WaveletThresholding
    :raises ValueError: on an unknown name.
    """
    from py4tsa.tsa import WaveletThreshold

    name = str(getattr(par, "ThresholdRule", None) or DEFAULT_THRESHOLD_RULE).strip().lower()
    if name not in THRESHOLD_RULES:
        raise ValueError(f"ThresholdRule is {name!r}; expected one of {THRESHOLD_RULES}")
    return getattr(WaveletThreshold, name)


#: The table of a configuration file whose sub-tables override the run's
#: values for one detector.
DETECTOR_TABLE = "detectors"


def load_config(path, ifo=None, **overrides):
    """A run configuration from a TOML file, for one detector.

    The file's top-level keys are the run's parameters; a ``[detectors.<IFO>]``
    table overrides them for that detector (its AR order, its channel, ...).
    JSON files, the format `Parameters.load` reads, are accepted as well and
    have no detector tables. Keyword arguments override both. The file and the
    detector are recorded as `ConfigFile` and `itf`.

    :type path: str
    :param path: a ``.toml`` or ``.json`` file.
    :type ifo: str or None
    :param ifo: the detector whose table applies, e.g. "H1"; None applies none.
    :param overrides: values that replace the file's.
    :return: Parameters
    :raises KeyError: if `ifo` is given and the file has no table for it while
        it has detector tables.
    """
    import os
    import tomllib

    if str(path).endswith(".json"):
        with open(path) as fh:
            data = json.load(fh)
    else:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    tables = data.pop(DETECTOR_TABLE, {})
    if ifo is not None:
        if tables and ifo not in tables:
            raise KeyError(f"{path} has no [{DETECTOR_TABLE}.{ifo}] table; it has {sorted(tables)}")
        data.update(tables.get(ifo, {}))
        data["itf"] = ifo
    data.update(overrides)
    data["ConfigFile"] = os.path.abspath(str(path))
    return Parameters(**data)
