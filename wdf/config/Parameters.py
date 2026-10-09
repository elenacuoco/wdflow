"""The run configuration, read from a TOML (or JSON) file and shared by every
stage.

One object carries what the search is told: which channel of which frames over
which segments, the conditioning parameters, the analysis window and its
overlap, and where the output goes. Stages read it rather than being given
arguments one at a time, so a run is described by one file and reproduced from
it.

Nothing here decides anything. A key the configuration does not set takes its
default from wdf/config/defaults.toml (`wdf.config.defaults`), and no value of
the search is written anywhere else; `REQUIRED_KEYS` have no default.
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

from wdf.config.defaults import defaults, parameter

_RUN = defaults("run")
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
#: a run does not give `LowFrequencyCut` (`[run] LowFrequencyCut` of
#: wdf/config/defaults.toml). It is the one default every stage reads, through
#: `low_frequency_cut`.
DEFAULT_LOW_FREQUENCY_CUT_HZ = _RUN["LowFrequencyCut"]


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
#: The rule of a run that does not name one (`[run] ThresholdRule`).
DEFAULT_THRESHOLD_RULE = _RUN["ThresholdRule"]


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


def block_length_of(window):
    """The block rule's default L for a window, ``round(ln window)``, at least 1,
    as p4TSA's `WaveletThreshold::BlockLengthOf` computes it.

    :param window: samples.
    :return: int
    """
    import math

    return max(1, int(math.floor(math.log(int(window)) + 0.5)))


def check_bases(par, schedule):
    """Refuse, before any data is read, a `Bases` list the block rule cannot run.

    Under the block rule a band of a packet or local cosine basis must hold at
    least one block: ``window / M >= L`` for ``LocalCos<M>`` (``window >> depth
    >= L`` for a packet level), L being `BlockLength` or, at 0, ``round(ln
    window)``. p4TSA's `WDF2Classify.SetBases` refuses the same bases when the
    search is built; this check runs when the worker is built and names the
    parameters. Each window of the schedule is checked with its own L.

    :param par: run configuration: `Bases` (or `bases`), `ThresholdRule`,
        `BlockLength`.
    :param schedule: `window_schedule(par)`.
    :return: None
    :raises ValueError: naming the basis, `Bases`, the window, L and the
        largest M allowed there.
    """
    from wdf.analysis.wavelets import packet_depth

    names = getattr(par, "Bases", None) or getattr(par, "bases", None)
    rule = str(getattr(par, "ThresholdRule", None) or DEFAULT_THRESHOLD_RULE).strip().lower()
    if not names or rule != "block":
        return
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    given = int(getattr(par, "BlockLength", None) or _RUN["BlockLength"])
    for window, _ in schedule:
        length = given if given > 0 else block_length_of(window)
        for name in names:
            depth = packet_depth(str(name))
            if depth > 0 and (int(window) >> depth) < length:
                largest = int(window) // length
                largest = 1 << (largest.bit_length() - 1) if largest >= 1 else 0
                raise ValueError(
                    f"Bases holds {name!r}, whose bands are shorter than one block of the block "
                    f"rule at window {window} (BlockLength L {length}"
                    f"{'' if given > 0 else ' = round(ln window)'}): a basis of segment M needs "
                    f"window / M >= L, so at most M = {largest} here (LocalCos{largest}). "
                    f"Remove it from Bases, lower its M, or lengthen the window.")


#: The table of a configuration file whose sub-tables override the run's
#: values for one detector.
DETECTOR_TABLE = "detectors"
#: The keys a run must give, which have no default: the data (`file`,
#: `channel`, `sampling`, `gps`), the output (`outdir`, `run`), the read block
#: `len`, the noise model (`ARorder`, `learn`, `preWhite` offline) and the
#: search (`window`, `overlap`, `threshold`). Every other key has its default in
#: wdf/config/defaults.toml.
REQUIRED_KEYS = ("file", "channel", "sampling", "gps", "outdir", "run", "len", "ARorder", "learn",
                 "preWhite", "window", "overlap", "threshold")


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

    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

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


#: Stage 2 of the network workflow (`Reconstruction`): "wavegram", the
#: paper's detector graph and event wavegrams (`wdf.analysis.detector_graph`);
#: "track", tracks seeded by the triggers through the full cell map and the
#: waveform along them (`wdf.analysis.track_reconstruction`).
RECONSTRUCTIONS = ("wavegram", "track")
#: Stage 3 (`Coincidence`): "wavegram", `WavegramCoincidenceFinder`; "track",
#: `TrackCoincidenceFinder`.
COINCIDENCES = ("wavegram", "track")
#: Stage 4 (`Scorer`): "classic", a column of the candidate table
#: (`NetworkRankColumn`); "gnn", `GNNCoincidenceScorer` loaded from `GNNModel`.
SCORERS = ("classic", "gnn")
#: The default of every stage (`[run] Reconstruction`, `Coincidence`, `Scorer`).
DEFAULT_RECONSTRUCTION, DEFAULT_COINCIDENCE, DEFAULT_SCORER = (
    _RUN["Reconstruction"], _RUN["Coincidence"], _RUN["Scorer"])


def _choice(par, key, allowed, default):
    value = str(getattr(par, key, None) or default).strip().lower()
    if value not in allowed:
        raise ValueError(f"{key} {value!r}: use one of {allowed}")
    return value


def reconstruction_mode(par):
    """The run's `Reconstruction`, one of `RECONSTRUCTIONS`, default
    `[run] Reconstruction`.

    :raises ValueError: on an unknown value.
    """
    return _choice(par, "Reconstruction", RECONSTRUCTIONS, DEFAULT_RECONSTRUCTION)


def network_path(par):
    """The run's stages 2, 3 and 4: (`Reconstruction`, `Coincidence`,
    `Scorer`).

    The two paths do not mix: a track reconstruction is compared by
    `TrackCoincidenceFinder` and the event wavegrams by
    `WavegramCoincidenceFinder`. "gnn" needs `GNNModel`, the path of a model
    `GNNCoincidenceScorer.save` wrote.

    :return: tuple of str
    :raises ValueError: on an unknown value, a mixed path, or "gnn" without a
        model.
    """
    reconstruction = reconstruction_mode(par)
    coincidence = _choice(par, "Coincidence", COINCIDENCES, DEFAULT_COINCIDENCE)
    scorer = _choice(par, "Scorer", SCORERS, DEFAULT_SCORER)
    if reconstruction != coincidence:
        raise ValueError(f"Reconstruction {reconstruction!r} with Coincidence {coincidence!r}: "
                         "the wavegram and the track paths do not mix; set both to "
                         "'wavegram' or both to 'track'")
    if scorer == "gnn" and not parameter(par, "GNNModel", "run"):
        raise ValueError("Scorer 'gnn' needs GNNModel, the path of a saved GNNCoincidenceScorer")
    return reconstruction, coincidence, scorer
