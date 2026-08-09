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

    The same data may be searched at several window lengths at once: a
    transient longer than the window cannot have its extent measured within
    one, and only a longer window reaches the lower frequency bands. Each of
    `window` and `overlap` is either a single value, applying to every window
    length, or a list.

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
