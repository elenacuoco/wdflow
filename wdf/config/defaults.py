"""The analysis defaults of `wdf`, read from one file.

Every value an analysis takes when its caller or its run configuration does
not give one is written in ``wdf/config/defaults.toml``, shipped with the
package. Modules read it through `defaults`, which returns one table of that
file; they hold no analysis value of their own. A run configuration
(`wdf.config.Parameters.load_config`) gives the values of one run, and a key
it sets replaces the default of that key.

The file is organised in tables named after the module, or the stage, that
reads them (``[ridge]``, ``[track.coincidence]``, ``[mock.dataset]``, ...).
A key named as a run-configuration key (CamelCase, e.g. ``RidgeSlot``) is
the default of that key; a lower-case key is the default of a library
argument or setting. Every key carries a trailing comment, ``# [unit]
meaning``, which `docs/generate_configuration.py` turns into the
configuration reference.

Conventions of the file:

- the string ``"<unset>"`` stands for "not set" (Python None): the code then
  derives the value, as the key's comment states;
- arrays are returned as tuples, so that a default is immutable and can be a
  dataclass field default;
- values tied to the sampling rate are written in seconds, hertz, or as a
  fraction of the rate or of the Nyquist frequency, and converted with the
  run's rate where they are used.
"""
import functools
import os
from collections.abc import Mapping

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

#: The shipped defaults file.
DEFAULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defaults.toml")
#: The value that stands for "not set" in the defaults file.
UNSET = "<unset>"


def _convert(value):
    if isinstance(value, str) and value == UNSET:
        return None
    if isinstance(value, list):
        return tuple(_convert(v) for v in value)
    return value


@functools.lru_cache(maxsize=None)
def _tables(path=DEFAULTS_FILE):
    """``{dotted table name: {key: value}}`` of the defaults file."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    out = {}

    def walk(prefix, table):
        values = {k: v for k, v in table.items() if not isinstance(v, dict)}
        if prefix:
            out[prefix] = values
        for key, value in table.items():
            if isinstance(value, dict):
                walk(f"{prefix}.{key}" if prefix else key, value)

    walk("", data)
    return out


class Defaults(Mapping):
    """One table of the defaults file, read-only.

    A key missing from the table raises KeyError naming the table, so that a
    module reading a default the file does not hold fails when it is
    imported, not with a value of its own.
    """

    def __init__(self, section, values):
        self.section = section
        self._values = {k: _convert(v) for k, v in values.items()}

    def __getitem__(self, key):
        try:
            return self._values[key]
        except KeyError:
            raise KeyError(f"{key!r} is not in [{self.section}] of {DEFAULTS_FILE}") from None

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"Defaults([{self.section}], {self._values!r})"


@functools.lru_cache(maxsize=None)
def defaults(section):
    """The table `section` of the defaults file, e.g. ``defaults("ridge")`` or
    ``defaults("track.coincidence")``.

    :param section: dotted table name.
    :return: Defaults -- a read-only mapping of the table's keys.
    :raises KeyError: if the file has no such table.
    """
    tables = _tables()
    if section not in tables:
        raise KeyError(f"[{section}] is not a table of {DEFAULTS_FILE}")
    return Defaults(section, tables[section])


def parameter(par, key, section):
    """The run's value of `key`, or its default in table `section` when the
    run does not set it (absent or None).

    :param par: run configuration (`wdf.config.Parameters`).
    :param key: run-configuration key, e.g. "RidgeSlot".
    :param section: the table of the defaults file holding its default.
    """
    value = getattr(par, key, None)
    return defaults(section)[key] if value is None else value


def tables():
    """``{table: {key: value}}`` of the whole defaults file, values as the
    file writes them ("<unset>" kept)."""
    return {name: dict(values) for name, values in _tables().items()}
