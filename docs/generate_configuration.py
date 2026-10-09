"""Write ``docs/configuration.md``, the configuration reference, from the code
and the shipped configurations.

For every key a shipped configuration (``configs/*.toml``) sets, the page
gives the value the code takes when the key is absent from that file, the
value of each shipped configuration, and the meaning and unit stated by the
configuration's comment. A second table lists the keys the code reads, with
the value it takes, that no shipped configuration sets.

The defaults are read from the code: the configuration is loaded for H1 at
4096 Hz without the key, the worker and every settings class of stages 2 to
4 are built from it, and the value they write back is reported. A default
derived from another key is therefore shown evaluated with the rest of the
configuration. The workers need py4tsa.

Run from the repository root::

    python docs/generate_configuration.py            # writes the page
    python docs/generate_configuration.py --check    # exit 1 if it is stale
"""
import argparse
import glob
import os
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUTPUT = os.path.join(ROOT, "docs", "configuration.md")
#: The shipped configurations, in the column order of the page.
CONFIGS = ("online_default", "online_zerophase_default", "online_zerophase_track", "offline_default")
#: The detector a configuration is loaded for, and the run's inputs.
IFO = "H1"
INPUTS = {"sampling": 4096, "channel": "H1:X", "file": "x.gwf"}
#: Keys set by the loader or the run itself, not by a configuration.
RUNTIME = {"ConfigFile", "itf", "channel", "file", "sampling"}
ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*(?:#\s*(.*))?$")
SECTION = re.compile(r"^\[(.+)\]\s*$")


def read(path):
    """The text of a file."""
    with open(path) as fh:
        return fh.read()


def comments(path):
    """``{key: comment}`` of the top-level assignments of a TOML file."""
    out, table = {}, None
    for line in read(path).splitlines():
        m = SECTION.match(line)
        if m:
            table = m.group(1)
            continue
        m = ASSIGNMENT.match(line)
        if m and table is None and m.group(3):
            out[m.group(1)] = m.group(3).strip()
    return out


def parse(par):
    """Build the worker, its band-pass and every settings class of stages 2
    to 4 from `par`, which receives the values they use."""
    from wdf.analysis.alerts import AlertSettings
    from wdf.analysis.combine import CombineSettings
    from wdf.analysis.hrss import HrssSettings
    from wdf.analysis.network_driver import NetworkSettings
    from wdf.analysis.pair_screen import PairScreenSettings
    from wdf.analysis.ridge_tracking import RidgeSettings
    from wdf.analysis.track_coincidence import TrackCoincidenceSettings
    from wdf.analysis.track_reconstruction import TrackSettings
    from wdf.processes.BandPassDownSampling import BandPassDownSampling
    from wdf.processes.causal_census import CensusSettings
    from wdf.processes.glitch_subtraction import M4Settings
    from wdf.processes.wdfUnitOffline import wdfUnitOffline
    from wdf.processes.wdfUnitOnline import wdfUnitOnline

    name = str(getattr(par, "SearchName", ""))
    par = (wdfUnitOnline if name.startswith("online") else wdfUnitOffline)(par).par
    rate = float(par.resampling)
    BandPassDownSampling(par)
    for build in (NetworkSettings.from_parameters, PairScreenSettings.from_parameters,
                  AlertSettings.from_parameters, HrssSettings.from_parameters,
                  CombineSettings.from_parameters, TrackCoincidenceSettings.from_parameters,
                  M4Settings.from_parameters, CensusSettings.from_parameters,
                  lambda p: RidgeSettings.from_parameters(p, rate),
                  lambda p: TrackSettings.from_parameters(p, rate)):
        build(par)
    return par


def load(path, drop=None):
    """A configuration for `IFO`, without the key `drop`."""
    from wdf.config.Parameters import DETECTOR_TABLE, Parameters

    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    tables = data.pop(DETECTOR_TABLE, {})
    data.update(tables.get(IFO, {}))
    data.update(INPUTS, itf=IFO, ConfigFile=os.path.relpath(path, ROOT))
    data.pop(drop, None)
    return Parameters(**data)


def detector_values(path):
    """``{key: {ifo: value}}`` of the detector tables of a configuration."""
    from wdf.config.Parameters import DETECTOR_TABLE

    with open(path, "rb") as fh:
        tables = tomllib.load(fh).get(DETECTOR_TABLE, {})
    out = {}
    for ifo, table in tables.items():
        for key, value in table.items():
            out.setdefault(key, {})[ifo] = value
    return out


def show(value):
    """A value as it is written in a TOML file; an expression of the code as
    code; `REQUIRED` as it is."""
    if value is None:
        return "--"
    if isinstance(value, Expression):
        return f"`{' '.join(value.split())}`"
    if value is REQUIRED or value is UNSET:
        return value
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, str):
        return f'`"{value}"`'
    if isinstance(value, (list, tuple)):
        return "`[" + ", ".join(show(v).strip("`") for v in value) + "]`"
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return f"`{value:.1f}`"
    return f"`{value!r}`"


def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def source_defaults():
    """``{key: default}`` read from the source of `wdf`: the third argument of
    ``getattr(x, "Key", default)`` and the second of ``get("Key", default)``,
    or, when that is None, the right side of an ``or`` that follows the call
    (and the last argument of ``_choice(par, "Key", allowed, default)``). A
    literal is returned as its value, anything else as its source text.
    Also returns the names the code reads as attributes, and the keys it
    reads with ``getattr``."""
    import ast

    found, attributes, by_getattr = {}, set(), set()
    for path in sorted(glob.glob(os.path.join(ROOT, "wdf", "**", "*.py"), recursive=True)):
        text = read(path)
        tree = ast.parse(text)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                attributes.add(node.attr)
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None)
            at = {"getattr": 1, "get": 0, "_choice": 1}.get(name)
            if at is None or len(node.args) <= at + 1:
                continue
            key = node.args[at]
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            default = node.args[-1] if name == "_choice" else node.args[at + 1]
            parent = parents.get(node)
            if (isinstance(default, ast.Constant) and default.value is None
                    and isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or)
                    and parent.values[0] is node):
                default = parent.values[1]
            try:
                value = ast.literal_eval(default)
            except ValueError:
                value = constant(path, default)
                if value is None:
                    value = Expression(ast.get_source_segment(text, default))
            if name == "getattr":
                by_getattr.add(key.value)
            if found.get(key.value) is None:
                found[key.value] = value
    return found, attributes, by_getattr


def constant(path, node):
    """The value of a module-level name of the module at `path`, or None when
    `node` is not a name of it."""
    import ast
    import importlib

    if not isinstance(node, ast.Name):
        return None
    module = os.path.relpath(path, ROOT)[:-3].replace(os.sep, ".")
    return getattr(importlib.import_module(module), node.id, None)


class Expression(str):
    """A default stated in the code as an expression."""


#: A key the code reads with no default.
REQUIRED = "required"
#: A key whose absence the code reads as "not set" (None).
UNSET = "unset"


def default_of(path, key, scanned):
    """The value the code takes for `key` with the rest of `path`: the value
    written back by the worker and the settings classes when they fill it
    in, else the default the source states (`source_defaults`), else
    `UNSET` when the source reads its absence as None, `REQUIRED` when it
    reads the key with no default."""
    defaults, attributes, _ = scanned
    try:
        value = getattr(parse(load(path, drop=key)), key, None)
    except Exception:  # noqa: BLE001 -- a build that needs the key falls back on the source
        value = None
    if value is not None:
        return value
    if key in defaults:
        return UNSET if defaults[key] is None else defaults[key]
    return REQUIRED if key in attributes else UNSET


def generate():
    paths = {name: os.path.join(ROOT, "configs", f"{name}.toml") for name in CONFIGS}
    found = sorted(glob.glob(os.path.join(ROOT, "configs", "*.toml")))
    if sorted(paths.values()) != found:
        raise SystemExit(f"configs/ holds {found}; CONFIGS names {sorted(paths.values())}")
    values, notes, tables = {}, {}, {}
    for name, path in paths.items():
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        data.pop("detectors", None)
        values[name] = data
        tables[name] = detector_values(path)
        for key, text in comments(path).items():
            notes.setdefault(key, text)
    keys = []
    for name in CONFIGS:
        keys += [k for k in values[name] if k not in keys]
    for name in CONFIGS:
        keys += [k for k in tables[name] if k not in keys]

    lines = [
        "# Configuration reference",
        "",
        "<!-- Generated by docs/generate_configuration.py from the code and configs/*.toml;",
        "     do not edit by hand. -->",
        "",
        "A run is configured by a TOML file (`wdf.config.Parameters.load_config`). Its",
        "top-level keys are the run's parameters; a `[detectors.<IFO>]` table overrides",
        "them for one detector. Every value the run uses, given or defaulted, is",
        "written to its `parametersUsed` file.",
        "",
        "Units: `window`, `overlap`, `ARorder`, `ARWarmup`, `RidgeWindow` and",
        "`RidgeOverlap` are in samples of the decimated stream; durations are in",
        "seconds; frequencies in hertz. The meaning column states the unit of every",
        "other key.",
        "",
        "**Default** is the value the code takes when the key is absent, read from",
        "the code: the first shipped configuration that sets the key is loaded for",
        "H1 at 4096 Hz without it, the worker and the settings of stages 2 to 4 are",
        "built from it, and the value they fill in is reported (a default derived",
        "from another key is therefore shown evaluated with the rest of that file).",
        "A key filled in later in the run is reported with the default its source",
        "states, as a value or as the expression that computes it. \"required\"",
        "means the code reads the key with no default; \"unset\" that the code",
        "reads its absence as not set, with the behaviour the meaning column states. The four columns that follow",
        "are the shipped configurations; a value set per detector is given as",
        "`H1/L1/V1`; -- is a key the file does not set.",
        "",
        "## Keys of the shipped configurations",
        "",
        "| Key | Default | online | online zero-phase | online zero-phase track | offline | Meaning |",
        "|---|---|---|---|---|---|---|",
    ]
    scanned = source_defaults()
    for key in keys:
        source = next(paths[n] for n in CONFIGS if key in values[n] or key in tables[n])
        row = [f"`{key}`", show(default_of(source, key, scanned))]
        for name in CONFIGS:
            per = tables[name].get(key)
            if per:
                row.append("/".join(show(per.get(i)).strip("`") for i in ("H1", "L1", "V1")))
            else:
                row.append(show(values[name].get(key)))
        row.append(cell(notes.get(key, "")))
        lines.append("| " + " | ".join(row) + " |")

    used = vars(parse(load(paths[CONFIGS[0]])))
    named = {k for k in scanned[2] if k[:1].isupper()}
    rest = sorted((set(used) | named) - set(keys) - RUNTIME, key=str.lower)
    lines += [
        "",
        "## Keys the code reads that no shipped configuration sets",
        "",
        "Each is described with its stage in [Online and offline](design/online-offline.md)",
        "and in the docstring of the class that reads it. A value is the one",
        "`online_default.toml` runs with, else the default the source states.",
        "",
        "| Key | Default |",
        "|---|---|",
    ]
    for k in rest:
        value = used.get(k, scanned[0].get(k))
        lines.append(f"| `{k}` | {cell(show(UNSET if value is None else value))} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the page is stale")
    args = parser.parse_args()
    text = generate()
    if args.check:
        current = read(OUTPUT) if os.path.exists(OUTPUT) else ""
        if current != text:
            print(f"{OUTPUT} is stale: run python docs/generate_configuration.py")
            return 1
        return 0
    with open(OUTPUT, "w") as fh:
        fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
