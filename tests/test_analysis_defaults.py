"""Every analysis value of `wdf` is in ``wdf/config/defaults.toml``.

The modules of `wdf` hold no number of their own: a value the analysis uses
is either given by the caller or by the run configuration, or read from the
defaults file through `wdf.config.defaults.defaults`. These tests read the
source of every module and fail on

- a numeric literal outside the allow-list below;
- a ``getattr(par, "Key", "text")`` fallback written in the code;
- a key the code reads from the defaults file that the file does not hold,
  and a key of the file that no code reads;
- a key of the file without its ``# [unit] meaning`` comment.

The allow-list, and why each entry is not an analysis value:

- the values 0, 1, 2, -1, 0.5 and -0.5: identities, units, halving and
  doubling (a centre is ``0.5 * (lo + hi)``, the Nyquist frequency
  ``0.5 * rate``), signs;
- the floating-point guards 1e-300, 1e-30, 1e-12 and 1e-9: floors against a
  division by zero or the logarithm of zero, and round-off margins of float
  comparisons, far below any value the analysis distinguishes;
- integers that index or shape a structure: subscripts and slices, the
  shape given to ``numpy.zeros``/``ones``/``empty``/``full``, the repeat of a
  tuple, a comparison with ``len()``, ``.ndim`` or ``.size``, ``axis=``,
  the ``type=`` of a DCT and the digits of ``round``;
- matplotlib styling keywords (`STYLE_KEYWORDS`): presentation, not
  analysis;
- `wdf/config/constants.py`, the mathematical, physical and unit
  constants, which holds no choice of the analysis.
"""
import ast
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(ROOT, "wdf")
DEFAULTS = os.path.join(PACKAGE, "config", "defaults.toml")

#: Values that are not analysis values (module docstring).
ALLOWED_VALUES = {0, 1, 2, -1, 0.5, -0.5}
#: Floating-point guards.
FLOAT_GUARDS = {1e-300, 1e-30, 1e-12, 1e-9}
#: Files whose numbers are constants of nature, units and formulas.
EXEMPT = {"wdf/config/constants.py"}
#: Files left out for now, with the reason.
PENDING = {
    "wdf/analysis/waveform_completion.py":
        "being changed on this branch by another task; its values move with it",
}
#: Keyword arguments whose values are presentation.
STYLE_KEYWORDS = {"figsize", "dpi", "fontsize", "alpha", "lw", "linewidth", "linewidths", "s",
                  "height", "zorder", "markersize", "ms", "pad", "labelpad", "ncol", "width"}
#: Keyword arguments whose values are structural.
STRUCTURAL_KEYWORDS = {"axis", "type", "ndigits", "decimals", "stacklevel", "indent"}
SHAPE_FUNCTIONS = {"zeros", "ones", "empty", "full"}
LOOKUP_FUNCTIONS = {"defaults"}


def modules():
    for folder, _, files in os.walk(PACKAGE):
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(folder, name)
                yield os.path.relpath(path, ROOT).replace(os.sep, "/"), path


def _value(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float)):
        return -node.operand.value
    return None


def _call_name(node):
    if isinstance(node, ast.Call):
        f = node.func
        return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
    return None


def _sized(node):
    return (_call_name(node) == "len"
            or (isinstance(node, ast.Attribute) and node.attr in ("ndim", "size")))


def _structural(node, parents):
    """Whether the literal `node` indexes or shapes a structure, or styles a
    plot."""
    child = node
    parent = parents.get(child)
    while parent is not None:
        if isinstance(parent, ast.Subscript) and child is parent.slice:
            return True
        if isinstance(parent, ast.keyword):
            return parent.arg in STRUCTURAL_KEYWORDS | STYLE_KEYWORDS
        if isinstance(parent, ast.Call):
            name = _call_name(parent)
            if name in SHAPE_FUNCTIONS and parent.args and child is parent.args[0]:
                return True
            if name == "round" and len(parent.args) > 1 and child is parent.args[1]:
                return True
            return False
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Mult) and child is parent.right \
                and isinstance(parent.left, ast.Tuple):
            return True
        if isinstance(parent, ast.Compare):
            return any(_sized(side) for side in [parent.left] + parent.comparators)
        if isinstance(parent, (ast.stmt, ast.Lambda, ast.comprehension)):
            return False
        child, parent = parent, parents.get(parent)
    return False


def literal_violations(path):
    tree = ast.parse(open(path).read())
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    out = []
    for node in ast.walk(tree):
        value = _value(node)
        if value is None or isinstance(node, ast.Constant) and isinstance(parents.get(node), ast.UnaryOp):
            continue
        if value in ALLOWED_VALUES or abs(value) in FLOAT_GUARDS:
            continue
        if _structural(node, parents):
            continue
        out.append(f"{node.lineno}: {value!r}")
    return out


def test_no_analysis_value_is_written_in_the_code():
    found = []
    for rel, path in modules():
        if rel in EXEMPT or rel in PENDING:
            continue
        found += [f"{rel}:{v}" for v in literal_violations(path)]
    assert not found, ("numbers written in the code; move them to wdf/config/defaults.toml "
                       "(or, for a constant of nature or of a formula, wdf/config/constants.py):\n"
                       + "\n".join(found))


def _run_key_read(node):
    """Whether `node` is ``getattr(x, "Key", ...)`` of a run key (CamelCase)."""
    return (_call_name(node) == "getattr" and isinstance(node.func, ast.Name)
            and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
            and str(node.args[1].value)[:1].isupper())


def _fallback(node):
    return isinstance(node, ast.Constant) and node.value is not None


def test_no_fallback_is_written_in_the_code():
    """A run key read with a written fallback, ``getattr(par, "Key", True)``
    or ``getattr(par, "Key", None) or "text"``: the fallback is a default and
    belongs to the defaults file (`wdf.config.defaults.parameter`)."""
    found = []
    for rel, path in modules():
        if rel in PENDING:
            continue
        for node in ast.walk(ast.parse(open(path).read())):
            if _run_key_read(node) and len(node.args) == 3 and _fallback(node.args[2]):
                found.append(f"{rel}:{node.lineno}: {node.args[1].value} = {node.args[2].value!r}")
            if (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)
                    and _run_key_read(node.values[0])
                    and any(_fallback(v) for v in node.values[1:])):
                found.append(f"{rel}:{node.lineno}: {node.values[0].args[1].value} or ...")
    assert not found, "run keys with a fallback written in the code:\n" + "\n".join(found)


def keys_read():
    """``({table: {key: [where]}}, {tables read whole})`` from the source."""
    read, whole = {}, set()
    for rel, path in modules():
        tree = ast.parse(open(path).read())
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        names = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and _call_name(node.value) in LOOKUP_FUNCTIONS
                    and node.value.args and isinstance(node.value.args[0], ast.Constant)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names[target.id] = node.value.args[0].value
        for node in ast.walk(tree):
            table = None
            if isinstance(node, ast.Subscript):
                base = node.value
                if isinstance(base, ast.Name) and base.id in names:
                    table = names[base.id]
                elif _call_name(base) in LOOKUP_FUNCTIONS and base.args \
                        and isinstance(base.args[0], ast.Constant):
                    table = base.args[0].value
                if table is not None and isinstance(node.slice, ast.Constant):
                    read.setdefault(table, {}).setdefault(node.slice.value, []).append(
                        f"{rel}:{node.lineno}")
            elif _call_name(node) == "parameter" and len(node.args) == 3 \
                    and all(isinstance(a, ast.Constant) for a in node.args[1:]):
                read.setdefault(node.args[2].value, {}).setdefault(node.args[1].value, []).append(
                    f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Name) and node.id in names and isinstance(node.ctx, ast.Load) \
                    and not (isinstance(parents.get(node), ast.Subscript)
                             and parents[node].value is node
                             and isinstance(parents[node].slice, ast.Constant)):
                whole.add(names[node.id])
    return read, whole


def test_every_key_the_code_reads_is_in_the_defaults_file():
    from wdf.config.defaults import tables

    file = tables()
    read, _ = keys_read()
    missing = [f"[{t}] {k} ({', '.join(w)})" for t, keys in read.items() for k, w in keys.items()
               if k not in file.get(t, {})]
    assert not missing, "keys read by the code and missing from defaults.toml:\n" + "\n".join(missing)


def test_every_key_of_the_defaults_file_is_read():
    from wdf.config.defaults import tables

    read, whole = keys_read()
    unused = [f"[{t}] {k}" for t, keys in tables().items() if t not in whole
              for k in keys if k not in read.get(t, {})]
    assert not unused, "keys of defaults.toml no code reads:\n" + "\n".join(unused)


COMMENTED = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=.*#\s*\[[^\]]+\]\s*\S")
ASSIGNMENT = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=")


def test_every_default_states_its_unit_and_meaning():
    bad = [f"defaults.toml:{n}: {line.strip()}"
           for n, line in enumerate(open(DEFAULTS), 1)
           if ASSIGNMENT.match(line) and not COMMENTED.match(line)]
    assert not bad, "defaults without '# [unit] meaning':\n" + "\n".join(bad)


@pytest.mark.parametrize("module", sorted(
    rel[:-3].replace("/", ".").removesuffix(".__init__") for rel, _ in modules()))
def test_every_module_imports_with_the_defaults_file(module):
    """A default read when a module is imported (a dataclass field, a default
    argument, a module constant) is in the defaults file: the import does not
    raise KeyError. Modules whose optional dependencies are missing are
    skipped."""
    import importlib

    try:
        importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] in ("py4tsa", "torch", "torch_geometric",
                                                        "pycbc", "gwpy", "coloredlogs"):
            pytest.skip(f"{module} needs {error.name}")
        raise
