"""docs/configuration.md is the reference generated from
wdf/config/defaults.toml and the shipped configurations, and is current."""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_configuration_reference_is_current():
    spec = importlib.util.spec_from_file_location(
        "generate_configuration", os.path.join(ROOT, "docs", "generate_configuration.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with open(module.OUTPUT) as fh:
        assert fh.read() == module.generate(), \
            "docs/configuration.md is stale: run python docs/generate_configuration.py"
