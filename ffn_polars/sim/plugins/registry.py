import importlib
import os
from pathlib import Path

PLUGIN_REGISTRY = {}


def load_plugins():
    plugin_dir = Path(__file__).parent
    for file in os.listdir(plugin_dir):
        if file.endswith(".py") and file not in (
            "__init__.py",
            "registry.py",
            "base.py",
        ):
            name = file[:-3]
            module = importlib.import_module(f"ffn_polars.sim.plugins.{name}")
            if (
                hasattr(module, "name")
                and hasattr(module, "type")
                and hasattr(module, "apply")
            ):
                PLUGIN_REGISTRY[module.name] = module


load_plugins()
