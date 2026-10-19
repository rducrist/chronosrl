"""Environment registry that maps an environment name to the module building it."""

import importlib
from types import ModuleType


def get_environment(environment_name: str) -> ModuleType:
    """Return the environment package (``go2``, ``crazyflow`` or ``jaxgcrl``) for a name."""
    if environment_name.startswith("go2"):
        module = "go2"
    elif environment_name.startswith("crazyflow_"):
        module = "crazyflow"
    else:
        module = "jaxgcrl"

    return importlib.import_module(f"chronosrl.environments.{module}")
