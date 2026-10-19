"""Locating and loading the TOML experiment configs in `configs/`."""

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _lists_to_tuples(value: Any) -> Any:
    # config_flags can override tuples from the CLI, but not lists
    if isinstance(value, dict):
        return {key: _lists_to_tuples(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value


def default_config_path(environment_name: str) -> Path:
    """Return the path of the default TOML config for an environment."""
    return CONFIG_DIR / f"{environment_name}.toml"


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML file into nested dicts, with lists converted to tuples."""
    with open(path, "rb") as file:
        return _lists_to_tuples(tomllib.load(file))
