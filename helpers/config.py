"""File to load config from config.toml and provide a global config object. No need to modify this file."""

import time
import tomllib
from types import SimpleNamespace

import petname


with open("./config.toml", "rb") as f:
    data = tomllib.load(f)

# Start with default config
_config_dict = data.get("default", {}).copy()

# Global config object
config = SimpleNamespace(**_config_dict)


"""Globals to store runtime information like start time, run id, etc."""
config.runtime = SimpleNamespace(
    START_TIME=None,
    RUN_ID=None,
)


def load_config(mode: str) -> None:
    """Load config with debug or prod mode.

    Args:
        mode (str): mode to load config for (debug/prod)
    """
    config.runtime.RUN_ID = petname.generate(2)
    config.runtime.START_TIME = time.time()
    mode_overrides = data.get(mode, {})
    _config_dict.update(mode_overrides)

    # Update the existing global object instead of recreating it
    for key, value in mode_overrides.items():
        setattr(config, key, value)
