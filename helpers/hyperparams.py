"""Load hyperparameters from the toml file. No need to modify this file."""
import tomllib
from types import SimpleNamespace

with open("./hyperparameters.toml", "rb") as f:
    data = tomllib.load(f)

hyperparams = SimpleNamespace(**data["default"])
