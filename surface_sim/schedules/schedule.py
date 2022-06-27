"""Parameter configuration (Config) class."""
from dataclasses import dataclass
from typing import Any, Dict, List, Type, TypeVar

import yaml

T = TypeVar("T", bound="Schedule")


@dataclass
class Schedule:
    """Config class containing data, train and model hyperparameters and checkpoint/summary directories."""

    name: str
    gate_set: List[str]
    layers: List[Dict[str, Any]]

    @classmethod
    def from_yaml(cls: Type[T], filename: str) -> T:
        """
        from_yaml Create new qrennd.utils.Config instance from YAML configuarion file.

        Parameters
        ----------
        filename : str
            The YAML file name.

        Returns
        -------
        T
            The initialised qrennd.utils.Config object based on the yaml.
        """
        with open(filename, "r") as file:
            setup = yaml.safe_load(file)

        name = setup.get("name")
        gate_set = setup.get("gate_set")
        layers = setup.get("layers")

        return cls(name, gate_set, layers)
