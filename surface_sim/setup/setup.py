from collections import defaultdict
from typing import Dict, Type, TypeVar

import yaml

T = TypeVar("T", bound="Setup")


class Setup:
    def __init__(self, setup_dict: Dict):
        self._qubits = {}
        self._gates = defaultdict(dict)

        self._name = setup_dict.get("name")
        self._description = setup_dict.get("description")
        self._load_setup(setup_dict)

    def _load_setup(self, setup):
        params = setup.get("setup")
        if not params:
            raise ValueError("setup not found or contains no information")

        for params_dict in params:
            qubits = tuple(params_dict.pop("qubits", tuple()))
            qubit = str(params_dict.pop("qubit", ""))
            if qubit:
                qubits = (qubit,)
            if qubits in self._qubits.keys():
                raise ValueError("Parameters defined repeatedly in the setup.")
            self._qubits[qubits] = params_dict

    @classmethod
    def from_yaml(cls: Type[T], filename: str) -> T:
        """
        from_yaml Create new surface_sim.setup.Setup instance from YAML configuarion file.

        Parameters
        ----------
        filename : str
            The YAML file name.

        Returns
        -------
        T
            The initialised surface_sim.setup.Setup object based on the yaml.
        """
        with open(filename, "r") as file:
            setup = yaml.safe_load(file)
            return cls(setup)

    def set_param(self, param, param_val, *qubits):
        if qubits:
            self._qubits[qubits][param] = param_val
        else:
            self._qubits[tuple()][param] = param_val

    def param(self, param, *qubits):
        try:
            return self._qubits[qubits][param]
        except KeyError:
            pass

        try:
            return self._qubits[tuple()][param]
        except KeyError:
            pass

        raise KeyError(
            'Parameter "{}" is not defined for qubit(s) {}'.format(
                param, ", ".join(qubits)
            )
        )
