from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union

import yaml

T = TypeVar("T", bound="Setup")


class Setup:
    def __init__(self, setup: Dict[str, Any]) -> None:
        self._qubits = {}

        self.name = setup.get("name")
        self.description = setup.get("description")
        self._load_setup(setup)

    def _load_setup(self, setup: Dict[str, Any]) -> None:
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
    def from_yaml(cls: Type[T], filename: Union[str, Path]) -> T:
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

    def to_dict(self) -> Dict[str, Any]:
        setup = dict()

        setup["name"] = self.name
        setup["version"] = "1"
        setup["description"] = self.description

        qubit_params = []
        for qubits, params in self._qubits.items():
            params_copy = deepcopy(params)
            num_qubits = len(qubits)
            if num_qubits == 1:
                params_copy["qubit"] = qubits[0]
            elif num_qubits == 2:
                params_copy["qubits"] = list(qubits)
            qubit_params.append(params_copy)
        setup["setup"] = qubit_params

        return setup

    def to_yaml(self, filename: Union[str, Path]) -> None:
        setup = self.to_dict()
        with open(filename, "w") as file:
            yaml.dump(setup, file, default_flow_style=False)

    def set_param(self, param: str, param_val: float, *qubits: str) -> None:
        if qubits:
            self._qubits[qubits][param] = param_val
        else:
            self._qubits[tuple()][param] = param_val

    def param(self, param: str, *qubits: str) -> float:
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
