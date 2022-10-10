from typing import Optional, Union, Sequence
from ..setup import Setup

from ..circuits import Gate


class Model:
    def __init__(self, setup: Setup, include_noise: Optional[bool] = False) -> None:
        self._setup = setup
        self._incl_noise = include_noise

    def hadamard(self, qubits: Union[str, Sequence[str]], time: int) -> Gate:
        h_gate = Gate(
            qubits=qubits,
            label="H",
            time=time,
        )
        return h_gate

    def cphase(self, qubits: Union[str, Sequence[str]], time: int) -> Gate:
        cz_gate = Gate(
            qubits=qubits,
            label="CZ",
            time=time,
        )
        return cz_gate

    def measure(self, qubits, time):
        measurement = Gate(
            qubits=qubits,
            label="M",
            time=time,
        )
        return measurement
