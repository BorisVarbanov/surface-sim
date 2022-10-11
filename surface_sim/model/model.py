from typing import Sequence, Union

from ..circuits import Gate
from ..setup import Setup


class Model:
    def __init__(self, setup: Setup) -> None:
        self.setup = setup

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
