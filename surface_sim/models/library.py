from itertools import pairwise
from typing import Iterable

from stim import CircuitInstruction

from ..setup import Setup
from .model import Model


class CircuitNoiseModel(Model):
    def __init__(self, setup: Setup) -> None:
        super().__init__(setup)

    def x_gate(self, qubits: Iterable[int]) -> Iterable[CircuitInstruction]:
        targets = list(qubits)

        yield CircuitInstruction("X", targets)

        for qubit in qubits:
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [qubit], [prob])

    def z_gate(self, qubits: Iterable[int]) -> Iterable[CircuitInstruction]:
        targets = list(qubits)

        yield CircuitInstruction("Z", targets)

        for qubit in qubits:
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [qubit], [prob])

    def hadamard(self, qubits: Iterable[int]) -> Iterable[CircuitInstruction]:
        targets = list(qubits)

        yield CircuitInstruction("H", targets)

        for qubit in qubits:
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [qubit], [prob])

    def cphase(self, qubits: Iterable[int]) -> Iterable[CircuitInstruction]:
        targets = list(qubits)
        if len(targets) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        yield CircuitInstruction("CZ", targets)

        for pair in pairwise(targets):
            prob = self.param("cz_error_prob", *pair)
            yield CircuitInstruction("DEPOLARIZE2", pair, [prob])

    def measure(self, qubits: Iterable[int]) -> Iterable[CircuitInstruction]:
        targets = list(qubits)

        for target in targets:
            prob = self.param("meas_error_prob", target)
            yield CircuitInstruction("X_ERROR", [target], [prob])

        yield CircuitInstruction("M", targets)

    def reset(self, qubits: Iterable[int]) -> Iterable[CircuitInstruction]:
        targets = list(qubits)
        yield CircuitInstruction("R", targets)

        for target in targets:
            prob = self.param("reset_error_prob", target)
            yield CircuitInstruction("X_ERROR", [target], [prob])

    def idle(self, qubits: Iterable[int]) -> Iterable[CircuitInstruction]:
        targets = list(qubits)

        for target in targets:
            prob = self.param("idle_error_prob", target)
            yield CircuitInstruction("DEPOLARIZE1", [target], [prob])
