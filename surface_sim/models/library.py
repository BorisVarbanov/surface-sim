from typing import Iterable, Iterator, List, Tuple

from qec_util import Layout
from stim import CircuitInstruction

from ..setup import Setup
from .model import Model


def grouper(iterable: Iterable[str], block_size: int) -> Iterator[Tuple[str, ...]]:
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF ValueError
    args = [iter(iterable)] * block_size
    return zip(*args, strict=True)


class CircuitNoiseModel(Model):
    def __init__(self, setup: Setup, layout: Layout) -> None:
        super().__init__(setup, layout)

    def x_gate(self, qubits: List[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)
        yield CircuitInstruction("X", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def z_gate(self, qubits: List[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("Z", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def hadamard(self, qubits: List[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("H", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def cphase(self, qubits: List[str]) -> Iterator[CircuitInstruction]:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("CZ", inds)

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("cz_error_prob", *qubit_pair)
            yield CircuitInstruction("DEPOLARIZE2", ind_pair, [prob])

    def measure(self, qubits: List[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            yield CircuitInstruction("X_ERROR", [ind], [prob])

        yield CircuitInstruction("M", inds)

    def reset(self, qubits: List[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)
        yield CircuitInstruction("R", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            yield CircuitInstruction("X_ERROR", [ind], [prob])

    def idle(self, qubits: List[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("idle_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])
