from typing import Iterable, Iterator, List, Tuple
import numpy as np

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


class BiasedCircuitNoiseModel(Model):
    def __init__(self, setup: Setup, layout: Layout) -> None:
        super().__init__(setup, layout)

    def prob1(self, error_operation, qubit):
        rB = 1 / (1 + 1 / self.param("bias_factor", qubit))
        rj = (1 - rB) / 2
        p = self.param(error_operation, qubit)
        if self.param("bias_pauli", qubit) == 1:  # Pauli X
            rx, ry, rz = rB, rj, rj
        elif self.param("bias_pauli", qubit) == 2:  # Pauli Y
            rx, ry, rz = rj, rB, rj
        elif self.param("bias_pauli", qubit) == 3:  # Pauli Z
            rx, ry, rz = rj, rj, rB
        else:
            raise TypeError("'bias_pauli' attribute from setup must be: 1, 2 or 3")

        r = np.array([rx, ry, rz])
        return p * r  # [p * rx, p * ry, p * rz]

    def prob2(self, error_operation, qubit1, qubit2):
        rB = (1 / 8) * 1 / (1 + 1 / self.param("bias_factor", qubit1, qubit2))
        rj = (1 - 8 * rB) / 7
        p = self.param(error_operation, qubit1, qubit2)
        if self.param("bias_pauli", qubit1, qubit2) == 1:  # Pauli X
            rix, riy, riz = rB, rj, rj
            rxi, rxx, rxy, rxz = rB, rB, rB, rB
            ryi, ryx, ryy, ryz = rj, rB, rj, rj
            rzi, rzx, rzy, rzz = rj, rB, rj, rj
        elif self.param("bias_pauli", qubit1, qubit2) == 2:  # Pauli Y
            rix, riy, riz = rj, rB, rj
            rxi, rxx, rxy, rxz = rj, rj, rB, rj
            ryi, ryx, ryy, ryz = rB, rB, rB, rB
            rzi, rzx, rzy, rzz = rj, rj, rB, rj
        elif self.param("bias_pauli", qubit1, qubit2) == 3:  # Pauli Z
            rix, riy, riz = rj, rj, rB
            rxi, rxx, rxy, rxz = rj, rj, rj, rB
            ryi, ryx, ryy, ryz = rj, rj, rj, rB
            rzi, rzx, rzy, rzz = rB, rB, rB, rB
        else:
            raise TypeError("'bias_pauli' attribute from setup must be: 1, 2 or 3")

        r = np.array(
            [rix, riy, riz, rxi, rxx, rxy, rxz, ryi, ryx, ryy, ryz, rzi, rzx, rzy, rzz]
        )
        return p * r

    def x_gate(self, qubits: List[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)
        yield CircuitInstruction("X", inds)

        for qubit, ind in zip(qubits, inds):
            probs = self.prob1("sq_error_prob", qubit)
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], probs)

    def z_gate(self, qubits: List[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("Z", inds)

        for qubit, ind in zip(qubits, inds):
            probs = self.prob1("sq_error_prob", qubit)
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], probs)

    def hadamard(self, qubits: List[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("H", inds)

        for qubit, ind in zip(qubits, inds):
            probs = self.prob1("sq_error_prob", qubit)
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], probs)

    def cphase(self, qubits: List[str]) -> Iterator[CircuitInstruction]:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("CZ", inds)

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            probs = self.prob2("cz_error_prob", *qubit_pair)
            yield CircuitInstruction("PAULI_CHANNEL_2", ind_pair, probs)

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
            probs = self.prob1("idle_error_prob", qubit)
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], probs)
