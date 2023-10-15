from itertools import product
from typing import Iterable, Iterator, Sequence, Tuple

import numpy as np
from qec_util import Layout
from stim import CircuitInstruction

from ..setup import Setup
from .model import Model


def grouper(iterable: Iterable[str], block_size: int) -> Iterator[Tuple[str, ...]]:
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF ValueError
    args = [iter(iterable)] * block_size
    return zip(*args)  # , strict=True


def biased_prefactors(biased_pauli: str, biased_factor: float, num_qubits: int):
    """
    biased_prefactors Return a biased channel prefactors.

    The bias of the channel is defined as any error operator that
    applied the biased Pauli operator on any qubit.

    Parameters
    ----------
    biased_pauli : str
        The biased Pauli operator, represented as a string
    biased_factor : float
        The strength of the bias.

        A bias factor of 1 corresponds to a standard depolarizing channel.
        A bias factor of 0 leads to no probability of biased errors occurring.
        A bias channel tending towards infinify (but inf not supported) leads to
        only the biased errors occurring.
    num_qubits : int
        The number of qubits in the channel.

    Returns
    -------
    np.ndarray
        The array of prefactors
    """
    paulis = ["I", "X", "Y", "Z"]
    # get all pauli combinations and remove identity operator
    operators = list(product(paulis, repeat=num_qubits))[1:]
    num_ops = len(operators)

    biased_ops = [op for op in operators if biased_pauli in op]
    num_biased = len(biased_ops)

    nonbias_prefactor = 1 / (num_biased * (biased_factor - 1) + num_ops)
    bias_prefactor = biased_factor * nonbias_prefactor

    prefactors = []
    for op in operators:
        if biased_pauli in op:
            prefactors.append(bias_prefactor)
        else:
            prefactors.append(nonbias_prefactor)
    prefactors = np.array(prefactors)

    return prefactors


class CircuitNoiseModel(Model):
    def __init__(self, setup: Setup, layout: Layout) -> None:
        super().__init__(setup, layout)

    def x_gate(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)
        yield CircuitInstruction("X", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def z_gate(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("Z", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def s_gate(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("S", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def s_dag_gate(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("S_DAG", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def hadamard(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("H", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def cphase(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("CZ", inds)

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("cz_error_prob", *qubit_pair)
            yield CircuitInstruction("DEPOLARIZE2", ind_pair, [prob])

    def measure(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            yield CircuitInstruction("X_ERROR", [ind], [prob])

            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                yield CircuitInstruction("MZ", [ind], [prob])
            else:
                yield CircuitInstruction("MZ", [ind])

    def reset(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)
        yield CircuitInstruction("R", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            yield CircuitInstruction("X_ERROR", [ind], [prob])

    def idle(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("idle_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])

    def idle_two_qubit(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.layout.get_inds(qubits)

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("cz_error_prob", *qubit_pair)
            yield CircuitInstruction("DEPOLARIZE2", ind_pair, [prob])


class BiasedCircuitNoiseModel(Model):
    def __init__(self, setup: Setup, layout: Layout) -> None:
        super().__init__(setup, layout)

    def x_gate(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)
        yield CircuitInstruction("X", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            probs = prob * prefactors
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], probs)

    def z_gate(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("Z", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            probs = prob * prefactors
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], probs)

    def hadamard(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("H", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            probs = prob * prefactors
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], probs)

    def cphase(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.layout.get_inds(qubits)

        yield CircuitInstruction("CZ", inds)

        for qubit_pair, ind_pair in zip(grouper(qubits, 2), grouper(inds, 2)):
            prob = self.param("cz_error_prob", *qubit_pair)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", *qubit_pair),
                biased_factor=self.param("biased_factor", *qubit_pair),
                num_qubits=2,
            )
            probs = prob * prefactors
            yield CircuitInstruction("PAULI_CHANNEL_2", ind_pair, probs)

    def measure(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("meas_error_prob", qubit)
            yield CircuitInstruction("X_ERROR", [ind], [prob])

            if self.param("assign_error_flag", qubit):
                prob = self.param("assign_error_prob", qubit)
                yield CircuitInstruction("MZ", [ind], [prob])
            else:
                yield CircuitInstruction("MZ", [ind])

    def reset(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)
        yield CircuitInstruction("R", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("reset_error_prob", qubit)
            yield CircuitInstruction("X_ERROR", [ind], [prob])

    def idle(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.layout.get_inds(qubits)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("idle_error_prob", qubit)
            prefactors = biased_prefactors(
                biased_pauli=self.param("biased_pauli", qubit),
                biased_factor=self.param("biased_factor", qubit),
                num_qubits=1,
            )
            prob = prob * prefactors
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], prob)
