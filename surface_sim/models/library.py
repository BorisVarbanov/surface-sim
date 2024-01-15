from typing import Iterable, Iterator, Sequence, Tuple, Any

import numpy as np
from qec_util import Layout
from stim import CircuitInstruction

from ..setup import Setup
from .model import Model
from .util import biased_prefactors, grouper, idle_error_probs


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


class DecoherenceModel(Model):
    """An coherence-limited noise model using T1 and T2"""

    def __init__(self, setup: Setup, symmetric_noise: bool = False) -> Any:
        self._sym_noise = symmetric_noise
        return super().__init__(setup)

    def generic_op(
        self, name: str, qubits: Sequence[str]
    ) -> Iterator[CircuitInstruction]:
        """
        generic_op Returns the circuit instructions for a generic operation (that is supported by Stim) on the given qubits.

        Parameters
        ----------
        name : str
            The name of the gate (as defined in Stim)
        qubits : Sequence[str]
            The qubits to apply the gate to.

        Yields
        ------
        Iterator[CircuitInstruction]
            The circuit instructions for a generic gate on the given qubits.
        """
        if self._sym_noise:
            duration = 0.5 * self._setup.gate_durations[name]

            yield from self.idle(qubits, duration)
            yield CircuitInstruction(name, qubits)
            yield from self.idle(qubits, duration)
        else:
            duration = self._setup.gate_durations[name]

            yield CircuitInstruction(name, qubits)
            yield from self.idle(qubits, duration)

    def x_gate(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("X", qubits)

    def y_gate(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("Y", qubits)

    def z_gate(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("Z", qubits)

    def s_gate(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("S", qubits)

    def hadamard(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("H", qubits)

    def cnot(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("CNOT", qubits)

    def cphase(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("CZ", qubits)

    def swap(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("SWAP", qubits)

    def iswap(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("ISWAP", qubits)

    def cphase(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("CZ", qubits)

    def measure(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("M", qubits)

    def reset(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield from self.generic_op("R", qubits)

    def idle(
        self, qubits: Sequence[int], duration: float
    ) -> Iterator[CircuitInstruction]:
        """
        idle Returns the circuit instructions for an idling period on the given qubits.

        Parameters
        ----------
        qubits : Sequence[str]
            The qubits to idle.
        duration : float
            The duration of the idling period.

        Yields
        ------
        Iterator[CircuitInstruction]
            The circuit instructions for an idling period on the given qubits.
        """
        for qubit in qubits:
            relax_time = self.param("T1", qubit)
            deph_time = self.param("T2", qubit)
            # check that the parameters are physical
            assert (relax_time > 0) and (deph_time > 0) and (2 * deph_time < relax_time)

            error_probs = idle_error_probs(relax_time, deph_time, duration)

            yield CircuitInstruction("PAULI_CHANNEL_1", [qubit], error_probs)
