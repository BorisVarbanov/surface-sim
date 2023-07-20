from itertools import product
from typing import Iterable, Iterator, Sequence, Tuple

import numpy as np
from stim import CircuitInstruction

from .model import Model
from .util import twirled_amp_phase_damping as amp_phase_damp


def grouper(iterable: Iterable[str], block_size: int) -> Iterator[Tuple[str, ...]]:
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF ValueError
    args = [iter(iterable)] * block_size
    return zip(*args, strict=True)


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


class ExperimentalNoiseModel(Model):
    """An experimental noise model"""

    def x_gate(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        """
        x_gate Returns the circuit instructions for an X gate on the given qubits.

        Parameters
        ----------
        qubits : Sequence[str]
            The qubits to apply the X gate to.

        Yields
        ------
        Iterator[CircuitInstruction]
            The circuit instructions for an X gate on the given qubits.
        """
        inds = self.layout.get_inds(qubits)
        yield CircuitInstruction("X", inds)

        for qubit, ind in zip(qubits, inds):
            gate_error = self.param("gate_error", qubit)
            depol_prob = 2 * gate_error

            yield CircuitInstruction("DEPOLARIZE1", [ind], [depol_prob])

    def hadamard(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        """
        hadamard Returns the circuit instructions for a Hadamard gate on the given qubits.

        Parameters
        ----------
        qubits : Sequence[str]
            The qubits to apply the Hadamard gate to.

        Yields
        ------
        Iterator[CircuitInstruction]
            The circuit instructions for a Hadamard gate on the given qubits.
        """
        inds = self.layout.get_inds(qubits)
        yield CircuitInstruction("H", inds)

        for qubit, ind in zip(qubits, inds):
            gate_error = self.param("gate_error", qubit)
            depol_prob = 2 * gate_error

            yield CircuitInstruction("DEPOLARIZE1", [ind], [depol_prob])

    def cphase(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        """
        cphase Returns the circuit instructions for a CPHASE gate on the given qubits.

        Parameters
        ----------
        qubits : Sequence[str]
            The list of pairs of qubits to apply the CPHASE gate to.
            The first qubit in each pair is the control qubit, while the second is the target qubit.

        Yields
        ------
        Iterator[CircuitInstruction]
            The circuit instructions for a CPHASE gate on the given qubits.

        Raises
        ------
        ValueError
            If the number of qubits is not even.
        """
        if len(qubits) % 2 != 0:
            raise ValueError("Expected and even number of qubits.")

        inds = self.layout.get_inds(qubits)
        yield CircuitInstruction("CZ", inds)

        qubit_pairs = grouper(qubits, 2)
        ind_pairs = grouper(inds, 2)

        for qubit_pair, ind_pair in zip(qubit_pairs, ind_pairs):
            gate_error = self.param("gate_error", *qubit_pair)
            depol_prob = 4 * gate_error / 3

            yield CircuitInstruction("DEPOLARIZE2", ind_pair, [depol_prob])

    def measure(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        """
        measure Returns the circuit instructions for a measurement on the given qubits.

        Parameters
        ----------
        qubits : Sequence[str]
            The qubits to measure.

        Yields
        ------
        Iterator[CircuitInstruction]
            The circuit instructions for a measurement on the given qubits.
        """
        inds = self.layout.get_inds(qubits)

        for qubit, ind in zip(qubits, inds):
            assign_error = self.param("assign_error", qubit)
            qnd_error = self.param("qnd_error", qubit)

            yield CircuitInstruction("X_ERROR", [ind], [assign_error])
            yield CircuitInstruction("M", [ind])
            yield CircuitInstruction("X_ERROR", [ind], [qnd_error])

    def reset(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        """
        reset Returns the circuit instructions for a reset on the given qubits.

        Parameters
        ----------
        qubits : Sequence[str]
            The qubits to reset.

        Yields
        ------
        Iterator[CircuitInstruction]
            The circuit instructions for a reset on the given qubits.
        """
        inds = self.layout.get_inds(qubits)
        yield CircuitInstruction("R", inds)

        for qubit, ind in zip(qubits, inds):
            res_exc_error = self.param("res_exc_error", qubit)
            yield CircuitInstruction("X_ERROR", [ind], [res_exc_error])

    def idle(
        self, qubits: Sequence[str], duration: float
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
        inds = self.layout.get_inds(qubits)

        for qubit, ind in zip(qubits, inds):
            relax_time = self.param("relax_time", qubit)
            deph_time = self.param("deph_time", qubit)

            idle_probs = list(amp_phase_damp(relax_time, deph_time, duration))
            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], idle_probs)

    def echoed_idle(
        self, qubits: Sequence[str], duration: float
    ) -> Iterator[CircuitInstruction]:
        """
        idle Returns the circuit instructions for an idling period on the given
        qubits that includes an echo pulse in the middle.

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
        inds = self.layout.get_inds(qubits)
        duration = self.param("meas_duration")

        half_duration = duration / 2

        for qubit, ind in zip(qubits, inds):
            relax_time = self.param("relax_time", qubit)
            deph_time = self.param("deph_time", qubit)

            idle_probs = list(amp_phase_damp(relax_time, deph_time, half_duration))

            gate_error = self.param("gate_error", qubit)
            depol_prob = 2 * gate_error

            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], idle_probs)

            yield CircuitInstruction("X", [ind])
            yield CircuitInstruction("DEPOLARIZE1", [ind], [depol_prob])

            yield CircuitInstruction("PAULI_CHANNEL_1", [ind], idle_probs)

    def sq_idle(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        """
        sq_idle Utility function for inserting idling after a single qubit gate.

        Parameters
        ----------
        qubits : Sequence[str]
            The qubits to idle.

        Yields
        ------
        Iterator[CircuitInstruction]
            The circuit instructions for an idling period on the given qubits.
        """
        duration = self.param("sq_gate_duration")
        yield from self.idle(qubits, duration)

    def cz_idle(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        """
        cz_idle Utility function for inserting idling after a CZ gate.

        Parameters
        ----------
        qubits : Sequence[str]
            The qubits to idle.

        Yields
        ------
        Iterator[CircuitInstruction]
            The circuit instructions for an idling period on the given qubits.
        """
        duration = self.param("cz_gate_duration")
        yield from self.idle(qubits, duration)

    def meas_idle(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        """
        meas_idle Utility function for inserting idling on data qubits while ancilla qubits are measured.

        Parameters
        ----------
        qubits : Sequence[str]
            The qubits to idle.

        Yields
        ------
        Iterator[CircuitInstruction]
            The circuit instructions for an idling period on the given qubits.
        """
        duration = self.param("meas_duration")
        yield from self.echoed_idle(qubits, duration)
