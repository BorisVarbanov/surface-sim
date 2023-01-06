from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from itertools import chain
from typing import Any, Optional, Sequence, Tuple, Union

import stim

from ..operations import BaseOperation


def gate_time(gate: Gate) -> int:
    return gate.time


class CircuitBase(ABC):
    def __init__(
        self,
        qubits: Union[str, Sequence[str]],
        time: int,
    ) -> None:
        if isinstance(qubits, str):
            self._qubits = (qubits,)
        elif isinstance(qubits, Sequence):
            self._qubits = tuple(qubits)
        else:
            raise ValueError(
                f"qubits expected as str or str Sequence, instead got {type(qubits)}."
            )

        if not isinstance(time, int):
            raise ValueError(f"time expected as int type, instead got {type(time)}.")
        if time < 0:
            raise ValueError("time expected to be a positive int.")

        self._time = time

    @abstractmethod
    def __copy__(self) -> CircuitBase:
        """__copy__ Abstract copy constructor."""
        pass

    @property
    def qubits(self) -> Tuple[str, ...]:
        return self._qubits

    @property
    def time(self) -> int:
        return self._time

    @abstractmethod
    def shift(self, time: int) -> CircuitBase:
        pass

    @property
    def num_qubits(self) -> int:
        return len(self._qubits)

    @property
    def depth(self) -> int:
        return max(gate.time for gate in self.gates)

    @property
    @abstractmethod
    def gates(self) -> Tuple[Gate, ...]:
        """List of gates, associated with this circuit."""
        pass

    def __add__(self, other: CircuitBase) -> Circuit:
        """
        __add__ Adds another base circuit element.
        Parameters
        ----------s
        other : CircuitBase
            The circuit element to be added.
        Returns
        -------
        Circuit
            The combined circuit.
        """
        shared_qubits = set(self.qubits).intersection(other.qubits)

        if shared_qubits:
            max_gap = 0

            for qubit in shared_qubits:
                gap = self.last_time(qubit) - other.next_time(qubit)
                if gap > max_gap:
                    max_gap = gap

            shifted_circ = other.shift(max_gap + 1)

            new_qubits = set(other.qubits).difference(shared_qubits)
            qubits = tuple(chain(self.qubits, new_qubits))
            gates = tuple(chain(self.gates, shifted_circ.gates))
            return Circuit(qubits, gates)

        qubits = tuple(chain(self.qubits, other.qubits))
        gates = tuple(chain(self.gates, other.gates))
        return Circuit(qubits, gates)

    def compile(self) -> stim.Circuit:
        circuit = stim.Circuit()

        for gate in self.gates:
            inds = [self.qubits.index(q) for q in gate.qubits]
            for unit_op in gate.operation.units:
                circuit.append(unit_op.to_instruction(inds))

        return circuit

    def next_time(self, qubit: str) -> int:
        if qubit not in self._qubits:
            raise ValueError("qubit not in circuit.")

        for gate in self.gates:
            if qubit in gate.qubits:
                return gate.time

    def last_time(self, qubit: str) -> int:
        if qubit not in self._qubits:
            raise ValueError("qubit not in circuit.")

        for gate in reversed(self.gates):
            if qubit in gate.qubits:
                return gate.time


class Gate(CircuitBase):
    def __init__(
        self,
        operation: BaseOperation,
        qubits: Union[str, Sequence[str]],
        time: int,
        name: Optional[str] = None,
        *,
        plot_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(qubits, time)
        self._op = operation
        self._name = name
        self.plot_metadata = plot_metadata

    @property
    def name(self) -> str:
        return self._name

    @property
    def operation(self) -> str:
        return self._op

    def __copy__(self) -> Gate:
        gate_copy = self.__class__(
            operation=self._op,
            qubits=self._qubits,
            time=self._time,
            name=self._name,
            plot_metadata=self.plot_metadata,
        )
        return gate_copy

    def __repr__(self) -> str:
        return f"{self._name}({qubits_str(self.qubits)})"

    def __str__(self) -> str:
        if self.num_qubits == 1:
            return f"{self._name} on {qubits_str(self.qubits)}"
        return f"{self._name} on ({qubits_str(self.qubits)})"

    @property
    def gates(self) -> Tuple[Gate, ...]:
        return (self,)

    def shift(self, time: int) -> CircuitBase:
        shifted_copy = copy(self)
        shifted_copy._time = time
        return shifted_copy


class Circuit(CircuitBase):
    def __init__(
        self,
        qubits: Union[str, Sequence[str]],
        gates: Sequence[Gate],
    ) -> None:
        time = min(gate.time for gate in gates)
        super().__init__(qubits, time)
        if isinstance(gates, Gate):
            self._gates = (gates,)
        elif isinstance(gates, Sequence):
            self._gates = tuple(gates)
        else:
            raise ValueError(
                "gates expected to be of type Gate or "
                f"Sequence[Gate], instead got {type(gates)}"
            )

    def __copy__(self) -> Circuit:
        return self.__class__(
            self._qubits,
            self._gates,
        )

    @property
    def gates(self) -> Tuple[Gate, ...]:
        return self._gates

    def shift(self, time: int) -> CircuitBase:
        shifted_gates = [gate.shift(time) for gate in self.gates]
        return Circuit(self._qubits, shifted_gates)


def qubits_str(qubits: Sequence[str]) -> str:
    """
    _get_q_str Converts a list of qubit labels to a comma-separated string format.
    Parameters
    ----------
    qubits : List[str]
        The list of qubits.
    Returns
    -------
    str
        The qubit list string.
    """
    if len(qubits) == 1:
        return qubits[0]
    return ", ".join(qubits)


def qubit_time(
    gates: Sequence[Gate], qubit: str, reverse: Optional[bool] = False
) -> int:
    gate_iter = reversed(gates) if reverse else iter(gates)
    for gate in gate_iter:
        if qubit in gate.qubits:
            return gate.time
    raise ValueError(f"qubit {qubit} not in any gate in the provided gates.")
