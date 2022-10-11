from abc import ABC, abstractmethod
from copy import copy
from typing import List, Optional, Sequence, Tuple, Union

from ..states import State


class CircuitBase(ABC):
    def __init__(self, qubits: Union[str, Sequence[str]], time: int) -> None:
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
    def __copy__(self) -> "CircuitBase":
        """__copy__ Abstract copy constructor."""
        pass

    @property
    def qubits(self) -> Tuple[str, ...]:
        return self._qubits

    @property
    def time(self) -> int:
        return self._time

    @property
    def num_qubits(self) -> int:
        return len(self._qubits)

    @property
    @abstractmethod
    def gates(self) -> Tuple["Gate", ...]:
        """List of gates, associated with this circuit."""
        pass

    def __len__(self) -> int:
        return len(self.gates)

    def __add__(self, other: "CircuitBase") -> "Circuit":
        """
        __add__ Adds another base circuit element.

        Parameters
        ----------
        other : CircuitBase
            The circuit element to be added.

        Returns
        -------
        Circuit
            The combined circuit.
        """
        shared_qubits = set(self.qubits).intersection(other.qubits)

        if shared_qubits:
            max_time_gap = 0

            for qubit in shared_qubits:
                last_time = qubit_time(self.gates, qubit, reverse=True) + 1
                next_time = qubit_time(other.gates, qubit)

                time_gap = last_time - next_time
                if time_gap > max_time_gap:
                    max_time_gap = time_gap

            time_shift = max_time_gap + 2 * other.time
            shifted_circ = other.shift_to(time=time_shift)

            new_qubits = set(other.qubits).difference(shared_qubits)
            qubits = self.qubits + tuple(new_qubits)
            gates = self.gates + shifted_circ.gates
            return Circuit(qubits, gates)

        qubits = self.qubits + other.qubits
        gates = self.gates + other.gates
        return Circuit(qubits, gates)


class Gate(CircuitBase):
    def __init__(
        self, qubits: Union[str, Sequence[str]], time: int, label: str
    ) -> None:
        super().__init__(qubits, time)
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    def __copy__(self) -> "Gate":
        gate_copy = self.__class__(
            self._qubits,
            self._time,
            self._label,
        )
        return gate_copy

    def __repr__(self) -> str:
        return f"{self.label}({qubits_str(self.qubits)})"

    def __str__(self) -> str:
        if self.num_qubits == 1:
            return f"{self._label} on {qubits_str(self.qubits)}"
        return f"{self._label} on ({qubits_str(self.qubits)})"

    @property
    def gates(self) -> Tuple["Gate"]:
        return (self,)

    @abstractmethod
    def apply_to(self, state: State) -> Union[None, int]:
        pass


class Circuit(CircuitBase):
    def __init__(
        self,
        qubits: Union[str, Sequence[str]],
        gates: Tuple[Gate, ...],
    ) -> None:

        if isinstance(gates, Gate):
            self._gates = (gates,)
        elif isinstance(gates, Sequence):
            self._gates = tuple(gates)
        else:
            raise ValueError(
                "gates expected to be either a Gate or a sequence of Gates."
            )

        time = min([gate.time for gate in self._gates])
        super().__init__(qubits, time)

    def __copy__(self) -> "Circuit":
        return self.__class__(
            copy(self._qubits),
            copy(self._gates),
        )

    @property
    def gates(self) -> Tuple[Gate]:
        return self._gates

    def apply_to(self, state: State) -> Union[None, List[int]]:
        outcomes = []
        for gate in self.gates:
            outcome = gate.apply_to(state)
            if outcome:
                outcomes.append(outcome)
        return outcomes or None

    """
    @time.setter
    def time(self, new_time: int) -> None:
        if not isinstance(new_time, int):
            raise ValueError(
                f"time expected as int type, instead got {type(new_time)}."
            )
        if new_time < 0:
            raise ValueError("time expected to be a positive int.")

        time_shift = new_time - self._time
        for gate in self.gates:
            gate.time += time_shift
        self._time = new_time
    """


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
