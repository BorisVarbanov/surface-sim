from abc import ABC, abstractmethod
from copy import copy
from typing import Iterator, List, Optional, Tuple, Union, Sequence
from matplotlib.axes import Axes

# from ..util import param_parse
from ..layouts import Layout


class CircuitBase(ABC):
    """Abstract base  circuit class that handles circuit addition."""

    @abstractmethod
    def __copy__(self) -> "CircuitBase":
        """__copy__ Abstract copy constructor."""
        pass

    @property
    @abstractmethod
    def qubits(self) -> Tuple[str]:
        """qubits Abstract qubits property"""
        pass

    @property
    @abstractmethod
    def time(self) -> int:
        """time Abstract time property"""
        pass

    @property
    @abstractmethod
    def gates(self) -> List["CircuitBase"]:
        """List of gates, associated with this circuit."""
        pass

    def shift_to(self, time: int) -> "CircuitBase":
        """
        shift Shifts the operation in time.

        Parameters
        ----------
        time : int
            The new time of the circuit.

        Returns
        -------
        CircuitBase
            The time-shifted circuit.

        Raises
        ------
        ValueError
            If time is provided as a type other than int.
        """
        if not isinstance(time, int):
            raise ValueError(f"time expected as int, instead got {type(time)}")
        circ_copy = copy(self)
        circ_copy._time = time
        return circ_copy

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
            time_gaps = []

            for qubit in shared_qubits:
                last_time = self._get_qubit_time(qubit, reverse=True) + 1
                next_time = other._get_qubit_time(qubit)

                time_gap = last_time - next_time
                time_gaps.append(time_gap)
            print(time_gaps)
            time_shift = max(time_gaps) + 2 * other.time
            print(time_shift)
            shifted_circ = other.shift_to(time=time_shift)

            new_qubits = set(other.qubits).difference(shared_qubits)
            qubits = self.qubits + tuple(new_qubits)
            gates = self.gates + shifted_circ.gates
            return Circuit(qubits, gates)

        qubits = self.qubits + other.qubits
        gates = self.gates + other.gates
        return Circuit(qubits, gates)

    def _get_qubit_time(
        self, qubit: str, reverse: Optional[bool] = False
    ) -> "CircuitBase":
        gate_iter = reversed(self.gates) if reverse else iter(self.gates)
        for gate in gate_iter:
            if qubit in gate.qubits:
                return gate.time
        raise ValueError(f"qubit {qubit} not in any gate in the provided gates.")


class Gate(CircuitBase):
    """Circuit gate object."""

    def __init__(
        self,
        qubits: Union[str, List[str]],
        label: str,
        time: Optional[int] = 0,
    ) -> None:
        """
        __init__ Initialize the gate.

        Parameters
        ----------
        qubits : Union[str, List[str]]
            Qubit or list of qubits.
        label : str
            Gate label, compatible with the Stim simulator.
        time : Optional[int], optional
            The time of the gate, by default 0
        """
        if not qubits:
            raise ValueError("qubits must be specified.")
        if isinstance(qubits, str):
            self._qubits = (qubits,)
        elif isinstance(qubits, Sequence):
            if not all(isinstance(q, str) for q in qubits):
                raise ValueError("qubits expected as list of str type.")
            self._qubits = tuple(qubits)
        else:
            raise ValueError("qubits expected as a str type or a tuple of str types.")

        if not isinstance(time, int):
            raise ValueError(f"time expected as int type, instead got {type(time)}.")
        if time < 0:
            raise ValueError("time expected to be a positive int.")
        self._time = time

        self._label = label

    def __copy__(self) -> "Gate":
        gate_copy = self.__class__(
            self._qubits,
            self._label,
            self._time,
        )
        return gate_copy

    @property
    def qubits(self) -> List[str]:
        """Gates, associated with this circuit."""
        return self._qubits

    @property
    def time(self) -> int:
        return self._time

    @time.setter
    def time(self, new_time: int) -> None:
        if not isinstance(new_time, int):
            raise ValueError(
                f"time expected as int type, instead got {type(new_time)}."
            )
        if new_time < 0:
            raise ValueError("time expected to be a positive int.")
        self._time = new_time

    def __repr__(self) -> str:
        q_str = _get_q_str(self.qubits)
        return f"{self._label}({q_str})"

    def __str__(self) -> str:
        q_str = _get_q_str(self.qubits)
        if self.num_qubits == 1:
            return f"{self._label} on {q_str}"
        return f"{self._label} on ({q_str})"

    @property
    def gates(self) -> Tuple[CircuitBase]:
        return (self,)


class Circuit(CircuitBase):
    def __init__(
        self,
        qubits: Union[str, Sequence[str]],
        gates: Union[Gate, Sequence[Gate]],
    ) -> None:
        if not qubits:
            raise ValueError("qubits must be be specified.")
        if isinstance(qubits, str):
            self._qubits = (qubits,)
        elif isinstance(qubits, tuple):
            if not all(isinstance(q, str) for q in qubits):
                raise ValueError("qubits expected as list of str type.")
            self._qubits = qubits
        else:
            raise ValueError("qubits expected as a str type or a tuple of str types.")

        if not gates:
            raise ValueError("qubits must be be specified.")
        if isinstance(gates, Gate):
            self._gates = (gates,)
        elif isinstance(gates, Sequence):
            if not all(isinstance(g, Gate) for g in gates):
                raise ValueError("gates expected as list of Gate types.")
            self._gates = tuple(gates)
        else:
            raise ValueError(
                "gates expected to be either a Gate or a sequence of Gates."
            )
        self._time = min([gate.time for gate in self._gates])

    def __copy__(self) -> "Circuit":
        return self.__class__(
            copy(self._qubits),
            copy(self._gates),
            copy(self._label),
        )

    def __bool__(self) -> bool:
        return bool(self.gates)

    def __eq__(self, other: CircuitBase) -> bool:
        if not isinstance(other, CircuitBase):
            raise NotImplementedError
        return self.qubits == other.qubits and self.gates == other.gates

    def __ne__(self, other: CircuitBase) -> bool:
        return not self == other

    def __len__(self) -> int:
        return len(self.gates)

    def __iter__(self) -> Iterator[Gate]:
        return iter(self.gates)

    @property
    def qubits(self) -> Tuple[str]:
        return self._qubits

    @property
    def num_qubits(self) -> int:
        return len(self._qubits)

    @property
    def width(self) -> int:
        return len(self._qubits)

    @property
    def depth(self) -> int:
        return len(self._gates)

    @property
    def gates(self) -> Tuple[Gate]:
        return self._gates

    @property
    def time(self) -> int:
        return self._time

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

    def plot(
        self,
        layout: Optional[Layout] = None,
        *,
        axis: Optional[Axes] = None,
        qubit_order: Optional[Sequence[str]] = None,
    ):
        from .plotter import plot as circ_plot

        return circ_plot(self, layout, ax=axis, qubit_order=qubit_order)


def _get_q_str(qubits: List[str]) -> str:
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
        return str(qubits[0])
    return ", ".join([str(q) for q in qubits])
