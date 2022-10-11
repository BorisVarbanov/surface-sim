from typing import Sequence, Union

from ..states import State
from .circuit import Gate


class Hadamard(Gate):
    def __init__(self, qubits: Union[str, Sequence[str]], time: int) -> None:
        super().__init__(qubits, time, label="H")

    def apply_to(self, state: State) -> None:
        qubit = self.qubits[0]
        if not state.is_leaked(qubit):
            state.tableau.h(state.index(qubit))


class Measure(Gate):
    def __init__(self, qubits: Union[str, Sequence[str]], time: int) -> None:
        super().__init__(qubits, time, label="M")

    def apply_to(self, state: State) -> int:
        qubit = self.qubits[0]
        if state.is_leaked(qubit):
            return 2
        return state.tableau.measure(state.index(qubit))


class CPhase(Gate):
    def __init__(self, qubits: Union[str, Sequence[str]], time: int) -> None:
        super().__init__(qubits, time, label="CZ")

    def apply_to(self, state: State) -> None:
        ctrl_qubit, tar_qubit = self.qubits
        if not (state.is_leaked(ctrl_qubit) or state.is_leaked(tar_qubit)):
            state.tableau.cz(state.index(ctrl_qubit), state.index(tar_qubit))


class Reset(Gate):
    def __init__(self, qubits: Union[str, Sequence[str]], time: int) -> None:
        super().__init__(qubits, time, label="R")

    def apply_to(self, state: State) -> None:
        qubit = self.qubits[0]
        state.tableau.reset(state.index(qubit))
        state.set_comp(qubit)
