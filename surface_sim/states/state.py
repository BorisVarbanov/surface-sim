from typing import Sequence, Tuple

from stim import TableauSimulator


class State:
    def __init__(self, qubits: Sequence[str]) -> None:
        self.qubits = parse_qubits(qubits)
        self.inds = {qubits[ind]: ind for ind in range(self.num_qubits)}
        self.leaked_inds = set()

        self.tableau = TableauSimulator()
        self.tableau.set_num_qubits(self.num_qubits)

    def __contains__(self, qubit: str) -> bool:
        return qubit in self.qubits

    def _validate_qubit(self, qubit: str) -> None:
        if qubit not in self.qubits:
            raise ValueError(f"Qubit {qubit} not in state.")

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    def is_leaked(self, qubit: str) -> bool:
        self._validate_qubit(qubit)
        return self.index(qubit) in self.leaked_inds

    def set_leaked(self, qubit: str) -> None:
        self.leaked_inds.add(self.index(qubit))

    def set_comp(self, qubit: str) -> None:
        self._validate_qubit(qubit)
        self.leaked_inds.discard(self.index(qubit))

    def index(self, qubit: str) -> int:
        return self.inds[qubit]


def parse_qubits(qubits: Sequence[str]) -> Tuple[str]:
    qubits = tuple(qubits)
    for qubit in qubits:
        if not isinstance(qubit, str):
            raise ValueError(
                f"Each qubit expected as string, instead received type {type(qubit)}"
            )

    num_qubits = len(qubits)

    if num_qubits != len(set(qubits)):
        raise ValueError("qubits contains repetetions of the same qubit")

    return qubits
