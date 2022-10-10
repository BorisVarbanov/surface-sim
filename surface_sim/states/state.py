from dataclasses import dataclass, field
from stim import TableauSimulator


@dataclass
class State:
    qubits: set[str]
    leaked_qubits: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._sim = TableauSimulator()
        self._sim.set_num_qubits(self.num_qubits)

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)
