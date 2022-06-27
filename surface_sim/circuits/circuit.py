"""Module for circuit construction."""
from stim import Circuit  # type: ignore

from ..layouts import Layout
from ..schedules import Schedule
from .util import check_schedule


def get_circuit(
    layout: Layout,
    schedule: Schedule,
    model=None,
) -> Circuit:

    check_schedule(layout, schedule)

    qubits = layout.get_qubits()
    circuit = Circuit()

    for layer in schedule.layers:
        layer_gates = layer["gates"]

        for gate in layer_gates:
            label = gate["label"]
            inv_qubits = gate["qubits"]

            for gate_qubits in inv_qubits:
                if isinstance(gate_qubits, str):
                    gate_qubits = (gate_qubits,)

                inds = (qubits.index(q) for q in gate_qubits)

                circuit.append(label, inds)
        circuit.append("TICK")
    return circuit
