"""Module for circuit construction."""
from stim import Circuit  # type: ignore

from ..layout import Layout

OPERATION_LABELS = dict(hadamard="H", cphase="CZ", measure="M")


def get_circuit(
    layout: Layout,
    schedule,
    model=None,
) -> Circuit:
    qubits = layout.get_qubits()
    qubit_times = {qubit: 0.0 for qubit in qubits}

    circuit = Circuit()

    for layer in schedule.layers:
        for gate in layer.get("gates"):
            gate_label = gate.get("label")
            op_label = OPERATION_LABELS[gate_label]

            n_qubits = gate.get("num_qubits")
            gate_qubits = gate.get("qubits")
            qubit_inds = []

            gate_time = gate.get("time")

            if n_qubits == 1:
                for qubit in gate_qubits:
                    if qubit not in qubits:
                        raise ValueError(f"Qubit {qubit} not in provided qubit list.")

                    qubit_inds.append(qubits.index(qubit))
                    qubit_times[qubit] += gate_time

            elif n_qubits == 2:
                for ctrl_q, target_q in gate_qubits:
                    if ctrl_q not in qubits or target_q not in qubits:
                        raise ValueError(
                            f"Qubit(s) {ctrl_q}, {target_q}  not in layout."
                        )

                    if target_q not in layout.get_neighbors(ctrl_q):
                        raise ValueError(
                            f"Qubit {target_q} not coupled to {ctrl_q} in layout."
                        )

                    for qubit in (ctrl_q, target_q):
                        qubit_inds.append(qubits.index(qubit))
                        qubit_times[qubit] += gate_time

            else:
                raise ValueError(
                    f"Unexpected number of qubits ({n_qubits}) for gate {gate_label}"
                )

            circuit.append_operation(op_label, qubit_inds)

        circuit.append("TICK")
