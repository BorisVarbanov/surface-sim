from ..layouts import Layout
from ..schedules import Schedule

SUPPORTED_LABELS = ["H", "CZ", "M", "MR"]


def check_schedule(layout: Layout, schedule: Schedule) -> None:
    qubits = layout.get_qubits()

    for layer in schedule.layers:
        layer_gates = layer["gates"]

        for gate in layer_gates:
            label = gate["label"]
            if label not in SUPPORTED_LABELS:
                raise ValueError(f"Gate label {label} not supported by Stim.")

            n_qubits = gate["num_qubits"]
            if n_qubits >= 3:
                raise ValueError("Gates involving 3 or more qubits not supported yet.")

            gate_qubits = gate["qubits"]

            if n_qubits == 1:
                for qubit in gate_qubits:
                    if qubit not in qubits:
                        raise ValueError(f"Qubit {qubit} not in provided qubit list.")

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
