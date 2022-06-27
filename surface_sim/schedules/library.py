from math import sqrt

from ..layouts import Layout
from .schedule import Schedule


def parallel_qec_round(layout: Layout) -> Schedule:
    stab_types = ["x_type", "z_type"]

    distance = int(sqrt(len(layout.get_qubits(role="data"))))

    name = f"Distance-{distance} surface code parallel QEC round schedule."

    gate_set = ["H", "CZ", "M"]

    num_steps = 4
    gate_orders = dict(
        x_type=["north_east", "north_west", "south_east", "south_west"],
        z_type=["north_east", "south_east", "north_west", "south_west"],
    )

    layers = []

    hadamard_gates = [
        dict(
            label="H",
            num_qubits=1,
            qubits=layout.get_qubits(),
            time=20,
        )
    ]

    layers.append(dict(gates=hadamard_gates))

    for step_ind in range(num_steps):
        gates_list = []
        qubit_list = []
        for stab_type in stab_types:
            gate_direction = gate_orders[stab_type][step_ind]
            anc_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

            for anc_qubit in anc_qubits:
                neighbors = layout.param("neighbors", anc_qubit)
                data_qubit = neighbors[gate_direction]

                if data_qubit is not None:
                    if layout.param("freq_group", data_qubit) == "high":
                        qubit_pair = [anc_qubit, data_qubit]
                    else:
                        qubit_pair = [data_qubit, anc_qubit]

                    qubit_list.append(qubit_pair)

        cz_gates = dict(
            label="CZ",
            num_qubits=2,
            qubits=qubit_list,
            time=40,
        )
        gates_list.append(cz_gates)

        layers.append(dict(gates=gates_list))

    layers.append(dict(gates=hadamard_gates))

    measure_gates = dict(
        label="M",
        num_qubits=1,
        qubits=layout.get_qubits(role="anc"),
        time=600,
    )
    layers.append(
        dict(
            gates=[
                measure_gates,
            ]
        )
    )

    schedule = Schedule(name, gate_set, layers)
    return schedule
