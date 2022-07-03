from typing import Optional

from stim import Circuit  # type: ignore

from ..layouts import Layout

STAB_TYPES = ["x_type", "z_type"]
GATE_ORDERS = dict(
    x_type=["north_east", "north_west", "south_east", "south_west"],
    z_type=["north_east", "south_east", "north_west", "south_west"],
)
NUM_STEPS = 4


def log_measurement(
    layout: Layout,
    basis: Optional[str] = "z_basis",
    *,
    reset: Optional[bool] = False,
) -> Circuit:
    meas_circ = Circuit()
    qubits = layout.get_qubits()

    data_qubits = layout.get_qubits(role="data")
    meas_label = "MZ" if basis == "z_basis" else "MX"
    for data_qubit in data_qubits:
        meas_circ.append(meas_label, qubits.index(data_qubit))

    if reset:
        reset_label = "RZ" if basis == "z_basis" else "RX"
        for data_qubit in data_qubits:
            meas_circ.append(reset_label, qubits.index(data_qubit))

    return meas_circ


def parallel_qec_round(layout: Layout, *, reset: Optional[bool] = False) -> Circuit:
    qec_round_circ = Circuit()
    qubits = layout.get_qubits()

    for qubit in layout.get_qubits(role="anc", stab_type="x_type"):
        qec_round_circ.append("H", qubits.index(qubit))
    qec_round_circ.append("TICK")

    for step_ind in range(NUM_STEPS):
        for stab_type in STAB_TYPES:
            anc_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
            direction = GATE_ORDERS[stab_type][step_ind]

            for anc_qubit in anc_qubits:
                neighbors = layout.param("neighbors", anc_qubit)
                data_qubit = neighbors[direction]

                if data_qubit:
                    if stab_type == "x_type":
                        qubit_pair = (anc_qubit, data_qubit)
                    else:
                        qubit_pair = (data_qubit, anc_qubit)

                    qec_round_circ.append("CNOT", (qubits.index(q) for q in qubit_pair))
        qec_round_circ.append("TICK")

    for qubit in layout.get_qubits(role="anc", stab_type="x_type"):
        qec_round_circ.append("H", qubits.index(qubit))
    qec_round_circ.append("TICK")

    meas_label = "MR" if reset else "M"
    for anc_qubit in layout.get_qubits(role="anc"):
        qec_round_circ.append(meas_label, qubits.index(anc_qubit))
    qec_round_circ.append("TICK")

    return qec_round_circ


def sequential_qec_round(layout: Layout, *, reset: Optional[bool] = False) -> Circuit:
    qec_round_circ = Circuit()
    qubits = layout.get_qubits()

    data_qubits = layout.get_qubits(role="data")

    for stab_type in STAB_TYPES:
        anc_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
        if stab_type == "x_type":
            rot_qubits = data_qubits + anc_qubits
        else:
            rot_qubits = anc_qubits

        for qubit in rot_qubits:
            qec_round_circ.append("H", qubits.index(qubit))
        qec_round_circ.append("TICK")

        cz_order = GATE_ORDERS[stab_type]

        for direction in cz_order:
            for anc_qubit in anc_qubits:
                neighbors = layout.param("neighbors", anc_qubit)
                data_qubit = neighbors[direction]

                if data_qubit:
                    if layout.param("freq_group", data_qubit) == "high":
                        qubit_pair = [anc_qubit, data_qubit]
                    else:
                        qubit_pair = [data_qubit, anc_qubit]

                    qec_round_circ.append("CZ", (qubits.index(q) for q in qubit_pair))
        qec_round_circ.append("TICK")

        for qubit in rot_qubits:
            qec_round_circ.append("H", qubits.index(qubit))
        qec_round_circ.append("TICK")

    meas_label = "MR" if reset else "M"
    for anc_qubit in layout.get_qubits(role="anc"):
        qec_round_circ.append(meas_label, qubits.index(anc_qubit))
    qec_round_circ.append("TICK")
    return qec_round_circ


def log_initialization(
    layout: Layout, log_state: Optional[int] = 0, basis: Optional[str] = "z_basis"
) -> Circuit:
    init_circ = Circuit()

    qubits = layout.get_qubits()

    reset_label = "R" if basis == "z_basis" else "RX"
    for qubit in qubits:
        init_circ.append(reset_label, qubits.index(qubit))

    if log_state == 1:
        gate = "X" if basis == "z_basis" else "Z"
        data_qubits = layout.get_qubits(role="data")

        for data_qubit in data_qubits:
            init_circ.append(gate, qubits.index(data_qubit))

    return init_circ
