from stim import Circuit

from ..layouts import Layout
from ..models import Model


def log_meas(
    model: Model,
    layout: Layout,
    rot_basis: bool = False,
) -> Circuit:
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    circuit = Circuit()

    if rot_basis:
        for instruction in model.hadamard(data_qubits):
            circuit.append(instruction)

        for instruction in model.idle(anc_qubits):
            circuit.append(instruction)

        circuit.append("TICK")

    for instruction in model.measure(data_qubits):
        circuit.append(instruction)

    for instruction in model.idle(anc_qubits):
        circuit.append(instruction)

    circuit.append("TICK")

    return circuit


def qec_round(model: Model, layout: Layout) -> Circuit:
    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")

    qubits = data_qubits + anc_qubits
    qubit_set = set(qubits)

    circuit = Circuit()

    for stab_type, gate_order in layout.interaction_order.items():
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

        if stab_type == "x_type":
            rot_qubits = data_qubits + stab_qubits

            for instruction in model.hadamard(rot_qubits):
                circuit.append(instruction)

            idling_qubits = list(qubit_set.difference(rot_qubits))
            for instruction in model.idle(idling_qubits):
                circuit.append(instruction)
            circuit.append("TICK")

        for ord_dir in gate_order:
            int_qubits = []
            for anc_qubit in stab_qubits:
                neighbors = layout.get_neighbors(anc_qubit, direction=ord_dir)
                for data_qubit in neighbors:
                    int_qubits.extend((anc_qubit, data_qubit))

            for instruction in model.cphase(int_qubits):
                circuit.append(instruction)

            idling_qubits = list(set(qubits).difference(int_qubits))
            for instruction in model.idle(idling_qubits):
                circuit.append(instruction)
            circuit.append("TICK")

        if stab_type == "x_type":
            rot_qubits = data_qubits + anc_qubits
        else:
            rot_qubits = stab_qubits

        for instruction in model.hadamard(rot_qubits):
            circuit.append(instruction)

        idling_qubits = list(qubit_set.difference(rot_qubits))
        for instruction in model.idle(idling_qubits):
            circuit.append(instruction)
        circuit.append("TICK")

    for instruction in model.measure(anc_qubits):
        circuit.append(instruction)

    for instruction in model.idle(data_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    circuit.append("TICK")


def log_init(
    model: Model,
    layout: Layout,
    log_state: int,
    rot_basis: bool = False,
) -> Circuit:
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    circuit = Circuit()

    for instruction in model.reset(data_qubits + anc_qubits):
        circuit.append(instruction)

    if rot_basis:
        for instruction in model.hadamard(data_qubits):
            circuit.append(instruction)

        for instruction in model.idle(anc_qubits):
            circuit.append(instruction)

        circuit.append("TICK")

    if log_state:
        instructions = (
            model.z_gate(data_qubits) if rot_basis else model.x_gate(data_qubits)
        )
        for instruction in instructions:
            circuit.append(instruction)

        for instruction in model.idle(anc_qubits):
            circuit.append(instruction)

        circuit.append("TICK")

    return circuit


def log_x(
    model: Model,
    layout: Layout,
) -> Circuit:
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    circuit = Circuit()

    for instruction in model.x_gate(data_qubits):
        circuit.append(instruction)

    for instruction in model.idle(anc_qubits):
        circuit.append(instruction)

    return circuit


def log_z(
    model: Model,
    layout: Layout,
) -> Circuit:
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    circuit = Circuit()

    for instruction in model.z_gate(data_qubits):
        circuit.append(instruction)

    for instruction in model.idle(anc_qubits):
        circuit.append(instruction)

    return circuit
