from itertools import chain

from stim import Circuit

from ..models import Model


def log_meas(model: Model, rot_basis: bool = False) -> Circuit:
    anc_qubits = model.layout.get_qubits(role="anc")
    data_qubits = model.layout.get_qubits(role="data")

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


def qec_round(model: Model) -> Circuit:
    data_qubits = model.layout.get_qubits(role="data")
    anc_qubits = model.layout.get_qubits(role="anc")

    qubits = data_qubits + anc_qubits

    circuit = Circuit()

    for stab_type, gate_order in model.layout.interaction_order.items():
        stab_qubits = model.layout.get_qubits(role="anc", stab_type=stab_type)

        if stab_type == "x_type":
            rot_qubits = data_qubits + stab_qubits

            for instruction in model.hadamard(rot_qubits):
                circuit.append(instruction)

            idling_qubits = [
                qubit for qubit in qubits if qubit not in rot_qubits
            ]  # Should be done by set.difference, but qubit order gets messed up
            for instruction in model.idle(idling_qubits):
                circuit.append(instruction)
            circuit.append("TICK")

        for ord_dir in gate_order:
            int_pairs = model.layout.get_neighbors(
                stab_qubits, direction=ord_dir, as_pairs=True
            )
            int_qubits = list(chain.from_iterable(int_pairs))

            for instruction in model.cphase(int_qubits):
                circuit.append(instruction)

            idling_qubits = [
                qubit for qubit in qubits if qubit not in int_qubits
            ]  # Should be done by set.difference, but qubit order gets messed up
            for instruction in model.idle(idling_qubits):
                circuit.append(instruction)
            circuit.append("TICK")

        if stab_type == "x_type":
            rot_qubits = data_qubits + anc_qubits
        else:
            rot_qubits = stab_qubits

        for instruction in model.hadamard(rot_qubits):
            circuit.append(instruction)

        idling_qubits = [
            qubit for qubit in qubits if qubit not in rot_qubits
        ]  # Should be done by set.difference, but qubit order gets messed up
        for instruction in model.idle(idling_qubits):
            circuit.append(instruction)
        circuit.append("TICK")

    for instruction in model.measure(anc_qubits):
        circuit.append(instruction)

    for instruction in model.idle(data_qubits):
        circuit.append(instruction)
    circuit.append("TICK")
    return circuit


def log_init(model: Model, log_state: int, rot_basis: bool = False) -> Circuit:
    anc_qubits = model.layout.get_qubits(role="anc")
    data_qubits = model.layout.get_qubits(role="data")

    circuit = Circuit()

    for instruction in model.reset(data_qubits + anc_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

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


def log_x(model: Model) -> Circuit:
    anc_qubits = model.layout.get_qubits(role="anc")
    data_qubits = model.layout.get_qubits(role="data")

    circuit = Circuit()

    for instruction in model.x_gate(data_qubits):
        circuit.append(instruction)

    for instruction in model.idle(anc_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    return circuit


def log_z(model: Model) -> Circuit:
    anc_qubits = model.layout.get_qubits(role="anc")
    data_qubits = model.layout.get_qubits(role="data")

    circuit = Circuit()

    for instruction in model.z_gate(data_qubits):
        circuit.append(instruction)

    for instruction in model.idle(anc_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    return circuit
