from itertools import chain, compress

from stim import Circuit, target_rec

from ..models import Model


def log_meas(
    model: Model,
    rot_basis: bool = False,
    meas_reset: bool = False,
) -> Circuit:
    """
    Returns stim circuit corresponding to a logical measurement
    of the given model.
    By default, the logical measurement is in the Z basis.
    If rot_basis, the logical measurement is in the X basis.
    """
    anc_qubits = model.layout.get_qubits(role="anc")
    data_qubits = model.layout.get_qubits(role="data")

    # With reset defect[n] = m[n] XOR m[n-1]
    # Wihtout reset defect[n] = m[n] XOR m[n-2]
    comp_rounds = 1 if meas_reset else 2

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

    num_data, num_anc = len(data_qubits), len(anc_qubits)
    stab_type = "x_type" if rot_basis else "z_type"
    stab_qubits = model.layout.get_qubits(role="anc", stab_type=stab_type)

    for anc_qubit in stab_qubits:
        neighbors = model.layout.get_neighbors(anc_qubit)
        neighbor_inds = model.layout.get_inds(neighbors)
        targets = [target_rec(ind - num_data) for ind in neighbor_inds]

        anc_ind = anc_qubits.index(anc_qubit)
        for round_ind in range(1, comp_rounds + 1):
            target = target_rec(anc_ind - num_data - round_ind * num_anc)
            targets.append(target)
        circuit.append("DETECTOR", targets)

    targets = [target_rec(ind) for ind in range(-num_data, 0)]
    circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    return circuit


def qec_round(
    model: Model, meas_reset: bool = False, meas_comparison: bool = True
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC cycle
    of the given model.

    Params
    -------
    meas_comparison
        If True, the detector is set to the measurement of the ancilla
        instead of to the comparison of consecutive syndromes.
    stab_type_det
        If specified, only adds detectors to the ancillas for the
        specific stabilizator type.
    """
    data_qubits = model.layout.get_qubits(role="data")
    anc_qubits = model.layout.get_qubits(role="anc")

    # With reset defect[n] = m[n] XOR m[n-1]
    # Wihtout reset defect[n] = m[n] XOR m[n-2]
    comp_round = 1 if meas_reset else 2

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

    if meas_reset:
        for instruction in model.reset(anc_qubits):
            circuit.append(instruction)
        for instruction in model.idle(data_qubits):
            circuit.append(instruction)

        circuit.append("TICK")

    # detectors ordered as in the measurements
    num_anc = len(anc_qubits)
    if meas_comparison:
        det_targets = []
        for ind in range(num_anc):
            target_inds = [ind - (comp_round + 1) * num_anc, ind - num_anc]
            targets = [target_rec(ind) for ind in target_inds]
            det_targets.append(targets)
    else:
        det_targets = [[target_rec(ind - num_anc)] for ind in range(num_anc)]

    for targets in det_targets:
        circuit.append("DETECTOR", targets)

    return circuit


def init_qubits(model: Model, data_init: int, rot_basis: bool = False) -> Circuit:
    """
    Returns stim circuit corresponding to a logical initialization
    of the given model.
    By default, the logical measurement is in the Z basis.
    If rot_basis, the logical measurement is in the X basis.
    """
    anc_qubits = model.layout.get_qubits(role="anc")
    data_qubits = model.layout.get_qubits(role="data")

    circuit = Circuit()

    for instruction in model.reset(data_qubits + anc_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    flipped_qubits = list(compress(data_qubits, data_init))

    if flipped_qubits:
        instructions = model.x_gate(flipped_qubits)
        for instruction in instructions:
            circuit.append(instruction)

    data_set = set(data_qubits)
    idle_qubits = list(data_set.difference(flipped_qubits))

    for instruction in model.idle(anc_qubits + idle_qubits):
        circuit.append(instruction)

    circuit.append("TICK")

    if rot_basis:
        for instruction in model.hadamard(data_qubits):
            circuit.append(instruction)

        for instruction in model.idle(anc_qubits):
            circuit.append(instruction)

        circuit.append("TICK")

    return circuit


def log_x(model: Model) -> Circuit:
    """
    Returns stim circuit corresponding to a logical X gate
    of the given model.
    """
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
    """
    Returns stim circuit corresponding to a logical Z gate
    of the given model.
    """
    anc_qubits = model.layout.get_qubits(role="anc")
    data_qubits = model.layout.get_qubits(role="data")

    circuit = Circuit()

    for instruction in model.z_gate(data_qubits):
        circuit.append(instruction)

    for instruction in model.idle(anc_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    return circuit