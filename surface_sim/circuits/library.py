from itertools import chain

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

    n_data, n_anc = len(data_qubits), len(anc_qubits)
    proj_mat = model.layout.projection_matrix(
        stab_type="x_type" if rot_basis else "z_type"
    )
    for anc in proj_mat.coords["anc_qubit"]:
        targets_meas = []
        for idx_data, data in enumerate(data_qubits):
            if proj_mat.sel(anc_qubit=anc, data_qubit=data) != 0:
                targets_meas.append(-n_data + idx_data)
        idx_anc = anc_qubits.index(anc)
        for ind in range(1, comp_rounds + 1):
            targets_meas.append(-n_data - (ind * n_anc) + idx_anc)
        circuit.append("DETECTOR", [target_rec(targ) for targ in targets_meas])

    targets_meas = [-n_data + idx_data for idx_data in range(n_data)]
    circuit.append("OBSERVABLE_INCLUDE", [target_rec(targ) for targ in targets_meas], 0)

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
    comp_rounds = 1 if meas_reset else 2

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
    n_anc = len(anc_qubits)
    if meas_comparison:
        targets_meas = [
            [-(comp_rounds + 1) * n_anc + idx, -n_anc + idx] for idx in range(n_anc)
        ]
    else:
        targets_meas = [[-n_anc + idx] for idx in range(n_anc)]
    for targs in targets_meas:
        circuit.append("DETECTOR", [target_rec(targ) for targ in targs])

    return circuit


def init_qubits(model: Model, log_state: int, rot_basis: bool = False) -> Circuit:
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
