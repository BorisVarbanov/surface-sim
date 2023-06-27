from itertools import chain, compress
import numpy as np

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
    if rot_basis:
        if model.layout.log_x:
            targets = [
                target_rec(data_qubits.index(data) - num_data)
                for data in data_qubits
                if data in model.layout.log_x
            ]
    else:
        if model.layout.log_z:
            targets = [
                target_rec(data_qubits.index(data) - num_data)
                for data in data_qubits
                if data in model.layout.log_z
            ]

    circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    return circuit


def qec_round(
    model: Model,
    meas_reset: bool = False,
    meas_comparison: bool = True,
    log_s_comparison: bool = False,
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
    log_s_comparison
        If True, the X detectors are set to X*Z as required for the
        logical S gate instead of just X.
    """
    data_qubits = model.layout.get_qubits(role="data")
    anc_qubits = model.layout.get_qubits(role="anc")

    qubits = set(data_qubits + anc_qubits)

    circuit = Circuit()
    int_order = model.layout.interaction_order
    stab_types = list(int_order.keys())

    for ind, stab_type in enumerate(stab_types):
        stab_qubits = model.layout.get_qubits(role="anc", stab_type=stab_type)
        rot_qubits = set(stab_qubits)
        if stab_type == "x_type":
            rot_qubits.update(data_qubits)

        if not ind:
            for instruction in model.hadamard(rot_qubits):
                circuit.append(instruction)

            idle_qubits = qubits - rot_qubits
            for instruction in model.idle(idle_qubits):
                circuit.append(instruction)
            circuit.append("TICK")

        for ord_dir in int_order[stab_type]:
            int_pairs = model.layout.get_neighbors(
                stab_qubits, direction=ord_dir, as_pairs=True
            )
            int_qubits = list(chain.from_iterable(int_pairs))

            for instruction in model.cphase(int_qubits):
                circuit.append(instruction)

            idle_qubits = qubits - set(int_qubits)
            for instruction in model.idle(idle_qubits):
                circuit.append(instruction)
            circuit.append("TICK")

        if not ind:
            for instruction in model.hadamard(qubits):
                circuit.append(instruction)
        else:
            for instruction in model.hadamard(rot_qubits):
                circuit.append(instruction)

            idle_qubits = qubits - rot_qubits
            for instruction in model.idle(idle_qubits):
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
    if log_s_comparison:
        if meas_comparison:
            comp_round = 1 if meas_reset else 2
        else:
            comp_round = 0
        x_stab = model.layout.get_qubits(role="anc", stab_type="x_type")
        pairs = logical_s_get_pairs(model.layout, "anc")
        stab_comp = dict([a, [b]] if a in x_stab else [b, [a]] for a, b in pairs)
        det_targets = get_det_targets(anc_qubits, comp_round, stab_comp)
    else:
        if meas_comparison:
            comp_round = 1 if meas_reset else 2
        else:
            comp_round = 0
        det_targets = get_det_targets(anc_qubits, comp_round)

    for targets in det_targets:
        circuit.append("DETECTOR", targets)

    return circuit


def get_det_targets(anc_qubits, comp_round, stab_comp={}):
    # With reset defect[n] = m[n] XOR m[n-1]
    # Wihtout reset defect[n] = m[n] XOR m[n-2]

    # detectors ordered as in the measurements
    num_anc = len(anc_qubits)

    def get_target(anc, comp_round=0):
        ind = anc_qubits.index(anc)
        return ind - num_anc - comp_round * num_anc

    det_targets = []
    for anc in anc_qubits:
        target_inds = [get_target(anc)]
        if comp_round:
            target_inds.append(get_target(anc, comp_round))
        if anc in stab_comp:
            for stab in stab_comp[anc]:
                target_inds.append(get_target(stab))
        targets = [target_rec(ind) for ind in target_inds]
        det_targets.append(targets)

    return det_targets


def init_qubits(model: Model, data_init: int, rot_basis: bool = False) -> Circuit:
    """
    Returns stim circuit corresponding to a logical initialization
    of the given model.
    By default, the logical measurement is in the Z basis.
    If rot_basis, the logical measurement is in the X basis.
    """
    anc_qubits = model.layout.get_qubits(role="anc")
    data_qubits = model.layout.get_qubits(role="data")

    qubits = set(data_qubits + anc_qubits)

    circuit = Circuit()

    for instruction in model.reset(qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    exc_qubits = set(compress(data_qubits, data_init))
    if exc_qubits:
        for instruction in model.x_gate(exc_qubits):
            circuit.append(instruction)

    idle_qubits = qubits - exc_qubits
    for instruction in model.idle(idle_qubits):
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


def log_s(model: Model) -> Circuit:
    """
    Returns stim circuit corresponding to a logical S gate
    of the given model using https://arxiv.org/abs/2302.07395.
    """
    anc_qubits = model.layout.get_qubits(role="anc")
    data_qubits = model.layout.get_qubits(role="data")

    circuit = Circuit()

    cz_pairs = logical_s_get_pairs(model.layout, "data")
    s_qubits, s_dag_qubits = logical_s_get_s_gates(model.layout)

    int_qubits = list(chain.from_iterable(cz_pairs))
    for instruction in model.cphase(int_qubits):
        circuit.append(instruction)

    for instruction in model.idle(anc_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    for instruction in model.s_gate(s_qubits):
        circuit.append(instruction)

    for instruction in model.s_dag_gate(s_dag_qubits):
        circuit.append(instruction)

    idle_qubits = set(anc_qubits + data_qubits) - set(s_qubits + s_dag_qubits)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    return circuit


def logical_s_get_s_gates(layout):
    # Assumes orientation where the boundary with more weight-2 X stabilizers is at the bottom
    # and that D1 is at the bottom left.
    # This layout can be generated with qec_utils

    nodes = dict(layout.graph.nodes)
    data_coords = {
        attr["coords"]: node for node, attr in nodes.items() if attr["role"] == "data"
    }

    # S gates
    s_qubits = ["D1"]
    current = np.array(nodes["D1"]["coords"]) + np.array([0, 2])
    while True:
        label = data_coords[tuple(current)]
        s_qubits.append(label)
        current += np.array([2, 2])
        if tuple(current + np.array([2, 2])) not in data_coords:
            break

    # S^\dagger gates
    label = data_coords[tuple(current)]
    s_dag_qubits = [label]

    return s_qubits, s_dag_qubits


def logical_s_get_pairs(layout, qubit_type):
    # Assumes orientation where the boundary with more weight-2 X stabilizers is at the bottom
    # and that D1 is at the bottom left.
    # This layout can be generated with qec_utils

    nodes = dict(layout.graph.nodes)
    if qubit_type == "data":
        data_coords = {
            attr["coords"]: node
            for node, attr in nodes.items()
            if attr["role"] == "data"
        }
        start = np.array(nodes["D1"]["coords"])  # minimum point
        max_row = max(data_coords, key=lambda x: x[0])[0]
        max_col = max(data_coords, key=lambda x: x[1])[1]
        final = np.array([max_row, max_col - 1])
    elif qubit_type == "anc":
        data_coords = {
            attr["coords"]: node
            for node, attr in nodes.items()
            if attr["role"] == "anc"
        }
        start = np.array(nodes["X1"]["coords"]) - np.array([0, 2])  # minimum point
        max_row = max(data_coords, key=lambda x: x[0])[0]
        max_col = max(data_coords, key=lambda x: x[1])[1]
        final = np.array([max_row, max_col - 1])
    else:
        raise (
            ValueError(
                f"qubit_type must be either 'data' or 'anc', but {qubit_type} was specified"
            )
        )

    move_to_pair = np.array([0, 2])
    pairs = []

    while (start <= final).all():
        current = start.copy()
        while (current <= final).all():
            current_pair = current + move_to_pair
            if tuple(current) in data_coords:
                label_pair = (
                    data_coords[tuple(current)],
                    data_coords[tuple(current_pair)],
                )
                pairs.append(label_pair)
            current += np.array([2, 2])

        move_to_pair += np.array([-2, 2])
        start += np.array([2, 0])

    return pairs
