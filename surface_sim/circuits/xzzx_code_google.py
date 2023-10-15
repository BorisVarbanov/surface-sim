"""
The circuits are based on the following paper by Google AI:
https://doi.org/10.1038/s41586-022-05434-1
https://doi.org/10.48550/arXiv.2207.06431 
"""

from itertools import chain, compress
from typing import List

import numpy as np

from stim import Circuit, target_rec

from ..models import Model


def qec_round_with_log_meas(
    model: Model,
    rot_basis: bool = False,
    meas_reset: bool = False,
    meas_comparison: bool = True,
    log_s_comparison: bool = None,
) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC cycle
    that includes the logical measurement
    of the given model.

    Params
    -------
    rot_basis
        By default, the logical measurement is in the Z basis.
        If rot_basis, the logical measurement is in the X basis.
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
    anc_qubits = model.layout.get_qubits(role="anc")
    data_qubits = model.layout.get_qubits(role="data")
    qubits = set(data_qubits + anc_qubits)
    num_data, num_anc = len(data_qubits), len(anc_qubits)

    # With reset defect[n] = m[n] XOR m[n-1]
    # Wihtout reset defect[n] = m[n] XOR m[n-2]
    comp_rounds = 1 if meas_reset else 2

    # a-h
    circuit = coherent_qec_part(model=model)

    # i (for logical measurement)
    stab_type = "x_type" if rot_basis else "z_type"
    stab_qubits = model.layout.get_qubits(role="anc", stab_type=stab_type)

    rot_qubits = set(anc_qubits)
    for direction in ("north_west", "south_east"):
        neighbors = model.layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits.update(neighbors)

    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - rot_qubits
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # j (for logical measurement)
    # with detectors ordered as in the measurements
    # 1) ancilla qubits
    for instruction in model.measure(anc_qubits):
        circuit.append(instruction)

    # detectors ordered as in the measurements
    if log_s_comparison is not None:
        if meas_comparison:
            comp_round = 1 if meas_reset else 2
        else:
            comp_round = 0
        x_stab = model.layout.get_qubits(role="anc", stab_type="x_type")
        pairs = logical_s_get_pairs(model.layout, "anc")
        stab_comp = dict([a, [b]] if a in x_stab else [b, [a]] for a, b in pairs)
        det_targets = get_det_targets(
            anc_qubits, comp_round, stab_comp, log_s_comparison
        )
    else:
        if meas_comparison:
            comp_round = 1 if meas_reset else 2
        else:
            comp_round = 0
        det_targets = get_det_targets(anc_qubits, comp_round)

    for targets in det_targets:
        circuit.append("DETECTOR", targets)

    # 2) data qubits
    for instruction in model.measure(data_qubits):
        circuit.append(instruction)

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


def get_det_targets(anc_qubits, comp_round, stab_comp={}, comp_stab=0):
    # With reset defect[n] = m[n] XOR m[n-1]
    # Wihtout reset defect[n] = m[n] XOR m[n-2]

    # detectors ordered as in the measurements
    num_anc = len(anc_qubits)

    def get_target(anc, c_round=0):
        ind = anc_qubits.index(anc)
        return ind - num_anc - c_round * num_anc

    det_targets = []
    for anc in anc_qubits:
        target_inds = [get_target(anc)]
        if comp_round:
            target_inds.append(get_target(anc, comp_round))
        if anc in stab_comp:
            for stab in stab_comp[anc]:
                target_inds.append(get_target(stab, comp_stab))
        targets = [target_rec(ind) for ind in target_inds]
        det_targets.append(targets)

    return det_targets


def coherent_qec_part(model: Model) -> Circuit:
    """
    Returns stim circuit corresponding to the steps "a" to "h" from the QEC cycle
    described in Google's paper for the given model.
    """
    data_qubits = model.layout.get_qubits(role="data")
    x_anc = model.layout.get_qubits(role="anc", stab_type="x_type")
    z_anc = model.layout.get_qubits(role="anc", stab_type="z_type")
    anc_qubits = x_anc + z_anc
    qubits = set(data_qubits + anc_qubits)

    circuit = Circuit()

    # a
    rot_qubits = set(anc_qubits)
    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    x_qubits = set(data_qubits)
    for instruction in model.x_gate(x_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # b
    int_pairs = model.layout.get_neighbors(
        anc_qubits, direction="north_east", as_pairs=True
    )
    int_qubits = list(chain.from_iterable(int_pairs))

    for instruction in model.cphase(int_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - set(int_qubits)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # c
    rot_qubits = set(data_qubits)
    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    x_qubits = set(anc_qubits)
    for instruction in model.x_gate(x_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # d
    x_pairs = model.layout.get_neighbors(x_anc, direction="north_west", as_pairs=True)
    z_pairs = model.layout.get_neighbors(z_anc, direction="south_east", as_pairs=True)
    int_pairs = chain(x_pairs, z_pairs)
    int_qubits = list(chain.from_iterable(int_pairs))

    for instruction in model.cphase(int_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - set(int_qubits)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # e
    x_qubits = qubits
    for instruction in model.x_gate(x_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # f
    x_pairs = model.layout.get_neighbors(x_anc, direction="south_east", as_pairs=True)
    z_pairs = model.layout.get_neighbors(z_anc, direction="north_west", as_pairs=True)
    int_pairs = chain(x_pairs, z_pairs)
    int_qubits = list(chain.from_iterable(int_pairs))

    for instruction in model.cphase(int_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - set(int_qubits)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # g
    rot_qubits = set(data_qubits)
    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    x_qubits = set(anc_qubits)
    for instruction in model.x_gate(x_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # h
    int_pairs = model.layout.get_neighbors(
        anc_qubits, direction="south_west", as_pairs=True
    )
    int_qubits = list(chain.from_iterable(int_pairs))

    for instruction in model.cphase(int_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - set(int_qubits)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    return circuit


def qec_round(
    model: Model,
    meas_reset: bool = False,
    meas_comparison: bool = True,
    log_s_comparison: bool = None,
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

    # With reset defect[n] = m[n] XOR m[n-1]
    # Wihtout reset defect[n] = m[n] XOR m[n-2]
    comp_round = 1 if meas_reset else 2

    # a-h
    circuit = coherent_qec_part(model=model)

    # i
    rot_qubits = set(anc_qubits)
    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    for instruction in model.x_gate(data_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # j
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
    if log_s_comparison is not None:
        if meas_comparison:
            comp_round = 1 if meas_reset else 2
        else:
            comp_round = 0
        x_stab = model.layout.get_qubits(role="anc", stab_type="x_type")
        pairs = logical_s_get_pairs(model.layout, "anc")
        stab_comp = dict([a, [b]] if a in x_stab else [b, [a]] for a, b in pairs)
        det_targets = get_det_targets(
            anc_qubits, comp_round, stab_comp, log_s_comparison
        )
    else:
        if meas_comparison:
            comp_round = 1 if meas_reset else 2
        else:
            comp_round = 0
        det_targets = get_det_targets(anc_qubits, comp_round)

    for targets in det_targets:
        circuit.append("DETECTOR", targets)

    return circuit


def init_qubits(model: Model, data_init: List[int], rot_basis: bool = False) -> Circuit:
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

    stab_type = "x_type" if rot_basis else "z_type"
    stab_qubits = model.layout.get_qubits(role="anc", stab_type=stab_type)

    rot_qubits = set()
    for direction in ("north_west", "south_east"):
        neighbors = model.layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits.update(neighbors)

    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - rot_qubits

    for instruction in model.idle(idle_qubits):
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
    qubits = set(anc_qubits + data_qubits)

    circuit = Circuit()

    # H due to XZZX code
    stab_type = "z_type"
    stab_qubits = model.layout.get_qubits(role="anc", stab_type=stab_type)

    rot_qubits = set()
    for direction in ("north_west", "south_east"):
        neighbors = model.layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits.update(neighbors)

    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - rot_qubits

    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # Logical S gate

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

    idle_qubits = qubits - set(s_qubits + s_dag_qubits)
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # H due to XZZX code
    stab_type = "z_type"
    stab_qubits = model.layout.get_qubits(role="anc", stab_type=stab_type)

    rot_qubits = set()
    for direction in ("north_west", "south_east"):
        neighbors = model.layout.get_neighbors(stab_qubits, direction=direction)
        rot_qubits.update(neighbors)

    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - rot_qubits

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
