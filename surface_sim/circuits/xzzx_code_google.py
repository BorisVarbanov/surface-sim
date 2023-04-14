"""
The circuits are based on the following paper by Google AI:
https://doi.org/10.1038/s41586-022-05434-1
https://doi.org/10.48550/arXiv.2207.06431 
"""

from itertools import chain, compress
from typing import List

from stim import Circuit, target_rec

from ..models import Model


def qec_round_with_log_meas(
    model: Model,
    rot_basis: bool = False,
    meas_reset: bool = False,
    meas_comparison: bool = True,
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

    rot_qubits = anc_qubits
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
    circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    return circuit


def coherent_qec_part(model: Model) -> Circuit:
    """
    Returns stim circuit corresponding to the steps "a" to "h" from the QEC cycle
    described in Google's paper for the given model.
    """
    data_qubits = model.layout.get_qubits(role="data")
    anc_qubits = model.layout.get_qubits(role="anc")
    qubits = set(data_qubits + anc_qubits)
    int_order = model.layout.interaction_order
    stab_types = list(int_order.keys())

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
    cz_qubits = set()
    for stab_type in stab_types:
        ord_dir = int_order[stab_type][0]
        int_pairs = model.layout.get_neighbors(
            stab_qubits, direction=ord_dir, as_pairs=True
        )
        int_qubits = list(chain.from_iterable(int_pairs))
        cz_qubits.update(int_qubits)

        for instruction in model.cphase(int_qubits):
            circuit.append(instruction)

    idle_qubits = qubits - cz_qubits
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
    cz_qubits = set()
    for stab_type in stab_types:
        ord_dir = int_order[stab_type][1]
        int_pairs = model.layout.get_neighbors(
            stab_qubits, direction=ord_dir, as_pairs=True
        )
        int_qubits = list(chain.from_iterable(int_pairs))
        cz_qubits.update(int_qubits)

        for instruction in model.cphase(int_qubits):
            circuit.append(instruction)

    idle_qubits = qubits - cz_qubits
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # e
    x_qubits = set(anc_qubits) + set(data_qubits)
    for instruction in model.x_gate(x_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

    # f
    cz_qubits = set()
    for stab_type in stab_types:
        ord_dir = int_order[stab_type][2]
        int_pairs = model.layout.get_neighbors(
            stab_qubits, direction=ord_dir, as_pairs=True
        )
        int_qubits = list(chain.from_iterable(int_pairs))
        cz_qubits.update(int_qubits)

        for instruction in model.cphase(int_qubits):
            circuit.append(instruction)

    idle_qubits = qubits - cz_qubits
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
    cz_qubits = set()
    for stab_type in stab_types:
        ord_dir = int_order[stab_type][3]
        int_pairs = model.layout.get_neighbors(
            stab_qubits, direction=ord_dir, as_pairs=True
        )
        int_qubits = list(chain.from_iterable(int_pairs))
        cz_qubits.update(int_qubits)

        for instruction in model.cphase(int_qubits):
            circuit.append(instruction)

    idle_qubits = qubits - cz_qubits
    for instruction in model.idle(idle_qubits):
        circuit.append(instruction)
    circuit.append("TICK")

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

    for instruction in model.idle(data_qubits):
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
