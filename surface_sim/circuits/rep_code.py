"""Module containing functions for generating circuits for a repetition code memory experiment."""
from itertools import chain, compress
from typing import List

from stim import Circuit, target_rec

from ..models import ExperimentalNoiseModel


def log_meas(model: ExperimentalNoiseModel) -> Circuit:
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
    comp_rounds = 2

    circuit = Circuit()

    for instruction in model.measure(data_qubits):
        circuit.append(instruction)

    meas_duration = model.param("meas_duration")
    for instruction in model.idle(anc_qubits, meas_duration):
        circuit.append(instruction)

    circuit.append("TICK")

    num_data = len(data_qubits)
    num_anc = len(anc_qubits)

    for anc_qubit in anc_qubits:
        neighbors = model.layout.get_neighbors(anc_qubit)
        neighbor_inds = (data_qubits.index(neighbor) for neighbor in neighbors)
        targets = [target_rec(ind - num_data) for ind in neighbor_inds]

        anc_ind = anc_qubits.index(anc_qubit)
        for round_ind in range(1, comp_rounds + 1):
            target = target_rec(anc_ind - num_data - round_ind * num_anc)
            targets.append(target)
        circuit.append("DETECTOR", targets)

    targets = [target_rec(ind) for ind in range(-num_data, 0)]
    circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    return circuit


def qec_round(model: ExperimentalNoiseModel, meas_comparison: bool = True) -> Circuit:
    """
    qec_round generates a circuit for a single round of error correction for the repetition code

    Parameters
    ----------
    model : Model
        The error model used for modelling operations
    meas_comparison : bool, optional
        Whether to compare with previous measurements when declaring the detectors, by default True

    Returns
    -------
    Circuit
        The circuit for a single round of error correction

    Raises
    ------
    NotImplementedError
        If the stabilizer type is not z_type
    NotImplementedError
        If the stabilizer type of the ancilla qubits is not the same
    """
    data_qubits = model.layout.get_qubits(role="data")
    anc_qubits = model.layout.get_qubits(role="anc")
    qubits = set(data_qubits + anc_qubits)

    stab_type = model.layout.param("stab_type", anc_qubits[0])
    if stab_type != "z_type":
        raise NotImplementedError("Only z_type is supported.")
    for anc_qubit in anc_qubits:
        anc_type = model.layout.param("stab_type", anc_qubit)
        if anc_type != stab_type:
            raise NotImplementedError("All ancilla qubits must be the same type.")

    # Wihtout reset defect[n] = m[n] XOR m[n-2]

    sq_gate_duration = model.param("sq_gate_duration")
    cz_gate_duration = model.param("cz_gate_duration")

    check_orders = (0, 1)

    circuit = Circuit()
    int_order = model.layout.interaction_order

    for ind, check_order in enumerate(check_orders):
        check_qubits = model.layout.get_qubits(role="anc", check_order=check_order)

        if not ind:
            for instruction in model.hadamard(check_qubits):
                circuit.append(instruction)

            idle_qubits = qubits.difference(check_qubits)
            for instruction in model.idle(idle_qubits, sq_gate_duration):
                circuit.append(instruction)
            circuit.append("TICK")

        for direction in int_order:
            int_pairs = model.layout.get_neighbors(
                check_qubits, direction=direction, as_pairs=True
            )
            int_qubits = list(chain.from_iterable(int_pairs))

            for instruction in model.cphase(int_qubits):
                circuit.append(instruction)

            idle_qubits = qubits.difference(int_qubits)
            for instruction in model.idle(idle_qubits, cz_gate_duration):
                circuit.append(instruction)
            circuit.append("TICK")

        if ind:
            for instruction in model.hadamard(check_qubits):
                circuit.append(instruction)

            idle_qubits = qubits.difference(check_qubits)
            for instruction in model.idle(idle_qubits, sq_gate_duration):
                circuit.append(instruction)
        else:
            for instruction in model.hadamard(anc_qubits):
                circuit.append(instruction)

            for instruction in model.idle(data_qubits, sq_gate_duration):
                circuit.append(instruction)
        circuit.append("TICK")

    for instruction in model.measure(anc_qubits):
        circuit.append(instruction)

    meas_duration = model.param("meas_duration")
    for instruction in model.echoed_idle(data_qubits, meas_duration):
        circuit.append(instruction)
    circuit.append("TICK")

    # detectors ordered as in the measurements
    num_anc = len(anc_qubits)
    if meas_comparison:
        det_targets = []
        for ind in range(num_anc):
            target_inds = [ind - 3 * num_anc, ind - num_anc]
            targets = [target_rec(ind) for ind in target_inds]
            det_targets.append(targets)
    else:
        det_targets = [[target_rec(ind - num_anc)] for ind in range(num_anc)]

    for targets in det_targets:
        circuit.append("DETECTOR", targets)

    return circuit


def init_qubits(model: ExperimentalNoiseModel, data_init: List[int]) -> Circuit:
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
    sq_gate_duration = model.param("sq_gate_duration")
    for instruction in model.idle(idle_qubits, sq_gate_duration):
        circuit.append(instruction)
    circuit.append("TICK")

    return circuit
