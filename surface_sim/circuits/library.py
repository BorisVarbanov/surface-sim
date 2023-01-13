from itertools import chain

from stim import Circuit

from ..models import Model


def log_meas(model: Model, rot_basis: bool = False) -> Circuit:
    """
    Returns stim circuit corresponding to a logical measurement
    of the given model. 
    By default, the logical measurement is in the Z basis. 
    If rot_basis, the logical measurement is in the X basis. 
    """
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

    n_data, n_anc = len(data_qubits), len(anc_qubits)
    proj_mat = model.layout.projection_matrix(stab_type="x_type" if rot_basis else "z_type")
    for anc in proj_mat.coords["anc_qubit"]:        
        detector_str = "DETECTOR"
        for idx_data, data in enumerate(data_qubits):
            if proj_mat.sel(anc_qubit=anc, data_qubit=data) != 0:
                detector_str += f" rec[{- n_data + idx_data}]"
        idx_anc = anc_qubits.index(anc)
        detector_str += f" rec[{- n_data - n_anc + idx_anc}] rec[{- n_data - 2*n_anc + idx_anc}]"
        circuit.append_from_stim_program_text(detector_str)

    observable_str = "OBSERVABLE_INCLUDE(0)"
    for idx_data, _ in enumerate(data_qubits):
        observable_str += f" rec[{-n_data + idx_data}]"
    circuit.append_from_stim_program_text(observable_str)

    return circuit


def qec_round(model: Model, time_comparison: int = 2, stab_type_det=None) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC cycle
    of the given model. 
    
    Params
    -------
    time_comparison
        Speficies the time difference (in qec cycle units) for
        the comparison of the ancilla outcomes in the detector
    stab_type_det
        If specified, only adds detectors to the ancillas for the 
        specific stabilizator type. 
    """
    if (not isinstance(time_comparison, int)) or time_comparison < 0:
        raise ValueError("'time_comparison' must be a positive integer,"
            f" but {time_comparison} (type={type(time_comparison)}) was given")

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

    n_anc = len(anc_qubits)
    stab_qubits = model.layout.get_qubits(role="anc", stab_type=stab_type_det)
    for idx, anc in enumerate(anc_qubits):
        if (stab_type_det is not None) and (anc not in stab_qubits):
            continue
        if time_comparison == 0:
            circuit.append_from_stim_program_text(f"DETECTOR rec[{-n_anc + idx}]")
        else:
            t = time_comparison + 1
            circuit.append_from_stim_program_text(f"DETECTOR rec[{-t*n_anc + idx}] rec[{-n_anc + idx}]")

    return circuit


def log_init(model: Model, log_state: int, rot_basis: bool = False) -> Circuit:
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
