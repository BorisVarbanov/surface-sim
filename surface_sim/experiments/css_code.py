import numpy as np
from stim import Circuit

from ..circuits.css_code import init_qubits, log_meas, qec_round
from ..models import Model


def memory_experiment(
    model: Model,
    num_rounds: int,
    data_init: np.ndarray,
    rot_basis: bool = False,
    meas_reset: bool = False,
    gauge_detectors: bool = True,
) -> Circuit:
    if not isinstance(num_rounds, int):
        raise ValueError(f"num_rounds expected as int, got {type(num_rounds)} instead.")

    if num_rounds <= 0:
        raise ValueError("num_rounds needs to be a positive integer")

    num_init_rounds = 1 if meas_reset else 2
    if gauge_detectors:
        stab_type_det = None
    else:
        stab_type_det = "x_type" if rot_basis else "z_type"

    init_circ = init_qubits(model, data_init, rot_basis)
    meas_circuit = log_meas(model, rot_basis, meas_reset)

    first_qec_circ = qec_round(
        model, meas_reset, meas_comparison=False, stab_type_det=stab_type_det
    )

    if num_rounds > num_init_rounds:
        qec_circ = qec_round(model, meas_reset)

        experiment = (
            init_circ
            + first_qec_circ * num_init_rounds
            + qec_circ * (num_rounds - num_init_rounds)
            + meas_circuit
        )

        return experiment

    experiment = (
        init_circ
        + first_qec_circ * min(num_rounds, num_init_rounds)
        + log_meas(model, rot_basis, meas_reset=1)
    )

    return experiment
