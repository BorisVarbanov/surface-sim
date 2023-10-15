import numpy as np
from stim import Circuit

from ..circuits.xzzx_code_google import (
    init_qubits,
    qec_round_with_log_meas,
    qec_round,
    log_s,
)
from ..models import Model


def memory_experiment(
    model: Model,
    num_rounds: int,
    data_init: np.ndarray,
    rot_basis: bool = False,
    meas_reset: bool = False,
) -> Circuit:
    if not isinstance(num_rounds, int):
        raise ValueError(f"num_rounds expected as int, got {type(num_rounds)} instead.")

    if num_rounds <= 0:
        raise ValueError("num_rounds needs to be a positive integer")

    num_init_rounds = 1 if meas_reset else 2

    init_circ = init_qubits(model, data_init, rot_basis)
    qec_meas_circuit = qec_round_with_log_meas(model, rot_basis, meas_reset)
    first_qec_circ = qec_round(model, meas_reset, meas_comparison=False)

    if num_rounds > num_init_rounds:
        qec_circ = qec_round(model, meas_reset)

        experiment = (
            init_circ
            + first_qec_circ * num_init_rounds
            + qec_circ * (num_rounds - 1 - num_init_rounds)
            + qec_meas_circuit
        )

        return experiment

    experiment = (
        init_circ
        + first_qec_circ * (min(num_rounds, num_init_rounds) - 1)
        + qec_round_with_log_meas(
            model, rot_basis, meas_reset=True, meas_comparison=False
        )
    )

    return experiment


def logical_s_experiment(
    model: Model,
    num_s_gates: int,
    data_init: np.ndarray,
    meas_reset: bool = False,
) -> Circuit:
    if not isinstance(num_s_gates, int):
        raise ValueError(
            f"num_s_gates expected as int, got {type(num_s_gates)} instead."
        )

    if num_s_gates <= 0:
        raise ValueError("num_s_gates needs to be a positive integer")

    assert num_s_gates % 2 == 0

    # Assume meas_reset = True for easier implementation of the detectors
    assert meas_reset
    # Rotation basis so that we see an effect of the logical S gate
    rot_basis = True

    init_circ = init_qubits(model, data_init, rot_basis)
    qec_meas_circuit = qec_round_with_log_meas(
        model, rot_basis, meas_reset=meas_reset, log_s_comparison=1
    )

    first_qec_circ = qec_round(
        model, meas_reset, meas_comparison=False, log_s_comparison=None
    )
    qec_circ_after_s_1 = qec_round(model, meas_reset=meas_reset, log_s_comparison=0)
    qec_circ_after_s_2 = qec_round(model, meas_reset=meas_reset, log_s_comparison=1)

    log_s_gate = log_s(model)

    experiment = init_circ + first_qec_circ
    for _ in range(int((num_s_gates - 1) / 2)):
        experiment += log_s_gate + qec_circ_after_s_1 + log_s_gate + qec_circ_after_s_2

    experiment += log_s_gate + qec_circ_after_s_1 + log_s_gate + qec_meas_circuit

    return experiment
