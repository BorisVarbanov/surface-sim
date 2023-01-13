from stim import Circuit

from ..circuits import init_qubits, log_meas, qec_round, MEASUREMENT_RESET
from ..models import Model


def memory_exp(
    model: Model,
    num_rounds: int,
    log_state: int,
    rot_basis: bool = False,
) -> Circuit:

    init_circ = init_qubits(model, log_state, rot_basis)
    qec_circ = qec_round(model)
    meas_circuit = log_meas(model, rot_basis)

    if MEASUREMENT_RESET:
        experiment = (
            init_circ
            + qec_round(model, meas_comparison=False)
            + qec_circ * (num_rounds - 1)
            + meas_circuit
        )
    else:
        experiment = (
            init_circ
            + qec_round(model, meas_comparison=False)
            + qec_round(model, meas_comparison=False)
            + qec_circ * (num_rounds - 2)
            + meas_circuit
        )
    return experiment
