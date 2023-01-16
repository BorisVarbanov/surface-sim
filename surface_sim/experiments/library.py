from stim import Circuit

from ..circuits import init_qubits, log_meas, qec_round
from ..models import Model


def memory_exp(
    model: Model,
    num_rounds: int,
    log_state: int,
    rot_basis: bool = False,
    meas_reset: bool = False,
) -> Circuit:

    init_circ = init_qubits(model, log_state, rot_basis)

    qec_circ = qec_round(model, meas_reset=meas_reset)
    init_qec_circ = qec_round(model, meas_reset=meas_reset, meas_comparison=False)
    meas_circuit = log_meas(model, rot_basis, meas_reset)

    num_init_rounds = 1 if meas_reset else 2

    experiment = (
        init_circ
        + init_qec_circ * num_init_rounds
        + qec_circ * (num_rounds - num_init_rounds)
        + meas_circuit
    )

    return experiment
