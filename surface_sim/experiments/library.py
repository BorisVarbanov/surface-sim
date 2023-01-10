from stim import Circuit

from ..circuits import log_init, log_meas, qec_round
from ..models import Model


def memory_exp(
    model: Model,
    num_rounds: int,
    log_state: int,
    rot_basis: bool = False,
) -> Circuit:

    init_circ = log_init(model, log_state, rot_basis)

    qec_circ = qec_round(model)
    meas_circuit = log_meas(model, rot_basis)

    experiment = init_circ + (qec_circ * num_rounds) + meas_circuit
    return experiment
