from stim import Circuit

from ..circuits import log_init, log_meas, qec_round
from ..layouts import Layout
from ..models import Model


def memory_exp(
    model: Model,
    layout: Layout,
    num_rounds: int,
    log_state: int,
    rot_basis: bool = False,
) -> Circuit:

    init_circ = log_init(model, layout, log_state, rot_basis)

    qec_circ = qec_round(model, layout)
    meas_circuit = log_meas(model, layout, rot_basis)

    experiment = init_circ + (qec_circ * num_rounds) + meas_circuit
    return experiment
