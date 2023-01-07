from .circuits import log_measurement, qec_round


def get_experiment(model, layout, num_rounds):
    # log_init = log_initialization(layout, log_state, basis)

    qec_circuit = qec_round(model, layout)
    meas_circuit = log_measurement(model, layout)

    experiment = (qec_circuit * num_rounds) + meas_circuit
    return experiment
