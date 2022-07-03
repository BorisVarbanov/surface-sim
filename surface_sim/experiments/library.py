from stim import Circuit  # type: ignore

from ..circuits.library import log_initialization, log_measurement, sequential_qec_round


def get_noise_channel(layout, label, channel_qubits, prob):
    noise = Circuit()
    qubits = layout.get_qubits()

    inds = [qubits.index(q) for q in channel_qubits]
    noise.append(label, inds, prob)
    return noise


def get_experiment(
    layout, log_state, init_basis, meas_basis, error_prob, num_rounds, *, reset=False
):
    log_init = log_initialization(layout, log_state, init_basis)

    incoming_noise = get_noise_channel(
        layout, "DEPOLARIZE1", layout.get_qubits(), error_prob
    )
    qec_round = incoming_noise + sequential_qec_round(layout, reset=reset)
    log_meas = incoming_noise + log_measurement(layout, basis=meas_basis, reset=reset)

    experiment = log_init + (qec_round * num_rounds) + log_meas
    return experiment
