from typing import Optional

import numpy as np
from qec_util import Layout
from stim import Circuit
from xarray import DataArray, Dataset


def sample_experiment(
    layout: Layout,
    experiment: Circuit,
    num_shots: int,
    num_rounds: int,
    seed: Optional[int] = None,
) -> Dataset:
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")

    num_anc = len(anc_qubits)
    num_data = len(data_qubits)

    shots = list(range(1, num_shots + 1))
    qec_rounds = list(range(1, num_rounds + 1))

    # generate noisy data
    sampler = experiment.compile_sampler(seed=seed)
    outcome_vec = sampler.sample(num_shots)

    outcomes = outcome_vec.reshape(num_shots, -1)
    anc_outcomes, data_outcomes = np.split(outcomes, [num_rounds * num_anc], axis=1)
    anc_outcomes = anc_outcomes.reshape(num_shots, num_rounds, num_anc)

    anc_meas = DataArray(
        data=anc_outcomes.astype(bool),
        dims=["shot", "qec_round", "anc_qubit"],
        coords=dict(shot=shots, qec_round=qec_rounds, anc_qubit=anc_qubits),
    )

    data_meas = DataArray(
        data=data_outcomes.astype(bool),
        dims=["shot", "data_qubit"],
        coords=dict(shot=shots, data_qubit=data_qubits),
    )

    # generate ideal data
    sampler = experiment.without_noise().compile_sampler(seed=seed)
    outcome_vec = sampler.sample(1)

    outcomes = outcome_vec.reshape(1, -1)
    ideal_anc_outcomes, ideal_data_outcomes = np.split(
        outcomes, [num_rounds * num_anc], axis=1
    )
    ideal_anc_outcomes = ideal_anc_outcomes.reshape(num_rounds, num_anc)
    ideal_data_outcomes = ideal_data_outcomes.reshape(num_data)

    ideal_anc_meas = DataArray(
        data=ideal_anc_outcomes.astype(bool),
        dims=["qec_round", "anc_qubit"],
        coords=dict(qec_round=qec_rounds, anc_qubit=anc_qubits),
    )
    ideal_data_meas = DataArray(
        data=ideal_data_outcomes.astype(bool),
        dims=["data_qubit"],
        coords=dict(data_qubit=data_qubits),
    )

    dataset = Dataset(
        data_vars=dict(
            anc_meas=anc_meas,
            data_meas=data_meas,
            ideal_data_meas=ideal_data_meas,
            ideal_anc_meas=ideal_anc_meas,
        ),
        coords=dict(seed=seed),
    )

    return dataset


def binary_to_int(x: np.ndarray) -> int:
    return np.sum(2 ** np.arange(len(x))[::-1] * x)


def int_to_binary(x: int, bits: int) -> np.ndarray:
    return np.array(list(map(int, bin(x)[2:].zfill(bits))))
