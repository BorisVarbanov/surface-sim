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

    shots = list(range(1, num_shots + 1))
    qec_rounds = list(range(1, num_rounds + 1))

    sampler = experiment.compile_sampler(seed=seed)
    outcome_vec = sampler.sample(num_shots)

    outcomes = outcome_vec.reshape(num_shots, -1)
    anc_outcomes, data_outcomes = np.split(outcomes, [num_rounds * num_anc], axis=1)
    anc_outcomes = anc_outcomes.reshape(num_shots, num_rounds, num_anc)

    anc_meas = DataArray(
        data=anc_outcomes.astype(int),
        dims=["shot", "qec_round", "anc_qubit"],
        coords=dict(shot=shots, qec_round=qec_rounds, anc_qubit=anc_qubits),
    )

    data_meas = DataArray(
        data=data_outcomes.astype(int),
        dims=["shot", "data_qubit"],
        coords=dict(shot=shots, data_qubit=data_qubits),
    )

    dataset = Dataset(
        data_vars=dict(
            anc_meas=anc_meas,
            data_meas=data_meas,
        ),
        coords=dict(seed=seed),
    )

    return dataset
