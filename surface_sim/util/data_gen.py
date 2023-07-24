from typing import Optional

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

    # generate noisy data
    sampler = experiment.compile_sampler(seed=seed)
    outcome_vec = sampler.sample(shots=num_shots).astype(bool)

    outcomes = outcome_vec.reshape(num_shots, -1)

    if num_rounds != 0:
        qec_rounds = list(range(1, num_rounds + 1))

        num_meas = num_rounds * num_anc
        anc_outcomes = outcomes[..., :num_meas]
        anc_outcomes = anc_outcomes.reshape(num_shots, num_rounds, num_anc)

        data_outcomes = outcomes[..., num_meas:]

        anc_meas = DataArray(
            data=anc_outcomes,
            dims=["shot", "qec_round", "anc_qubit"],
            coords=dict(shot=shots, qec_round=qec_rounds, anc_qubit=anc_qubits),
        )
        data_meas = DataArray(
            data=data_outcomes,
            dims=["shot", "data_qubit"],
            coords=dict(shot=shots, data_qubit=data_qubits),
        )

        data_vars = dict(anc_meas=anc_meas, data_meas=data_meas)
        dataset = Dataset(data_vars)

        return dataset

    data_meas = DataArray(
        data=outcomes,
        dims=["shot", "data_qubit"],
        coords=dict(shot=shots, data_qubit=data_qubits),
    )

    data_vars = dict(data_meas=data_meas)
    dataset = Dataset(data_vars)
    return dataset
