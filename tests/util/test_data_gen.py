import numpy as np

from qec_util.layouts import rot_surf_code

from surface_sim.util import sample_memory_experiment
from surface_sim.models import NoiselessModel
from surface_sim.experiments.surface_code_xzzx import memory_experiment


def test_sample_memory_experiment():
    layout = rot_surf_code(distance=3)
    qubit_ids = {q: i for i, q in enumerate(layout.get_qubits())}
    model = NoiselessModel(qubit_ids)
    circuit = memory_experiment(
        model=model,
        layout=layout,
        num_rounds=10,
        meas_reset=False,
        data_init=[0] * len(qubit_ids),
        rot_basis=True,
    )

    dataset = sample_memory_experiment(
        layout=layout,
        experiment=circuit,
        num_shots=100,
        num_rounds=10,
        seed=123,
    )

    assert set(dataset.data_vars) == {
        "ideal_anc_meas",
        "ideal_data_meas",
        "data_meas",
        "anc_meas",
    }
    assert set(dataset.coords) == {
        "qec_round",
        "seed",
        "shot",
        "anc_qubit",
        "data_qubit",
    }
    assert (dataset.qec_round.values == np.arange(1, 10 + 1)).all()
    assert (dataset.shot.values == np.arange(100)).all()
    assert (dataset.anc_qubit.values == np.array(layout.get_qubits(role="anc"))).all()
    assert (dataset.data_qubit.values == np.array(layout.get_qubits(role="data"))).all()
    assert dataset.seed == 123

    return
