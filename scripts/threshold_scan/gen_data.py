# %%
from pathlib import Path
from itertools import product, repeat
from typing import Union

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from qec_util.layouts import rot_surf_code

from surface_sim import Setup
from surface_sim.experiments.css_code import memory_experiment
from surface_sim.models import CircuitNoiseModel
from surface_sim.util import sample_experiment

# %%
EXP_DIR: Path = Path.cwd()
CONFIG_DIR: Path = EXP_DIR / "config"

# %%
SETUP_FILE = "circ_level_noise.yaml"
setup = Setup.from_yaml(CONFIG_DIR / SETUP_FILE)


# %% [markdown]
# # Generate the training data

# %%
DATASET_TYPE: str = "test"  # Possible types are "train", "dev" and "test"

# Fixed parameters
ROOT_SEED: Union[int, None] = np.random.randint(999999)  # Initial seed for the RNG
NUM_SHOTS: int = 20000  # Number of shots
ROT_BASIS: bool = False  # In the z-basis
MEAS_RESET: bool = False  # No resets following measurements
GAUGE_DETECTORS: bool = True  # Add gauge detectors

# Variable parameters
DISTANCES = [3, 7, 9, 11]
LOG_STATES = [0, 1]
DEPOL_PROBS = np.linspace(1e-2, 1e-3, 10)

# %%
root_seed_sequence = np.random.SeedSequence(ROOT_SEED)

run_parameters = tuple(product(DISTANCES, DEPOL_PROBS))
num_runs = len(run_parameters)

global_seeds = iter(root_seed_sequence.generate_state(num_runs, dtype="uint64"))

basis = "X" if ROT_BASIS else "Z"
num_states = len(LOG_STATES)

for distance, prob in run_parameters:
    print(f"Distance: {distance}, Error prob: {prob:.3f}")
    setup.set_var_param("prob", prob)

    layout = rot_surf_code(distance=distance)
    model = CircuitNoiseModel(setup, layout)

    data_qubits = layout.get_qubits(role="data")
    num_data = len(data_qubits)
    num_rounds = distance

    seed_sequence = np.random.SeedSequence(next(global_seeds))
    seeds = iter(seed_sequence.generate_state(num_states, dtype="uint64"))

    for state in LOG_STATES:
        data_init = list(repeat(bool(state), num_data))

        exp_name = f"surf-code_d{distance}_b{basis}_s{state}_n{NUM_SHOTS}_r{num_rounds}_p{prob:.3f}"

        exp_folder = EXP_DIR / DATASET_TYPE / exp_name
        exp_folder.mkdir(parents=True, exist_ok=True)

        experiment = memory_experiment(
            model=model,
            num_rounds=num_rounds,
            data_init=data_init,
            rot_basis=ROT_BASIS,
            meas_reset=MEAS_RESET,
            gauge_detectors=GAUGE_DETECTORS,
        )

        experiment.to_file(exp_folder / "circuit.stim")

        seed = next(seeds)
        dataset = sample_experiment(
            layout=layout,
            experiment=experiment,
            seed=seed,
            num_shots=NUM_SHOTS,
            num_rounds=num_rounds,
        )

        # assign these as coordinate for merging datasets later on. Add here any otther relevant parameters
        dataset["data_init"] = xr.DataArray(data_init, dims=["data_qubit"])
        dataset = dataset.assign_coords(
            rot_basis=ROT_BASIS,
            meas_reset=MEAS_RESET,
            distance=distance,
            error_prob=prob,
        )
        dataset.assign_attrs(seed=seed)
        dataset.to_netcdf(exp_folder / "measurements.nc")

        error_model = experiment.detector_error_model(
            decompose_errors=True,
            allow_gauge_detectors=GAUGE_DETECTORS,
        )
        error_model.to_file(exp_folder / "detector_error_model.dem")

# %%
