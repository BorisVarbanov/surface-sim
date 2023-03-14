# %%
import pathlib
from typing import List, Union
from itertools import repeat

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from qec_util.layouts import Layout, plot, set_coords

from surface_sim import Setup
from surface_sim.experiments.css_code import memory_exp
from surface_sim.models import BiasedCircuitNoiseModel
from surface_sim.util import sample_experiment

# %%
EXP_DIR = pathlib.Path.cwd()

CONFIG_DIR = EXP_DIR / "config"
if not CONFIG_DIR.exists():
    raise ValueError("Layout directory does not exist.")

# %%
LAYOUT_FILE = "d3_rotated_layout.yaml"
SETUP_FILE = "biased_circ_level_noise.yaml"

layout = Layout.from_yaml(CONFIG_DIR / LAYOUT_FILE)
set_coords(layout)
setup = Setup.from_yaml(CONFIG_DIR / SETUP_FILE)
model = BiasedCircuitNoiseModel(setup, layout)

# %%
fig, ax = plt.subplots()
_ = plot(layout, axis=ax)
plt.show()

# %% [markdown]
# # Generate the training data

# %%
DATASET_TYPE: str = "test"  # Possible types are "train", "dev" and "test"

# Fixed parameters
ROOT_SEED: Union[int, None] = np.random.randint(999999)  # Initial seed for the RNG
LIST_NUM_ROUNDS: List[int] = list(range(1, 21, 2))  # Number of rounds
NUM_SHOTS: int = 20000  # Number of shots
ROT_BASIS: bool = False  # In the z-basis
MEAS_RESET: bool = False  # No resets following measurements
ERROR_PROBS: float = 1e-3
BIAS_FACTOR: float = 1
BIAS_PAULI: str = "X"

# Variable parameters
data_qubits = layout.get_qubits(role="data")
num_data_qubits = len(data_qubits)

DATA_INITS: List[List[int]] = [
    list(repeat(state, num_data_qubits)) for state in (0, 1)
]  # Logical state(s)

# %%
root_seed_sequence = np.random.SeedSequence(ROOT_SEED)
num_runs = len(LIST_NUM_ROUNDS)
global_seeds = iter(root_seed_sequence.generate_state(num_runs, dtype="uint64"))

distance = layout.distance
basis = "X" if ROT_BASIS else "Z"

model.setup.set_var_param("prob", ERROR_PROBS)
model.setup.set_var_param("bias_factor", BIAS_FACTOR)
model.setup.set_var_param("bias_pauli", BIAS_PAULI)

for num_rounds in LIST_NUM_ROUNDS:
    print(num_rounds, end="\r")

    num_experiments = len(DATA_INITS)
    seed_sequence = np.random.SeedSequence(next(global_seeds))
    seeds = iter(seed_sequence.generate_state(num_experiments, dtype="uint64"))

    for data_init in DATA_INITS:
        init_str = "".join(map(str, data_init))
        exp_name = f"surf-code_d{layout.distance}_b{basis}_s{init_str}_n{NUM_SHOTS}_r{num_rounds}"

        exp_folder = EXP_DIR / DATASET_TYPE / exp_name
        exp_folder.mkdir(parents=True, exist_ok=True)

        experiment = memory_exp(
            model=model,
            num_rounds=num_rounds,
            data_init=data_init,
            rot_basis=ROT_BASIS,
            meas_reset=MEAS_RESET,
        )

        experiment.to_file(exp_folder / "circuit.stim")

        seed = next(seeds)
        dataset = sample_experiment(
            layout,
            experiment,
            seed=seed,
            num_shots=NUM_SHOTS,
            num_rounds=num_rounds,
        )

        # assign these as coordinate for merging datasets later on. Add here any otther relevant parameters
        dataset["data_init"] = xr.DataArray(
            np.array(data_init, dtype=bool), dims=["data_qubit"]
        )
        dataset = dataset.assign_coords(
            rot_basis=ROT_BASIS,
            meas_reset=MEAS_RESET,
        )
        dataset.assign_attrs(seed=seed)
        dataset.to_netcdf(exp_folder / "measurements.nc")

        error_model = experiment.detector_error_model(
            decompose_errors=True,
            allow_gauge_detectors=True,
            approximate_disjoint_errors=True,
        )
        error_model.to_file(exp_folder / "detector_error_model.dem")


# %%
