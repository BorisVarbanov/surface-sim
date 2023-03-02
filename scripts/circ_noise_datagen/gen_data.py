# %%
import pathlib
from typing import List, Union

import numpy as np
from matplotlib import pyplot as plt
from qec_util.layouts import Layout, plot, set_coords

from surface_sim import Setup
from surface_sim.experiments import memory_exp
from surface_sim.models import CircuitNoiseModel
from surface_sim.util import sample_experiment

# %%
EXP_DIR = pathlib.Path.cwd()

CONFIG_DIR = EXP_DIR / "config"
if not CONFIG_DIR.exists():
    raise ValueError("Layout directory does not exist.")

# %%
LAYOUT_FILE = "d3_rotated_layout.yaml"
SETUP_FILE = "circ_level_noise.yaml"

layout = Layout.from_yaml(CONFIG_DIR / LAYOUT_FILE)
set_coords(layout)
setup = Setup.from_yaml(CONFIG_DIR / SETUP_FILE)
model = CircuitNoiseModel(setup, layout)

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
LIST_NUM_ROUNDS: List[int] = list(range(1, 3 + 1, 2))  # Number of rounds
NUM_SHOTS: int = 100  # Number of shots
ROT_BASIS: bool = False  # In the z-basis
MEAS_RESET: bool = False  # No resets following measurements

# Variable parameters
LOG_STATES: List[int] = [0, 1]  # Logical state(s)
DEPOL_PROBS: List[float] = [1e-2]

# %%
root_seed_sequence = np.random.SeedSequence(ROOT_SEED)

num_probs = len(DEPOL_PROBS)
num_runs = len(LIST_NUM_ROUNDS) * num_probs

global_seeds = iter(root_seed_sequence.generate_state(num_runs, dtype="uint64"))

distance = layout.distance
basis = "X" if ROT_BASIS else "Z"
num_states = len(LOG_STATES)

for prob in DEPOL_PROBS:
    model.setup.set_var_param("prob", prob)

    for num_rounds in LIST_NUM_ROUNDS:
        print(num_rounds, end="\r")

        seed_sequence = np.random.SeedSequence(next(global_seeds))
        seeds = iter(seed_sequence.generate_state(num_states, dtype="uint64"))

        for log_state in LOG_STATES:
            exp_name = f"surf-code_d{distance}_b{basis}_s{log_state}_n{NUM_SHOTS}_r{num_rounds}_p{prob:.3f}"

            exp_folder = EXP_DIR / DATASET_TYPE / exp_name
            exp_folder.mkdir(parents=True, exist_ok=True)

            experiment = memory_exp(
                model=model,
                num_rounds=num_rounds,
                log_state=log_state,
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
            dataset = dataset.assign_coords(
                log_state=log_state,
                rot_basis=int(ROT_BASIS),
                meas_reset=int(MEAS_RESET),
                error_prob=prob,
            )
            dataset.attrs["seed"] = int(seed)
            dataset.to_netcdf(exp_folder / "measurements.nc")

            error_model = experiment.detector_error_model(
                decompose_errors=True,
                allow_gauge_detectors=True,
            )
            error_model.to_file(exp_folder / "detector_error_model.dem")


# %%
