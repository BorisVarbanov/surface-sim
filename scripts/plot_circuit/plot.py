# %%
import pathlib
from itertools import repeat, chain

from matplotlib import pyplot as plt

from qec_util.layouts import Layout
from surface_sim import Setup
from surface_sim.circuits.xzzx_code import qec_round
from surface_sim.circuits import plot_layer
from surface_sim.models import CircuitNoiseModel

# %%
EXP_DIR = pathlib.Path.cwd()
CONFIG_DIR = EXP_DIR / "config"

# %%
SETUP_FILE = "circ_level_noise.yaml"
LAYOUT_FILE = "d3_rotated_layout.yaml"

setup = Setup.from_yaml(CONFIG_DIR / SETUP_FILE)
layout = Layout.from_yaml(CONFIG_DIR / LAYOUT_FILE)
model = CircuitNoiseModel(setup, layout)

# %%
NUM_ROUNDS = 1
STATE = 0
BASIS = "Z"
MEAS_RESET = False
ERROR_PROB = 0.001

# %%
rot_basis = BASIS == "X"

data_qubits = layout.get_qubits(role="data")
data_init = list(repeat(STATE, len(data_qubits)))

model.setup.set_var_param("prob", ERROR_PROB)

# %%
round_circuit = qec_round(model=model, meas_reset=MEAS_RESET)
noiseless_circuit = round_circuit.without_noise()

# %%
SKIPPED_INSTRUCTIONS = {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE"}

fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(18, 6), dpi=200)

ax_iter = chain.from_iterable(axes)
for instruction in noiseless_circuit:
    if instruction.name not in SKIPPED_INSTRUCTIONS:
        axis = next(ax_iter)
        plot_layer(instruction, layout, axis=axis)
plt.show()

# %%
