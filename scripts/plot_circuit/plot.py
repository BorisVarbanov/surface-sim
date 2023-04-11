# %%
import pathlib
from itertools import chain, repeat

from matplotlib import pyplot as plt
from qec_util.layouts import Layout

from surface_sim import Setup
from surface_sim.circuits import plot_layer
from surface_sim.circuits.css_code import init_qubits, log_meas, qec_round
from surface_sim.models import CircuitNoiseModel

# %%
NOTEBOOK_DIR = pathlib.Path.cwd()
CONFIG_DIR = NOTEBOOK_DIR / "config"

IMG_DIR = NOTEBOOK_DIR / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %%
SETUP_FILE = "circ_level_noise.yaml"
LAYOUT_FILE = "d3_rotated_layout.yaml"

setup = Setup.from_yaml(CONFIG_DIR / SETUP_FILE)
layout = Layout.from_yaml(CONFIG_DIR / LAYOUT_FILE)
model = CircuitNoiseModel(setup, layout)

# %%
NUM_ROUNDS = 1
STATE = 1
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

image_name = "qec_circuit"
for file_format in ("pdf", "png"):
    full_name = f"{image_name}.{file_format}"
    fig.savefig(
        IMG_DIR / full_name,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        format=file_format,
        pad_inches=0,
    )

plt.show()

# %%
data_init = list(repeat(1, len(data_qubits)))
init_circ = init_qubits(model, data_init, rot_basis=rot_basis)
noiseless_circuit = init_circ.without_noise()
# %%
fig, axes = plt.subplots(ncols=2, figsize=(6, 6), dpi=200)

ax_iter = chain(axes)
for instruction in noiseless_circuit:
    if instruction.name not in SKIPPED_INSTRUCTIONS:
        axis = next(ax_iter)
        plot_layer(instruction, layout, axis=axis)


image_name = "init_circ"
for file_format in ("pdf", "png"):
    full_name = f"{image_name}.{file_format}"
    fig.savefig(
        IMG_DIR / full_name,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        format=file_format,
        pad_inches=0,
    )

plt.show()
# %%


# %%
meas_circ = log_meas(model, meas_reset=True, rot_basis=rot_basis)
noiseless_circuit = meas_circ.without_noise()

fig, axis = plt.subplots(figsize=(3, 3), dpi=200)

for instruction in noiseless_circuit:
    if instruction.name not in SKIPPED_INSTRUCTIONS:
        plot_layer(instruction, layout, axis=axis)

image_name = "log_meas"
for file_format in ("pdf", "png"):
    full_name = f"{image_name}.{file_format}"
    fig.savefig(
        IMG_DIR / full_name,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        format=file_format,
        pad_inches=0,
    )

plt.show()
# %%
