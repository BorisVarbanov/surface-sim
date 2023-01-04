# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: surface-sim-kernel
#     language: python
#     name: surface-sim-kernel
# ---

# %%
import pathlib

import numpy as np
import stim
import xarray as xr
from matplotlib import pyplot as plt

from surface_sim.circuits.library import (
    log_initialization,
    log_measurement,
    sequential_qec_round,
)
from surface_sim.layouts import Layout, set_coords, surf_code_layout


# %%
def get_parity_mat(layout: Layout, stab_type: str) -> xr.DataArray:
    if stab_type not in ("x_type", "z_type"):
        raise ValueError(f"Stabilize type {stab_type} not recognized.")
    par_mat = []

    anc_qubits = layout.get_qubits(role="anc", stab_type=stab_type)
    data_qubits = layout.get_qubits(role="data")

    for anc_qubit in anc_qubits:
        nbrs = layout.get_neighbors(anc_qubit)
        anc_parity_check = [int(q in nbrs) for q in data_qubits]
        par_mat.append(anc_parity_check)

    data_arr = xr.DataArray(
        data=par_mat,
        dims=["anc_qubit", "data_qubit"],
        coords=dict(
            anc_qubit=anc_qubits,
            data_qubit=data_qubits,
        ),
        attrs=dict(stab_type=stab_type),
    )
    return data_arr


def get_init_state(layout: Layout, log_state: int) -> xr.DataArray:
    data_qubits = layout.get_qubits(role="data")
    state = np.repeat(log_state, len(data_qubits))

    init_state = xr.DataArray(
        data=state,
        dims=["data_qubit"],
        coords=dict(data_qubit=data_qubits),
        attrs=dict(log_state=log_state),
    )
    return init_state


def get_noise_channel(layout, label, channel_qubits, prob):
    noise = stim.Circuit()
    qubits = layout.get_qubits()

    inds = [qubits.index(q) for q in channel_qubits]
    noise.append(label, inds, prob)
    return noise


# %%
SCRIPT_DIR = pathlib.Path.cwd()

DATA_DIR = SCRIPT_DIR / "data"

TRAIN_DATA_DIR = DATA_DIR / "train"
TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

DEV_DATA_DIR = DATA_DIR / "dev"
DEV_DATA_DIR.mkdir(parents=True, exist_ok=True)

TEST_DATA_DIR = DATA_DIR / "test"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %%
DISTANCE = 3
LAYOUT = surf_code_layout(distance=DISTANCE)

# %%
PLOT_LAYOUT = True

if PLOT_LAYOUT:
    set_coords(LAYOUT)
    fig, ax = plt.subplots(dpi=100)
    LAYOUT.plot(axis=ax, draw_patches=True, label_qubits=False)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# # Generate the training data

# %% [markdown]
# Set the simulation parameters

# %%
LOG_BASIS = "z_basis"
LOG_STATES = [0, 1]

DATA_QUBITS = LAYOUT.get_qubits(role="data")
ANC_QUBITS = LAYOUT.get_qubits(role="anc")
NUM_ANC = len(ANC_QUBITS)
SAVE_DATA = True

NUM_ROUNDS = 20
QEC_ROUNDS = list(range(1, NUM_ROUNDS + 1))

NUM_SHOTS = 20000
SHOTS = list(range(1, NUM_SHOTS + 1))
BATCH_SIZE = None
ERROR_PROB = 0.001

# %% [markdown]
# Compile the experiment circuits

# %%
experiments = []

incoming_noise = get_noise_channel(
    LAYOUT, "DEPOLARIZE1", DATA_QUBITS + ANC_QUBITS, ERROR_PROB
)
qec_round = incoming_noise + sequential_qec_round(LAYOUT, reset=False)
log_meas = incoming_noise + log_measurement(LAYOUT, basis=LOG_BASIS, reset=False)

for log_state in LOG_STATES:
    log_init = log_initialization(LAYOUT, log_state=log_state, basis=LOG_BASIS)
    experiment = log_init + (qec_round * NUM_ROUNDS) + log_meas
    experiments.append(experiment)

# %%
qec_round

# %%
circuit = stim.Circuit.generated(
    "repetition_code:memory",
    rounds=9,
    distance=3,
    before_round_data_depolarization=0.15,
)

# %%
circuit

# %%
em = circuit.detector_error_model(decompose_errors=True)

# %%
em.num_observables

# %%
em.get_detector_coordinates()

# %%
err = em[1]

# %%
err.targets_copy()[0].val

# %% [markdown]
# run the simulation

# %%
for log_state, experiment in zip(LOG_STATES, experiments):
    sampler = experiment.compile_sampler()  # should be seeded.
    outcome_vec = sampler.sample(NUM_SHOTS)

    meas_outcomes = outcome_vec.reshape(NUM_SHOTS, -1)
    anc_outcomes, data_outcomes = np.split(
        meas_outcomes,
        [
            NUM_ROUNDS * NUM_ANC,
        ],
        axis=1,
    )
    anc_outcomes = anc_outcomes.reshape(NUM_SHOTS, NUM_ROUNDS, NUM_ANC)

    anc_meas = xr.DataArray(
        data=anc_outcomes,
        dims=["shot", "qec_round", "anc_qubit"],
        coords=dict(
            shot=SHOTS,
            qec_round=QEC_ROUNDS,
            anc_qubit=ANC_QUBITS,
        ),
    )

    data_meas = xr.DataArray(
        data=data_outcomes,
        dims=["shot", "data_qubit"],
        coords=dict(
            shot=SHOTS,
            data_qubit=DATA_QUBITS,
        ),
    )

    init_state = get_init_state(LAYOUT, log_state)

    dataset = xr.Dataset(
        data_vars=dict(anc_meas=anc_meas, data_meas=data_meas, init_state=init_state),
        attrs=dict(
            description="Distance-3 surface code experimental data.",
        ),
    )

    if SAVE_DATA:
        file_name = f"d{DISTANCE}_surf_code_seq_round_state_{log_state}_shots_{NUM_SHOTS}_rounds_{NUM_ROUNDS}_v2.nc"
        dataset.to_netcdf(DATA_DIR / file_name)

# %%
from typing import Optional

import numpy as np
import xarray as xr


def get_syndromes(anc_meas: xr.DataArray) -> xr.DataArray:
    syndromes = anc_meas ^ anc_meas.shift(qec_round=1, fill_value=0)
    syndromes.name = "syndromes"
    return syndromes


def get_defects(
    syndromes: xr.DataArray, frame: Optional[xr.DataArray] = None
) -> xr.DataArray:
    shifted_syn = syndromes.shift(qec_round=1, fill_value=0)

    if frame is not None:
        shifted_syn[dict(qec_round=0)] = frame

    defects = syndromes ^ shifted_syn
    defects.name = "defects"
    return defects


def get_final_defects(
    syndromes: xr.DataArray,
    proj_syndrome: xr.DataArray,
) -> xr.DataArray:
    last_syndrome = syndromes.isel(qec_round=-1)
    proj_anc = proj_syndrome.anc_qubit

    final_defects = last_syndrome.sel(anc_qubit=proj_anc) ^ proj_syndromes
    final_defects.name = "final_defects"
    return final_defects


# %%
syndromes = get_syndromes(dataset.anc_meas)

z_stab_frame = (
    dataset.init_state @ PAR_MAT
) % 2  # define the inital syndrome frame based on the initialized state
x_stab_frame = syndromes.sel(
    qec_round=1, anc_qubit=LAYOUT.get_qubits(role="anc", stab_type="x_type")
)
pauli_frame = xr.concat([x_stab_frame, z_stab_frame], dim="anc_qubit")

proj_syndromes = (dataset.data_meas @ PAR_MAT) % 2

defects = get_defects(syndromes, frame=pauli_frame)
final_defects = get_final_defects(syndromes, proj_syndromes)

# %%
final_defects

# %%
proj_syndromes[0]

# %%
dataset.data_meas[0]

# %%
example_anc_meas = dataset.anc_meas.sel(shot=range(1, 10001))


# %%
meas_syndromes = get_syndromes(dataset.anc_meas)
syndromes = xr.concat([meas_syndromes, proj_syndromes], dim="qec_round")

defects = get_defects(syndromes, initial_frame)

# %%
example_syndromes = get_syndromes(example_anc_meas)

# %%

# %%
circuit

import stim

# %%
from stim import TableauSimulator

# %%
t_sim = TableauSimulator()
t_sim.set_num_qubits(2)


# %%
def hadamard(qubit):
    if not state.is_leaked(qubit):
        state.h(state.index(qubit))


# %%
getattr(t_sim, "x")(0)

# %%
t_sim.state_vector()

# %%
# %%timeit
t_sim.x(0)

# %%
t_sim = TableauSimulator()
t_sim.num_qubits

# %%
t_sim.h(0, 1, 2)

# %%
t_sim.cx(0, 2, 1, 3)

# %%
t_sim.measure(0)

# %%

from surface_sim.circuits import gates

# %%
from surface_sim.states import State

# %%
qubits = ("X1", "X2")
state = State(qubits)

# %%
h = gates.Hadamard("X1", 0)

# %%
h.apply_to(state)

# %%
state.tableau.state_vector()

# %%
s = {"A", "B", "C", "D"}

# %%
s -= set(("A", "B"))

# %%
s

# %%
s.intersection(("A", "B"))


# %%
def hadamard(state: State, qubit: str) -> None:
    if qubit not in state.leaked_qubits:
        state.tableau.h(state.index(qubit))


# %%
from functools import partial

# %%
partial(hadamard, qubit="X1")

# %%
from functools import wraps


def operation(qubit):
    def function_logger(func):
        def wrapper(*args, **kwargs):
            date_time = datetime.now().strftime("%y-%m-%d %H:%M:%S")
            with open(filename, "a") as logfile:
                logfile.write(f"{f.__name__}: {args}, {date_time}\n")
            return f(*args, **kwargs)

        return wrapper

    return function_logger


def logged(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        print(func.__name__ + " was called")
        return func(*args, **kwargs)

    return with_logging


@logged
def f(x):
    return x + x * x


# %%
from functools import update_wrapper


class Operation:
    def __init__(self, func):
        update_wrapper(self, func)
        self.func = func
        self.num_calls = 0

    def __call__(self, *args, **kwargs):
        self.num_calls += 1
        print(f"Call {self.num_calls} of {self.func.__name__!r}")
        return self.func(*args, **kwargs)


@Operation
def hadamard_op(state: State, qubit: str) -> None:
    if qubit not in state.leaked_qubits:
        state.tableau.h(state.index(qubit))


# %%
class Operation(object):
    def __init__(self, ops, *qubits):
        self.ops = ops
        self.qubits = qubits

    def __call__(self, state, **pararams):
        for op in self.ops:
            op(state, qubits, **kwargs)


# %%
from surface_sim.circuits.circuit import Gate


def hadamard_op(state: State, qubit: str) -> None:
    if qubit not in state.leaked_qubits:
        state.tableau.h(state.index(qubit))


def gate_decorator(func):
    label = func.__name__

    def wrapper(qubits, time):
        print(qubits)
        # op = Operation(func, qubits)
        gate = Gate(qubits, time, label=label)
        return gate

    wrapper.__name__ = label
    return wrapper


@gate_decorator
def hadamard(qubit: str) -> None:
    return hadamard_op(qubit)


# %%
hadamard("X2", time=1)

# %%
had_op = hadamard
ops = (had_op,)

# %%
ops[0](state)

# %%
from collections import namedtuple

# %%
Operation = namedtuple("Operation", ["func", "qubits"])


# %%
def hadamard_op(state: State, qubit: str) -> None:
    ind = state.index(qubit)
    if ind not in state.leaked_inds:
        state.tableau.h(ind)


# %%
op = Operation(hadamard_op, "X2")

# %%
op.func(state, op.qubits)

# %%
from operator import itemgetter

# %%
qubits = ("X1", "X2", "X1", "X2", "X1", "X2")

# %%
# %%timeit
tuple(itemgetter(*qubits)(state.inds))

# %%
# %%timeit
(state.inds[qubit] for qubit in qubits)
