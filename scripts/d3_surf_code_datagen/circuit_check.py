# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: surface-sim-kernel
#     language: python
#     name: surface-sim-kernel
# ---

# %%
import yaml
import pathlib

from copy import copy
from stim import Circuit

# %%
NOTEBOOK_DIR = pathlib.Path.cwd()

SCHEDULES_DIR = NOTEBOOK_DIR / "schedules"

# %%
with open(SCHEDULES_DIR / "pipelined_qec_round.yaml") as file:
    schedule = yaml.safe_load(file)

# %%
circ_layers = copy(schedule.get("layers"))

# %%
from stim import Circuit

# %%
qubits = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "X1", "X2", "X3", "X4", "Z1", "Z2", "Z3", "Z4"]

OPERATION_LABELS = dict(hadamard = "H", cphase = "CZ", measure = 'M')
qubit_times = {qubit : 0.0 for qubit in qubits}

circuit = Circuit()

for layer in circ_layers:
    for gate in layer.get("gates"):
        gate_label = gate.get("label")
        op_label = OPERATION_LABELS[gate_label]
        
        n_qubits = gate.get("num_qubits")
        gate_qubits = gate.get("qubits")
        qubit_inds = []
        
        gate_time = gate.get("time")
        
        if n_qubits == 1:
            for qubit in gate_qubits:
                if qubit not in qubits:
                    raise ValueError(f"Qubit {qubit} not in provided qubit list.")
                
                qubit_inds.append(qubits.index(qubit))
                qubit_times[qubit] += gate_time
            
        elif n_qubits == 2:
            for ctrl_q, target_q in gate_qubits:
                if ctrl_q not in qubits or target_q not in qubits:
                    raise ValueError(
                        f"Qubit(s) {ctrl_q}, {target_q}  not in layout."
                    )

                #if target_q not in layout.get_neighbors(ctrl_q):
                #    raise ValueError(
                #        f"Qubit {target_q} not coupled to {ctrl_q} in layout."
                #    )
                
                for qubit in (ctrl_q, target_q):
                    qubit_inds.append(qubits.index(qubit))
                    qubit_times[qubit] += gate_time

        else:
            raise ValueError(
                f"Unexpected number of qubits ({n_qubits}) for gate {gate_label}"
            )
            
        circuit.append_operation(op_label, qubit_inds)
        
    circuit.append("TICK")

# %%
circuit

# %%
