from itertools import chain, count

from ..circuits import Circuit
from ..layouts import Layout
from ..models import Model

# Gate order as defined in the Versluis et. al. paper
GATE_ORDERS = dict(
    x_type=["north_east", "north_west", "south_east", "south_west"],
    z_type=["north_east", "south_east", "north_west", "south_west"],
)


def log_measurement(
    model: Model,
    layout: Layout,
    rot_basis: bool = False,
) -> Circuit:
    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")

    gates = []
    time_counter = count()

    if rot_basis:
        time = next(time_counter)
        for qubit in data_qubits:
            gate = model.hadamard(qubit)
            gates.append(gate.shift(time))

    time = next(time_counter)
    for qubit in data_qubits:
        measurement = model.measure(qubit)
        gates.append(measurement.shift(time))

    qubits = list(chain(data_qubits, anc_qubits))
    circuit = Circuit(qubits, gates)
    return circuit


def qec_round(model: Model, layout: Layout) -> Circuit:
    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")

    gates = []
    time_counter = count()

    time = next(time_counter)
    for stab_type, gate_order in GATE_ORDERS.items():
        stab_qubits = layout.get_qubits(role="anc", stab_type=stab_type)

        if stab_type == "x_type":
            rot_qubits = list(chain(data_qubits, stab_qubits))
        else:
            rot_qubits = stab_qubits

        for qubit in rot_qubits:
            gate = model.hadamard(qubit)
            gates.append(gate.shift(time))

        for ord_dir in gate_order:
            time = next(time_counter)
            for anc_q in stab_qubits:
                neighbors = layout.get_neighbors(anc_q, direction=ord_dir)
                for data_q in neighbors:
                    if layout.param("freq_group", data_q) == "high":
                        qubit_pair = [anc_q, data_q]
                    else:
                        qubit_pair = [data_q, anc_q]
                    cz_gate = model.cphase(*qubit_pair)
                    gates.append(cz_gate.shift(time))

        time = next(time_counter)
        for qubit in rot_qubits:
            gate = model.hadamard(qubit)
            gates.append(gate.shift(time))

    time = next(time_counter)
    for qubit in anc_qubits:
        measurement = model.measure(qubit)
        gates.append(measurement.shift(time))

    qubits = list(chain(data_qubits, anc_qubits))
    circuit = Circuit(qubits, gates)
    return circuit
