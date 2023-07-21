"""Module implement circuits for experiments."""
from typing import List
from stim import Circuit

from ..circuits.rep_code import init_qubits, log_meas, qec_round
from ..models import Model


def memory_experiment(model: Model, num_rounds: int, data_init: List[int]) -> Circuit:
    """
    memory_experiment Constructs a circuit for a memory experiment.

    Parameters
    ----------
    model : Model
        The error model used to generate the circuit.
    num_rounds : int
        The number of rounds of error correction to perform.
    data_init : NDArray
        The initial state of the data qubits.
    meas_reset : bool, optional
        Whether to reset the qubits after measurements , by default False

    Returns
    -------
    Circuit
        The circuit for the memory experiment.

    Raises
    ------
    ValueError
        If num_rounds is not a positive integer.
    ValueError
        If num_rounds is not an integer.
    """
    if not isinstance(num_rounds, int):
        raise ValueError(f"num_rounds expected as int, got {type(num_rounds)} instead.")

    if num_rounds < 0:
        raise ValueError("num_rounds needs to be a positive integer")

    init = init_qubits(model, data_init)

    if num_rounds == 0:
        meas = log_meas(model)
        experiment = init + meas
        return experiment

    init_round = qec_round(model)

    if num_rounds == 1:
        meas = log_meas(model, comp_rounds=1)
        experiment = init + init_round + meas
        return experiment

    meas = log_meas(model, comp_rounds=2)
    init_rounds = init_round * 2

    if num_rounds == 2:
        experiment = init + init_rounds + meas
        return experiment

    sub_round = qec_round(model, comp_rounds=2)
    sub_rounds = sub_round * (num_rounds - 2)

    experiment = init + init_rounds + sub_rounds + meas
    return experiment
