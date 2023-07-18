"""Module implement circuits for experiments."""
from typing import List
from stim import Circuit

from ..circuits.rep_code import init_qubits, log_meas, qec_round
from ..models import Model


def memory_experiment(
    model: Model,
    num_rounds: int,
    data_init: List[int],
    meas_reset: bool = False,
) -> Circuit:
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

    if num_rounds <= 0:
        raise ValueError("num_rounds needs to be a positive integer")

    num_init_rounds = 1 if meas_reset else 2

    init = init_qubits(model, data_init)
    meas = log_meas(model, meas_reset)

    init_qec_round = qec_round(model, meas_reset, meas_comparison=False)

    if num_rounds > num_init_rounds:
        init_rounds = init_qec_round * num_init_rounds

        sub_qec_round = qec_round(model, meas_reset)
        sub_rounds = sub_qec_round * (num_rounds - num_init_rounds)

        rep_checks = init_rounds + sub_rounds

    else:
        rep_checks = init_qec_round * min(num_rounds, num_init_rounds)

    experiment = init + rep_checks + meas

    return experiment
