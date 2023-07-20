from typing import Tuple
from math import exp, inf


def pure_dephasing_time(relax_time: float, deph_time: float) -> float:
    """
    pure_dephasing_time Calculates the pure dephasing time of a qubit given
    the relaxation time and dephasing time.

    Parameters
    ----------
    relax_time : float
        The relaxation time (T_1) of the qubit
    deph_time : float
        The dephasing time (T_2) of the qubit

    Returns
    -------
    float
        The pure dephasing time of the qubit

    Raises
    ------
    ValueError
        If the dephasing time is more than twice the relaxation time
    """
    if deph_time == 2 * relax_time:
        return inf

    relax_rate = 1 / relax_time
    deph_rate = 1 / deph_time

    pure_deph_rate = deph_rate - 0.5 * relax_rate
    pure_deph_time = 1 / pure_deph_rate
    if pure_deph_time < 0:
        raise ValueError(
            f"Unphysical dephasing time given T1 of {relax_time} and T2 of {deph_time}"
        )
    return pure_deph_time


def twirled_amp_phase_damping(
    relax_time: float, deph_time: float, duration: float
) -> Tuple[float]:
    """
    twirled_amp_phase_damping Returns the probabilities of X, Y, and Z errors
    for a Pauli-twirled amplitude and phase damping channel.

    Parameters
    ----------
    relax_time : float
        The relaxation time (T_1) of the qubit
    deph_time : float
        The dephasing time (T_2) of the qubit
    duration : float
        The duration of the amplitude-phase damping period

    Returns
    -------
    Tuple[float]
        The probabilities of X, Y, and Z errors
    """
    relax_prob = 1 - exp(-duration / relax_time)
    deph_prob = 1 - exp(-duration / deph_time)

    # pure_deph_time = pure_dephasing_time(relax_time, deph_time)
    # pure_deph_prob = 1 - exp(-duration / pure_deph_time)

    x_prob = relax_prob / 4
    y_prob = relax_prob / 4
    z_prob = deph_prob / 2 - relax_prob / 4

    # z_prob = 1 / 2 - relax_prob / 4 - (sqrt(1 - relax_prob) * (1 - pure_deph_prob)) / 2

    return x_prob, y_prob, z_prob
