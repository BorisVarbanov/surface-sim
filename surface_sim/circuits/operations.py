from ..states import State
import inspect
from typing import Any, Tuple, Callable


def hadamard(state: State, ind: int) -> None:
    if ind not in state.leaked_inds:
        state.tableau.h(state.index(ind))


def measure(state: State, ind: int) -> int:
    if ind in state.leaked_inds:
        return 2
    return state.tableau.measure(state.index(ind))


def cphase(state: State, ctrl_ind: int, tar_ind: int) -> None:
    ctrl_ind = state.index(ctrl_ind)
    tar_ind = state.index(tar_ind)

    if ctrl_ind not in state.leaked_inds:
        if tar_ind not in state.leaked_inds:
            state.tableau.cz(ctrl_ind, tar_ind)


def reset(state: State, ind: int) -> None:
    state.tableau.reset(state.index(ind))
    state.leaked_inds.discard(ind)


def x_gate(state: State, ind: int) -> None:
    if ind not in state.leaked_inds:
        state.tableau.x(state.index(ind))


def y_gate(state: State, ind: int) -> None:
    if ind not in state.leaked_inds:
        state.tableau.y(state.index(ind))


def z_gate(state: State, ind: int) -> None:
    if ind not in state.leaked_inds:
        state.tableau.x(state.index(ind))
