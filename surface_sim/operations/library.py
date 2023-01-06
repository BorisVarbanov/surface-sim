from typing import List, Optional, Union

from .operation import Operation


def hadamard() -> Operation:
    return Operation("H")


def cphase() -> Operation:
    return Operation("CZ")


def measure(
    rotate_basis: bool = False, error_prob: Optional[float] = None
) -> Operation:
    label = "MX" if rotate_basis else "MZ"
    if error_prob:
        return Operation(label, error_prob)
    return Operation(label)


def reset(rotate_basis: bool = False) -> Operation:
    label = "RX" if rotate_basis else "RZ"
    return Operation(label)


def depol_channel(
    error_prob: Union[float, List[float]], num_qubits: int = 1
) -> Operation:
    if num_qubits not in (1, 2):
        raise ValueError("num_qubits greater than 2 not supported.")

    if isinstance(error_prob, float):
        label = f"DEPOLARIZE{num_qubits}"

    elif isinstance(error_prob, list):
        label = f"PAULI_CHANNEL_{num_qubits}"
    else:
        raise ValueError(
            f"error_prob expected as float or List[floa], instead got {type(error_prob)}"
        )

    return Operation(label, error_prob)


def bitflip_channel(error_prob: float) -> Operation:
    return Operation("X_ERROR", error_prob)
