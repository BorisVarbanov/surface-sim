from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

from ..circuits import Gate
from ..operations import BaseOperation, ChainOperation
from ..setup import Setup


class Model:
    def __init__(self, setup: Setup) -> None:
        self.setup = setup


def gate(func: Callable) -> Callable:
    def wrapper(
        model: Model,
        *qubits: Iterable[str],
        time: int = 0,
        name: Optional[str] = None,
        **params,
    ) -> Gate:
        operation = func(model, *qubits, **params)

        if isinstance(operation, Sequence):
            operation = ChainOperation(*operation)
        if not isinstance(operation, BaseOperation):
            raise ValueError(
                "gate wrapper expects a function that outputs an Operation or Sequence[Operation]."
            )
        name = name or func.__name__
        return Gate(operation, qubits, time, name)

    return wrapper
