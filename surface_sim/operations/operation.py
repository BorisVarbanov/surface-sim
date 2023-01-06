from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Iterable, List

from stim import CircuitInstruction


class BaseOperation(ABC):
    @property
    @abstractmethod
    def units(self) -> List[Operation]:
        pass

    def join(self, other: BaseOperation) -> ChainOperation:
        return ChainOperation(self, other)


class Operation(BaseOperation):
    name: str
    args: Iterable[float]

    def __init__(self, name: str, *args: Iterable[float]) -> None:
        self._name = name
        self._args = list(args)

    def to_instruction(self, inds: List[int]) -> CircuitInstruction:
        instruction = CircuitInstruction(self.name, inds, self.args)
        return instruction

    @property
    def units(self) -> List[Operation]:
        return [self]

    @property
    def name(self) -> str:
        return self._name

    @property
    def args(self) -> List[Any]:
        return self._args

    def __repr__(self) -> str:
        if self._args:
            arg_str = ",".join(map(str, self._args))
            return f"{self._name}({arg_str})"
        return f"{self._name}"


class ChainOperation(BaseOperation):
    def __init__(self, *operations: Iterable[Operation]) -> None:
        self._units = list(chain.from_iterable(op.units for op in operations))

    @property
    def units(self) -> List[Operation]:
        return self._units

    def __repr__(self) -> str:
        ops_repr = ",".join(repr(unit_op) for unit_op in self._units)
        return f"Chain[{ops_repr}]"
