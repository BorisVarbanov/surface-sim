from typing import Any

from ..layouts import Layout
from ..setup import Setup


class Model(object):
    def __init__(self, setup: Setup, layout: Layout) -> None:
        self._setup = setup
        self._layout = layout

    @property
    def setup(self) -> Setup:
        return self._setup

    @property
    def layout(self) -> Layout:
        return self._layout

    def param(self, *qubits: str) -> Any:
        return self._setup.param(*qubits)
