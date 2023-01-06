from typing import Tuple

from ..operations import Operation
from ..operations import library as ops
from ..setup import Setup
from .model import Model, gate


class CircuitNoiseModel(Model):
    def __init__(self, setup: Setup) -> None:
        super().__init__(setup)

    @gate
    def hadamard(self, qubit: str) -> Tuple[Operation, Operation]:
        hadamard = ops.hadamard()
        error_prob = self.setup.param("sq_error_prob", qubit)
        depol_noise = ops.depol_channel(error_prob)
        return (hadamard, depol_noise)

    @gate
    def cphase(self, stat_qubit: str, flux_qubit: str) -> Tuple[Operation, Operation]:
        cphase = ops.cphase()
        error_prob = self.setup.param("sq_error_prob", stat_qubit, flux_qubit)
        depol_noise = ops.depol_channel(error_prob, num_qubits=2)
        return (cphase, depol_noise)

    @gate
    def measure(self, qubit: str) -> Tuple[Operation, Operation]:
        meas = ops.measure()
        error_prob = self.setup.param("meas_error_prob", qubit)
        bitflip_noise = ops.bitflip_channel(error_prob)
        return (bitflip_noise, meas)
