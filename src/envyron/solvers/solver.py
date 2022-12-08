from abc import ABC

from ..representations import EnvironDensity
from ..cores import NumericalCore


class Solver(ABC):
    """
    An Electrostatic Solver.
    """

    def __init__(self, core: NumericalCore) -> None:
        self.core = core

    def solve(self, density: EnvironDensity) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()
