from .solver import ElectrostaticSolver
from ..cores import CoreContainer

from ..representations import EnvironDensity, EnvironGradient
from ..physical import EnvironCharges


class DirectSolver(ElectrostaticSolver):
    """
    docstring
    """

    def __init__(self, cores: CoreContainer, core_method='none') -> None:
        super().__init__(cores)
        self.corrections_method = core_method

    def poisson_density(self, density: EnvironDensity) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    def poisson_charges(self, charges: EnvironCharges) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    def grad_poisson_density(self, density: EnvironDensity) -> EnvironGradient:
        """docstring"""
        raise NotImplementedError()

    def grad_poisson_charges(self, charges: EnvironCharges) -> EnvironGradient:
        """docstring"""
        raise NotImplementedError()
