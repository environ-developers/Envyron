from .solver import ElectrostaticSolver
from ..cores import CoreContainer

from ..representations import EnvironDensity, EnvironGradient
from ..physical import EnvironCharges

from dftpy.functional.hartree import Hartree

class DirectSolver(ElectrostaticSolver):
    """
    docstring
    """

    def __init__(self, cores: CoreContainer, core_method='none') -> None:
        super().__init__(cores)
        self.corrections_method = core_method

    @ElectrostaticSolver.charge_operation
    def poisson(self, density: EnvironDensity, *args, **kwargs) -> EnvironDensity:
        res = Hartree.compute(density=density, calcType={"V"}).potential

        # Hartree to Rydberg
        return 2.*res
