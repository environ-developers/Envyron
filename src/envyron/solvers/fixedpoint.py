# Using fixed point iteration to solve the poisson equation
# Using class and inheritance to implement the fixed point iteration
from typing import Optional
import numpy as np

from ..utils.constants import FPI
from ..domains import EnvironGrid
from ..physical import EnvironDielectric
from ..physical import EnvironElectrolyte
from ..physical import EnvironSemiconductor
from ..representations import EnvironDensity, EnvironGradient
from ..cores import CoreContainer
from ..solvers import DirectSolver
from .iterative import IterativeSolver


class FixedPointSolver(IterativeSolver):
    """
    Fixed point iteration solver for the Poisson equation
    Uses the FixedPointSolver class to implement the fixed point iteration
    We use rho, epsilon, the gradient of log epsilon and the mixing paraetr
    """

    def __init__(
        self,
        cores: CoreContainer,
        direct: DirectSolver,
        maxiter: Optional[int] = 100,
        tol: Optional[float] = 1.0e-10,
        auxiliary: Optional[str] = 'full',
        mixing: Optional[float] = 0.6,
    ) -> None:
        super().__init__(cores, direct, maxiter, tol, auxiliary)
        self.mixing = mixing

    def generalized(
        self,
        density: EnvironDensity,
        dielectric: EnvironDielectric,
        electrolyte: EnvironElectrolyte = None,
        semiconductor: EnvironSemiconductor = None,
    ) -> EnvironDensity:
        """docstring"""
        grid = dielectric.epsilon.grid
        # Initialize the iterative solver

        polarization_iter = EnvironDensity(grid)
        polarization_fixed = (
            1 - dielectric.epsilon) * density / dielectric.espsilon
        polarization_new = EnvironDensity(grid)

        for iteration in range(self.maxiter):

            density_total = density + polarization_iter + polarization_fixed

            #Compute the electrostatic field from the total charge density

            electrostatic_field = self.direct.grad_poisson(
                density_total, electrolyte, semiconductor)

            polarization_new[:] = np.einsum('lijk, lijk->ijk',
                                         dielectric.gradlogepsilon,
                                         electrostatic_field / FPI)

            residuals: EnvironDensity = (self.mixing - 1) * (polarization_iter - polarization_new)
            polarization_iter[:] = polarization_iter + residuals

            #Check the convergence of the fixed point iteration

            if residuals.euclidean_norm() < self.tol:
                density_total = density + polarization_iter + polarization_fixed
                return self.direct.poisson(density_total,electrolyte,semiconductor)
            else:
                raise ValueError('The fixed point iteration did not converge')
