from typing import Optional

import numpy as np

from ..utils.constants import FPI, E2
from ..representations import EnvironDensity
from ..cores import CoreContainer
from ..physical import (
    EnvironDielectric,
    EnvironElectrolyte,
    EnvironSemiconductor,
)
from . import DirectSolver, IterativeSolver


class FixedPointSolver(IterativeSolver):
    """
    Fixed point iteration solver for the Poisson equation.
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

    @IterativeSolver.charge_operation
    def generalized(
        self,
        density: EnvironDensity,
        dielectric: EnvironDielectric,
        electrolyte: EnvironElectrolyte = None,
        semiconductor: EnvironSemiconductor = None,
    ) -> EnvironDensity:
        """docstring"""
        grid = density.grid
        rhoiter = dielectric.iterative
        rhotot = dielectric.density
        eps = dielectric.epsilon
        gradlog = dielectric.gradlogepsilon

        rhozero = EnvironDensity(grid, np.array((1 - eps) * density / eps))
        residuals = EnvironDensity(grid)
        gradpoisson = EnvironDensity(grid)

        for _ in range(self.maxiter):
            rhotot[:] = density + rhoiter + rhozero

            gradpoisson[:] = self.direct.grad_poisson(
                rhotot,
                electrolyte,
                semiconductor,
            )

            residuals[:] = gradlog.scalar_product(gradpoisson) / FPI / E2 - rhoiter
            rhoiter[:] += self.mixing * residuals

            if residuals.euclidean_norm() < self.tol: break

        else:
            raise ValueError('The fixed point iteration did not converge')

        rhotot[:] = density + rhoiter + rhozero

        potential = self.direct.poisson(
            rhotot,
            electrolyte,
            semiconductor,
        )

        rhotot[:] = rhozero + rhoiter

        return potential
