from typing import Optional, overload

from multipledispatch import dispatch

import numpy as np

from .iterative import IterativeSolver
from ..cores import CoreContainer
from ..representations import EnvironDensity
from ..physical import (
    EnvironDielectric,
    EnvironElectrolyte,
    EnvironSemiconductor,
    EnvironCharges,
)


class GradientSolver(IterativeSolver):
    """docstring"""

    def __init__(
        self,
        cores: CoreContainer,
        preconditioner: str = "sqrt",
        conjugate: Optional[bool] = True,
        tol: Optional[float] = 1.0e-7,
        maxiter: Optional[int] = 100,
        verbosity: Optional[int] = 1,
    ) -> None:
        super().__init__(cores=cores, maxiter=maxiter, tol=tol)
        self.preconditioner = preconditioner
        self.conjugate = conjugate
        self.verbosity = verbosity

    @overload
    @dispatch(
        EnvironDensity,
        EnvironDielectric,
        EnvironElectrolyte,
        EnvironSemiconductor,
    )
    def generalized(
        self,
        density: EnvironDensity,
        dielectric: EnvironDielectric,
        electrolyte: EnvironElectrolyte = None,
        semiconductor: EnvironSemiconductor = None,
    ) -> EnvironDensity:
        """docstring"""
        grid = dielectric.epsilon.grid

        phi = EnvironDensity(grid)

        inv_sqrt = EnvironDensity(
            grid,
            np.reciprocal(np.sqrt(dielectric.epsilon)),
        )

        r = EnvironDensity(grid, density)
        z = EnvironDensity(grid)
        p = EnvironDensity(grid)
        Ap = EnvironDensity(grid)

        rzold = 0.0

        for i in range(self.maxiter):
            z[:] = self.cores.electrostatics.poisson(r * inv_sqrt) * inv_sqrt
            rznew = z.scalar_product(r)

            if abs(rzold) > 1.e-30 and not self.conjugate:
                beta = rznew / rzold
            else:
                beta = 0.0

            rzold = rznew

            p[:] = z + beta * p
            Ap[:] = z * dielectric.factsqrt + r + beta * Ap

            pAp = p.scalar_product(Ap)

            alpha = rznew / pAp
            phi += alpha * p
            r -= alpha * Ap

            delta_en = r.euclidean_norm()

            if delta_en <= self.tol: break

        return phi

    @overload
    @dispatch(EnvironCharges)
    def generalized(self, charges: EnvironCharges) -> EnvironDensity:
        """docstring"""
        self.generalized(
            charges.density,
            charges.dielectric,
            charges.electrolyte,
            charges.semiconductor,
        )
