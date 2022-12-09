from typing import Optional, Union, overload

import numpy as np
from multipledispatch import dispatch

from .iterative import IterativeSolver
from ..cores import NumericalCore, CoreContainer
from ..physical import EnvironDielectric, EnvironElectrolyte, EnvironSemiconductor, EnvironCharges
from ..representations import EnvironDensity
from ..utils.constants import FPI


class GradientSolver(IterativeSolver):
    """docstring"""

    def __init__(self,
                 cores: CoreContainer,
                 preconditioner: str = "sqrt",
                 steepest_descent: Optional[bool] = False,
                 tol: Optional[float] = 1.0e-7,
                 max_iter: Optional[int] = 100,
                 verbosity: Optional[int] = 1) -> None:
        super().__init__(cores=cores, max_iter=max_iter, tol=tol)
        self.preconditioner = preconditioner
        self.steepest_descent = steepest_descent
        self.verbosity = verbosity

    @overload
    @dispatch(EnvironCharges)
    def generalized(self, charges: EnvironCharges) -> EnvironDensity:
        """docstring"""
        if self.preconditioner == "sqrt":
            self.generalized(charges.density, charges.dielectric,
                             charges.electrolyte, charges.semiconductor)

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
        phi: EnvironDensity = EnvironDensity(grid=dielectric.epsilon.grid)

        r: EnvironDensity = density.copy()
        inv_sqrt_epsilon: EnvironDensity = np.reciprocal(
            np.sqrt(self.dielectric.epsilon))

        Ap: EnvironDensity = EnvironDensity(grid=self.dielectric.epsilon.grid)
        p: EnvironDensity = EnvironDensity(grid=self.dielectric.epsilon.grid)

        rzold: float = 0.0
        num_iter: int = 0

        while num_iter < self.max_iter:
            z = self.cores.electrostatics.poisson(
                r * inv_sqrt_epsilon) * inv_sqrt_epsilon
            rznew = z.scalar_product(r)

            if abs(rzold) > 1.e-30 and not self.steepest_descent:
                beta = rznew / rzold
            else:
                beta = 0.0

            p = z + beta * p
            Ap = z * dielectric.factsqrt + r + beta * Ap
            pAp = p.scalarProduct(Ap)
            alpha = rznew / pAp
            phi = phi + alpha * p  # In fortran environ phi is v
            r = r - alpha * Ap
            delta_en = r.euclidean_norm()
            delta_qm = r.quadratic_mean()

            num_iter += 1

            if delta_en <= self.tol:
                break
            rzold = rznew
        return phi