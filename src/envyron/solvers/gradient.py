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
                 cores: CoreContainer,,
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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def solve(self, density: EnvironDensity) -> EnvironDensity:
        if not self.tol > 0:
            raise ValueError("convergence tolerance must be greater than 0")
        if not self.max_iter > 0:
            raise ValueError("max iterations must be greater than 0")

        phi: EnvironDensity = EnvironDensity(grid=self.dielectric.epsilon.grid)

        r: EnvironDensity = density.copy()

        if self.preconditioner == "sqrt":
            inv_sqrt_epsilon: EnvironDensity = np.reciprocal(
                np.sqrt(self.dielectric.epsilon))
        elif self.preconditioner == "left":
            inv_epsilon: EnvironDensity = np.reciprocal(
                self.dielectric.epsilon)
        else:
            raise AttributeError("Invalid preconditioner keyword")

        Ap: EnvironDensity = EnvironDensity(grid=self.dielectric.epsilon.grid)
        p: EnvironDensity = EnvironDensity(grid=self.dielectric.epsilon.grid)

        rzold: float = 0.0
        num_iter: int = 0

        while num_iter < self.max_iter:
            if self.preconditioner == "none":
                z: EnvironDensity = r
            elif self.preconditioner == "sqrt":
                z: EnvironDensity = self.__preconditioner_sqrt__(
                    r, inv_sqrt_epsilon)
            elif self.preconditioner == "left":
                z: EnvironDensity = self.__preconditioner_left__(
                    r, inv_epsilon)

            rznew = z.scalar_product(r)

            if abs(rzold) > 1.e-30 and not self.steepest_descent:
                beta = rznew / rzold
            else:
                beta = 0.0

            p = z + beta * p
            Ap = z * self.dielectric.factsqrt + r + beta * Ap
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

    def _preconditioner_sqrt(
        self,
        rk: EnvironDensity,
        inv_sqrt_epsilon: Optional[Union[EnvironDensity, None]] = None
    ) -> EnvironDensity:
        """docstring"""
        if inv_sqrt_epsilon is None:
            inv_sqrt_epsilon = np.reciprocal(np.sqrt(self.dielectric.epsilon))
        return self.cores.electrostatics.poisson(
            rk * inv_sqrt_epsilon) * inv_sqrt_epsilon

    def _preconditioner_left(
        self,
        rk: EnvironDensity,
        inv_epsilon: Optional[Union[EnvironDensity, None]] = None
    ) -> EnvironDensity:
        """docstring"""
        if inv_epsilon is None:
            inv_epsilon = np.reciprocal(self.dielectric.epsilon)
        return self.cores.electrostatics.poisson(rk * inv_epsilon)
