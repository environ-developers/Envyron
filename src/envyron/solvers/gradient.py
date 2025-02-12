from typing import Optional

import numpy as np

from .direct import DirectSolver
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
        direct: DirectSolver,
        preconditioner: str = "sqrt",
        conjugate: Optional[bool] = True,
        maxiter: Optional[int] = 100,
        tol: Optional[float] = 1.0e-7,
        auxiliary: Optional[str] = '',
        verbosity: Optional[int] = 1,
    ) -> None:
        super().__init__(cores, direct, maxiter, tol, auxiliary)
        self.preconditioner = preconditioner
        self.conjugate = conjugate
        self.verbosity = verbosity

    @IterativeSolver.charge_operation
    def generalized(
        self,
        density: EnvironDensity,
        dielectric: EnvironDielectric,
        *args, **kwargs
    ) -> EnvironDensity:
        """docstring"""

        # optional arguments
        if 'electrolyte' in kwargs.keys():
            electrolyte = kwargs['electrolyte']
        if 'semiconductor' in kwargs.keys():
            semiconductor = kwargs['semiconductor']

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
            z[:] = self.direct.poisson(r * inv_sqrt) * inv_sqrt
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

    @IterativeSolver.charge_operation
    def linearized_pb(
        self,
        density: EnvironDensity,
        electrolyte: EnvironElectrolyte,
        dielectric: EnvironDielectric = None,
        screening: EnvironDensity = None,
        **kwargs
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()
