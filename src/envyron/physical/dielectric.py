from ..representations import EnvironDensity
from ..boundaries import EnvironBoundary, ElectronicBoundary
from ..utils.constants import FPI

import numpy as np


class EnvironDielectric:
    """docstring"""

    def __init__(
        self,
        boundary: EnvironBoundary,
        constant=1.0,
        need_gradient=False,
        need_factsqrt=False,
        need_auxiliary=False,
    ) -> None:
        self.boundary = boundary
        self.constant = constant

        self.epsilon = EnvironDensity(grid=boundary.grid)
        self.depsilon = EnvironDensity(grid=boundary.grid)
        self.gradlogepsilon = EnvironDensity(grid=boundary.grid, rank=3)

        self.need_gradient = need_gradient
        if self.need_gradient:
            self.gradient = EnvironDensity(grid=boundary.grid, rank=3)

        self.need_factsqrt = need_factsqrt
        if self.need_factsqrt:
            self.factsqrt = EnvironDensity(grid=boundary.grid)

        self.need_auxiliary = need_auxiliary
        if self.need_auxiliary:
            self.iterative = EnvironDensity(grid=boundary.grid)

        # polarization density and its multipoles
        self.density = EnvironDensity(grid=boundary.grid)
        self.charge = 0.
        self.dipole = np.zeros(3)
        self.quadrupole = np.zeros(3)

        self.updating = False

    def update(self) -> None:
        """docstring"""

        # if the boundary is updating flag the dielectric as updating
        if self.boundary.update_status > 0:
            self.updating = True

        # if the dielectric needs to be updated
        if self.updating:
            # only do it if the boundary is ready (update_status = 2)
            if self.boundary.update_status == 2:
                self.of_boundary()
                self.updating = False

    def of_boundary(self) -> None:
        """docstring"""

        # for the time being we just consider a uniform background (no regions)
        background = EnvironDensity(grid=self.boundary.grid)
        background[:, :, :] = self.constant

        if isinstance(self.boundary, ElectronicBoundary):
            self.epsilon[:] = np.exp(
                np.log(self.background) * (1. - self.boundary.switch))
            self.depsilon[:] = -self.epsilon * np.log(self.background)
            dlogeps = -np.log(self.background)
            if self.need_factsqrt:
                d2eps = self.epsilon * np.log(self.background)**2
        else:
            self.epsilon[:] = 1. + (self.background - 1.) * \
                (1. - self.boundary.switch)
            self.depsilon[:] = 1. - self.background
            dlogeps = self.depsilon / self.epsilon
            if self.need_factsqrt: d2eps = 0.

        # compute derived quantities

        self.gradlogepsilon[:] = self.boundary.gradient * dlogeps

        if self.need_gradient:
            self.gradient[:] = self.boundary.gradient * self.depsilon

        if self.need_factsqrt:
            self.factsqrt[:] = ( d2eps - 0.5 * self.depsilon**2 / self.epsilon ) * \
                self.boundary.gradient.modulus**2 + \
                self.depsilon * self.boundary.laplacian
            self.factsqrt *= 0.5 / FPI

    def of_potential(
        self,
        charges: EnvironDensity,
        potential: EnvironDensity,
    ) -> None:
        """docstring"""

        gradient = self.boundary.cores.derivatives.gradient(potential)
        self.density[:] = gradient.scalar_gradient_product(self.gradlogepsilon)
        self.density[:] = self.density / FPI + (
            1. - self.epsilon) / self.epsilon * charges
        self.charge = self.density.charge

    def de_dboundary(
        self,
        potential: EnvironDensity,
        de_dboundary: EnvironDensity,
    ) -> None:
        """docstring"""

        gradient = self.boundary.cores.derivatives.gradient(potential)
        de_dboundary -= gradient.modulus**2 * self.depsilon * 0.5 / FPI

    def dv_dboundary(
        self,
        potential: EnvironDensity,
        dpotential: EnvironDensity,
        dv_dboundary: EnvironDensity,
    ) -> None:
        """docstring"""

        gradient = self.boundary.cores.derivatives.gradient(potential)
        dgradient = self.boundary.cores.derivatives.gradient(dpotential)
        dv_dboundary -= gradient.scalar_gradient_product(
            dgradient) * self.depsilon / FPI
