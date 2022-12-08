from ..representations import EnvironDensity
from ..boundaries import EnvironBoundary

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
        raise NotImplementedError

    def of_potential(self) -> None:
        """docstring"""
        raise NotImplementedError

    def de_dboundary(self) -> None:
        """docstring"""
        raise NotImplementedError

    def dv_dboundary(self) -> None:
        """docstring"""
        raise NotImplementedError
