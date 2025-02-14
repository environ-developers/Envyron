import numpy as np
from numpy import ndarray

from ..domains import EnvironGrid
from ..representations import EnvironDensity, EnvironGradient
from ..physical import EnvironIons
from .core import NumericalCore
from ..utils.constants import TPI, FPI, MADELUNG, E2


class Analytic1DCore(NumericalCore):
    """docstring"""

    def __init__(self, grid: EnvironGrid, dim: int, axis: int) -> None:
        super().__init__(grid)

        if dim == 3 or dim < 0:
            raise ValueError(
                "Wrong dimensions for analytic one dimensional core")

        self.dim = dim
        self.pdim = 3 - dim

        if (dim == 1 or dim == 2) and (axis > 3 or axis < 1):
            raise ValueError(
                "Wrong choice of axis for analytic one dimensional core")

        self.axis = axis - 1
        self.r = np.zeros(grid.r.shape)
        self.origin = np.zeros(3)

        self.volume = grid.volume
        if dim == 0:
            self.size = grid.volume
        elif dim == 1:
            self.size = grid.volume / grid.cell.diagonal()[axis]
        elif dim == 2:
            self.size = grid.cell.diagonal()[axis]

    def update_origin(self, origin):
        """docstring"""

        self.origin = origin

        self.r, _ = self.grid.get_min_distance(
            self.origin,
            self.dim,
            self.axis,
        )

    def parabolic_correction(self, charges: EnvironDensity,
                             potential: EnvironDensity):
        """docstring"""

        charges.compute_multipoles(self.origin)

        fact = E2 * TPI / self.volume
        correction = EnvironDensity(charges.grid)

        if self.dim == 0:
            const = MADELUNG[0] * charges.charge * E2 / self.size**(
                1. / 3.) - fact * np.sum(charges.quadrupole) / 3.
            correction = -charges.charge * np.sum(
                self.r**2, axis=0) + 2 * np.einsum('i,ij->j', charges.dipole,
                                                   self.r)
            correction = correction * fact / 3. + const
        elif self.dim == 1:
            raise ValueError(
                "Wrong choice of axis for analytic one dimensional core")
        elif self.dim == 2:
            const = -np.pi / 3. * charges.charge / self.size * E2 - fact * charges.quadrupole[
                self.axis]
            correction = -charges.charge * self.r[
                self.axis, :]**2 + 2. * self.dipole[self.axis] * self.r[
                    self.axis, :]
            correction = correction * fact + const
        elif self.dim == 3:
            const = 0.
            correction = 0.
        else:
            raise ValueError(
                "Wrong choice of axis for analytic one dimensional core")

        potential = potential + correction

        return potential

    def parabolic_gradient(self, charges: EnvironDensity,
                           field: EnvironGradient):
        """docstring"""

        charges.compute_multipoles(self.origin)

        fact =  E2 * FPI / self.volume
        gradient = EnvironGradient(charges.grid)

        if self.dim == 0:
            for i in range(3):
                gradient.of_r[i, :] = (charges.dipole[i] -
                                       charges.charge * self.r[i, :]) / 3.
        elif self.dim == 1:
            raise ValueError(
                "Wrong choice of axis for analytic one dimensional core")
        elif self.dim == 2:
            gradient = charges.dipole[
                self.axis] - charges.charge * self.r[self.axis, :]
        elif self.dim == 3:
            gradient = 0.
        else:
            raise ValueError(
                "Wrong choice of axis for analytic one dimensional core")

        gradient = gradient * fact

        field = field + gradient

        return field

    def parabolic_force(self, ions: EnvironIons, auxiliary: EnvironDensity,
                        force: ndarray):
        """docstring"""

        auxiliary.compute_multipoles(self.origin)

        fact = E2 * FPI / self.volume
        ftmp = np.zeros(force.shape)

        for i in range(ions.count):

            if self.dim == 0:
                ftmp[:, i] = (auxiliary.charge * ions.coords[:, i] -
                              auxiliary.dipole[:]) / 3.
            elif self.dim == 1:
                raise ValueError(
                    "Wrong choice of axis for analytic one dimensional core")
            elif self.dim == 2:
                ftmp[self.axis, i] = auxiliary.charge * ions.coords[
                    self.axis, i] - auxiliary.dipole[self.axis]
            elif self.dim == 3:
                ftmp = 0.
            else:
                raise ValueError(
                    "Wrong choice of axis for analytic one dimensional core")
            ftmp[:, i] = ftmp[:, i] * fact * ions.smeared_ions[i].volume

        force = force + ftmp

        return force
