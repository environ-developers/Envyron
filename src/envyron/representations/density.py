from __future__ import annotations

from typing import Optional
from numpy import ndarray

import numpy as np

from . import EnvironField
from ..domains.cell import EnvironGrid


class EnvironDensity(EnvironField):
    """docstring"""

    def __new__(
        cls,
        grid: EnvironGrid,
        data: Optional[ndarray] = None,
        label: str = '',
    ) -> EnvironDensity:
        obj = super().__new__(cls, grid, rank=1, data=data, label=label)
        obj.charge = 0.
        obj.dipole = np.zeros(3)
        obj.quadrupole = np.zeros(3)
        return obj

    @property
    def charge(self) -> float:
        return self.__charge

    @charge.setter
    def charge(self, charge: float) -> None:
        """docstring"""
        self.__charge = charge

    @property
    def dipole(self) -> ndarray:
        return self.__dipole

    @dipole.setter
    def dipole(self, dipole: ndarray) -> None:
        """docstring"""
        self.__dipole = dipole

    @property
    def quadrupole(self) -> ndarray:
        return self.__quadrupole

    @quadrupole.setter
    def quadrupole(self, quadrupole: ndarray) -> None:
        """docstring"""
        self.__quadrupole = quadrupole

    def compute_multipoles(self, origin: ndarray) -> None:
        """docstring"""
        r, _ = self.grid.get_min_distance(origin)
        self.charge = self.integral()
        self.dipole = np.einsum('ijk,lijk', self, r) * self.grid.dV
        self.quadrupole = np.einsum('ijk,lijk', self, r**2) * self.grid.dV

    def euclidean_norm(self) -> float:
        """docstring"""
        return np.einsum('ijk,ijk', self, self)

    def quadratic_mean(self) -> float:
        """docstring"""
        return np.sqrt(self.euclidean_norm() / self.grid.nnrR)

    def scalar_product(self, density: EnvironDensity) -> float:
        """docstring"""
        return np.einsum('ijk,ijk', self, density) * self.grid.dV
