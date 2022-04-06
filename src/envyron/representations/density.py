from typing import Optional
from numpy import ndarray

import numpy as np

from dftpy.field import DirectField

from ..domains.cell import EnvironGrid


class EnvironDensity:
    """docstring"""

    def __init__(
        self,
        grid: EnvironGrid,
        label: str = '',
    ) -> None:
        self.label = label
        self.grid = grid
        self.of_r = DirectField(grid)
        self.charge = 0.
        self.dipole = 0.
        self.quadrupole = 0.

    @property
    def label(self) -> str:
        return self.__label

    @label.setter
    def label(self, label: str) -> None:
        """docstring"""
        self.__label = label

    @property
    def of_r(self) -> ndarray:
        return self.__of_r

    @of_r.setter
    def of_r(self, of_r: str) -> ndarray:
        """docstring"""
        self.__of_r = of_r

    @property
    def charge(self) -> float:
        return self.__charge

    @charge.setter
    def charge(self, charge: float) -> float:
        """docstring"""
        self.__charge = charge

    @property
    def dipole(self) -> float:
        return self.__dipole

    @dipole.setter
    def dipole(self, dipole: ndarray) -> ndarray:
        """docstring"""
        self.__dipole = dipole

    @property
    def quadrupole(self) -> float:
        return self.__quadrupole

    @quadrupole.setter
    def quadrupole(self, quadrupole: ndarray) -> ndarray:
        """docstring"""
        self.__quadrupole = quadrupole

    def compute_multipoles(self, origin: ndarray) -> Tuple[ndarray]:
        """docstring"""
        r, _ = self.grid.get_min_distance(origin)
        self.charge = self.of_r.integral()
        self.dipole = np.einsum('ijk,ijkl', self.of_r, r) * self.grid.dV
        self.quadropole = np.einsum('ijk,ijkl', self.of_r, r**2) * self.grid.dV

    def euclidean_norm(self) -> ndarray:
        """docstring"""
        return np.dot(self.of_r, self.of_r)

    def quadratic_mean(self) -> ndarray:
        """docstring"""
        return np.sqrt(self.euclidean_norm()) / self.grid.nnr

    def scalar_product(self, density: 'EnvironDensity') -> ndarray:
        """docstring"""
        return np.dot(self.of_r, density.of_r) * self.grid.dV
