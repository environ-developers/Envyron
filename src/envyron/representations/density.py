from __future__ import annotations

from typing import Optional
from numpy import ndarray

import numpy as np

from ..domains.cell import EnvironGrid
from . import EnvironField


class EnvironDensity(EnvironField):
    """docstring"""

    def __new__(
        cls,
        grid: EnvironGrid,
        data: Optional[ndarray] = None,
        label: str = '',
    ) -> EnvironDensity:
        obj = super().__new__(cls, grid, rank=1, data=data, label=label)
        obj._charge = None
        obj.dipole = np.zeros(3)
        obj.quadrupole = np.zeros(3)
        return obj

    @property
    def charge(self) -> float:
        if self._charge is None: self._charge: float = self.integral()
        return self._charge

    def compute_multipoles(self, origin: ndarray) -> None:
        """docstring"""
        r, _ = self.grid.get_min_distance(origin)
        self._charge = self.integral()
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
