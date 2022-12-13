from __future__ import annotations

from typing import Optional
from numpy import ndarray

import numpy as np

from multimethod import multimethod

from ..domains.cell import EnvironGrid
from . import EnvironField, EnvironDensity


class EnvironGradient(EnvironField):
    """docstring"""

    def __new__(
        cls,
        grid: EnvironGrid,
        data: Optional[ndarray] = None,
        label: str = '',
    ) -> EnvironGradient:
        obj = super().__new__(cls, grid, rank=3, data=data, label=label)
        obj._modulus = None
        return obj

    @property
    def modulus(self) -> EnvironDensity:
        if self._modulus is None: self._compute_modulus()
        return self._modulus

    def _compute_modulus(self) -> None:
        """docstring"""
        self._modulus = EnvironDensity(
            self.grid,
            data=np.sqrt(np.sum(self**2, 0)),
            label=f"{self.label or 'gradient'}_modulus",
        )

    @multimethod
    def scalar_product(
        self,
        gradient: EnvironGradient,
    ) -> EnvironDensity:
        """docstring"""
        data = np.einsum('l...,l...', self, gradient)
        return EnvironDensity(self.grid, data=data)

    @multimethod
    def scalar_product(
        self,
        density: EnvironDensity,
    ) -> ndarray:
        """docstring"""
        return np.einsum('lijk,ijk', self, density) * self.grid.dV
