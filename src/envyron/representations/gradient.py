from typing import Optional
from typing_extensions import Self
from numpy import ndarray

import numpy as np

from dftpy.field import DirectField

from ..domains.cell import EnvironGrid
from . import EnvironDensity


class EnvironGradient(DirectField):
    """docstring"""

    def __new__(
        cls,
        grid: EnvironGrid,
        data: Optional[ndarray] = None,
        label: str = '',
    ) -> Self:
        obj = super().__new__(cls, grid, rank=3, data=data)
        obj.label = label
        mod_label = f"{label or 'gradient'}_modulus"
        obj.modulus = EnvironDensity(grid, label=mod_label)
        return obj

    @property
    def label(self) -> str:
        return self.__label

    @label.setter
    def label(self, label: str) -> None:
        """docstring"""
        self.__label = label

    @property
    def modulus(self) -> EnvironDensity:
        return self.__modulus

    @modulus.setter
    def modulus(self, modulus: EnvironDensity) -> None:
        """docstring"""
        self.__modulus = modulus

    def standard_view(self) -> Self:
        """docstring"""
        return self.T.reshape(self.grid.nnr, 3)

    def update(self) -> None:
        """docstring"""
        self.modulus[:] = np.sqrt(np.sum(self**2, 0))

    def scalar_gradient_product(
        self,
        gradient: Self,
    ) -> EnvironDensity:
        """docstring"""
        data = np.einsum('l...,l...', self, gradient)
        return EnvironDensity(self.grid, data=data)

    def scalar_density_product(
        self,
        density: 'EnvironDensity',
    ) -> ndarray:
        """docstring"""
        return np.einsum('lijk,ijk', self, density) * self.grid.dV
