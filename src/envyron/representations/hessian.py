from typing import Optional
from numpy import ndarray

import numpy as np

from dftpy.field import DirectField

from ..domains.cell import EnvironGrid
from . import EnvironDensity, EnvironGradient

class EnvironHessian(DirectField):
    """docstring"""

    def __new__(
        cls,
        grid: EnvironGrid,
        data: Optional[ndarray] = None,
        label: str = '',
    ) -> None:
        obj = super().__new__(cls, grid, rank=9, data=data)
        obj.label = label
        mod_label = f"{label or 'hessian'}_laplacian"
        obj.laplacian = EnvironDensity(grid, label=mod_label)
        return obj

    @property
    def label(self) -> str:
        return self.__label

    @label.setter
    def label(self, label: str) -> None:
        """docstring"""
        self.__label = label

    @property
    def laplacian(self) -> EnvironDensity:
        return self.__laplacian

    @laplacian.setter
    def laplacian(self, laplacian: EnvironDensity) -> None:
        """docstring"""
        self.__laplacian = laplacian

    def update(self) -> None:
        """docstring"""
        reshaped = self.reshape(3, 3, *self.grid.nr)
        self.laplacian[:] = reshaped.trace()

    def scalar_gradient_product(
        self,
        gradient: EnvironGradient,
    ) -> EnvironGradient:
        """docstring"""
        reshaped = self.reshape(3, 3, *self.grid.nr)
        data = np.einsum('ml...,l...->m...', reshaped, gradient)
        return EnvironGradient(self.grid, data=data)
