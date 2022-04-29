from __future__ import annotations

from typing import Optional
from numpy import ndarray

import numpy as np

from ..domains.cell import EnvironGrid
from . import EnvironField, EnvironDensity, EnvironGradient


class EnvironHessian(EnvironField):
    """docstring"""

    def __new__(
        cls,
        grid: EnvironGrid,
        data: Optional[ndarray] = None,
        label: str = '',
    ) -> EnvironHessian:
        obj = super().__new__(cls, grid, rank=9, data=data, label=label)
        obj._laplacian = None
        return obj

    @property
    def laplacian(self) -> EnvironDensity:
        if self._laplacian is None: self._compute_laplacian()
        return self._laplacian

    def _compute_laplacian(self) -> None:
        """docstring"""
        self._laplacian = EnvironDensity(
            self.grid,
            data=self.trace(),
            label=f"{self.label or 'hessian'}_laplacian",
        )

    def trace(self) -> ndarray:
        """docstring"""
        return self[0] + self[4] + self[8]

    def scalar_gradient_product(
        self,
        gradient: EnvironGradient,
    ) -> EnvironGradient:
        """docstring"""
        reshaped = self.reshape(3, 3, *self.grid.nr)
        data = np.einsum('ml...,l...->m...', reshaped, gradient)
        return EnvironGradient(self.grid, data=data)
