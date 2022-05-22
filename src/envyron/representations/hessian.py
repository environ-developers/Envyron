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
        obj._trace = None
        return obj

    @property
    def trace(self) -> EnvironDensity:
        if self._trace is None: self._compute_trace()
        return self._trace

    def _compute_trace(self) -> None:
        """docstring"""
        self._trace = EnvironDensity(
            self.grid,
            data=self[0] + self[4] + self[8],
            label=f"{self.label or ''} laplacian".strip(),
        )

    def scalar_gradient_product(
        self,
        gradient: EnvironGradient,
    ) -> EnvironGradient:
        """docstring"""
        reshaped = self.reshape(3, 3, *self.grid.nr)
        data = np.einsum('ml...,l...->m...', reshaped, gradient)
        return EnvironGradient(self.grid, data=data)
