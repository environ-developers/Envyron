from __future__ import annotations

from typing import Optional
from numpy import ndarray

from dftpy.field import DirectField

from ..domains.cell import EnvironGrid


class EnvironField(DirectField):
    """docstring"""

    def __new__(
        cls,
        grid: EnvironGrid,
        rank: int = 1,
        data: Optional[ndarray] = None,
        label: str = '',
    ) -> EnvironField:
        obj = super().__new__(cls, grid, rank=rank, data=data)
        obj.label = label
        return obj

    def standard_view(self) -> 'EnvironField':
        """docstring"""
        return self.T.reshape(self.grid.nnr, self.rank)
