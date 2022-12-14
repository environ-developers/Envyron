import numpy as np

from ..domains import EnvironGrid

from .core import NumericalCore


class Analytic1DCore(NumericalCore):
    """docstring"""

    def __init__(self, grid: EnvironGrid, dim: int, axis: int) -> None:

        if dim == 3 or dim < 0 :
            raise ValueError("Wrong dimensions for analytic one dimensional core")
        self.dim = dim
        self.pdim = 3 - dim

        if (dim == 1 or dim == 2) and (axis > 3 or axis < 1):
            raise ValueError("Wrong choice of axis for analytic one dimensional core")
        self.axis = axis

        self.x = np.zeros((self.pdim,self.grid.nnr))

        self.origin = np.zeros(3)

        super().__init__(grid)

        if dim == 0 :
            self.size = grid.volume
        elif dim == 1 :
            self.size = grid.volume / grid.cell.diagonal()[axis]
        elif dim == 2 :
            self.size = grid.cell.diagonal()[axis]

    def update_origin(self,origin):
        """docstring"""

        self.origin = origin
