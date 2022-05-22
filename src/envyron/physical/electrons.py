from typing import Optional
from numpy import ndarray

import numpy as np

from ..domains import EnvironGrid
from ..representations import EnvironDensity


class EnvironElectrons:
    """docstring"""

    def __init__(self, grid: EnvironGrid) -> None:
        self.density = EnvironDensity(grid, label='electrons')
        
        self.updating = False

    def update(self, rho: ndarray, nelec: Optional[int] = None) -> None:
        """docstring"""
        self.density[:] = rho
        self.charge = self.density.charge
        self.count = int(np.rint(self.charge))

        if nelec is not None:
            error = np.abs(self.charge - nelec)

            if error > 5e-3:
                raise ValueError(
                    f"{error:.2e} error in integrated electronic charge")
