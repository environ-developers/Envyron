from typing import Optional
from numpy import ndarray

import numpy as np

from . import EnvironIons


class EnvironSystem:
    """docstring"""

    def __init__(
        self,
        ntypes: int,
        dim: int,
        axis: int,
        ions: EnvironIons,
    ) -> None:
        self.dim = dim
        self.axis = axis
        self.ions = ions

        self.com = np.zeros(3)
        self.width = 0.0

        self.ntypes = ntypes or self.ions.ntypes

        self.updating = False

    def update(self, center: Optional[ndarray] = None) -> None:
        """docstring"""

        if center is not None:
            self.com = center
        else:
            total_weight = 0.0

            for i in range(self.ions.count):
                itype = self.ions.itypes[i]

                if itype > self.ntypes - 1: continue

                weight = self.ions.iontypes[itype].weight
                self.com += self.ions.coords[i] * weight
                total_weight += weight

            self.com /= total_weight

        for i in range(self.ions.count):
            itype = self.ions.itypes[i]

            if itype > self.ntypes - 1: continue

            dist = 0.0

            for j in range(3):

                if self.dim == 1 and j == self.axis or \
                    self.dim == 2 and j != self.axis:
                    continue

                dist += (self.ions.coords[i, j] - self.com[j])**2

            self.width = max(self.width, dist)

        self.width = np.sqrt(self.width)
