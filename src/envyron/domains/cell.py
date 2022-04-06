from typing import Tuple
from numpy import ndarray

import numpy as np
from itertools import product

from dftpy.grid import DirectGrid


class EnvironGrid(DirectGrid):
    """docstring"""

    def __init__(
        self,
        at: ndarray,
        nr: ndarray,
        label: str = '',
        units: str = 'bohr',
    ) -> None:
        super().__init__(at, nr, units=units)
        self.label = label
        self.corners = np.array(list(product(range(2), repeat=3))).dot(at)

    @property
    def label(self) -> float:
        return self.__label

    @label.setter
    def label(self, label: str) -> str:
        """docstring"""
        self.__label = label

    def get_min_distance(
        self,
        origin: ndarray,
        dim: int = 0,
        axis: int = 0,
    ) -> Tuple[ndarray]:
        """docstring"""
        r = self._get_displacement(self.r.T, origin, dim, axis)
        r, r2 = self._apply_minimum_image_convension(r)
        return r, r2

    @staticmethod
    def _get_displacement(
        r1: ndarray,
        r2: ndarray,
        dim: int = 0,
        axis: int = 0,
    ) -> ndarray:
        """docstring"""

        dr = r1 - r2

        if dim != 0:

            if dim < 0 or dim > 2:
                raise ValueError("Dimensions out of bounds")

            # for simple vector displacement
            if len(dr.shape) == 1:
                if dim == 1:
                    dr[axis] = 0.
                elif dim == 2:
                    dr[np.arange(3) != axis] = 0.

            # for vectorized matrix displacement
            else:
                if dim == 1:
                    dr[:, :, :, axis] = 0.
                elif dim == 2:
                    dr[:, :, :, np.arange(3) != axis] = 0.

        return dr

    def _apply_minimum_image_convension(self, r: ndarray) -> Tuple[ndarray]:
        """docstring"""

        # apply minimum image convension
        s = np.matmul(r, self.get_reciprocal().lattice / 2 / np.pi)
        s -= np.floor(s)
        r = np.einsum('nm,ijkm', self.lattice, s)
        
        # pre-corner-check results
        rmin = r
        r2min = np.sum(r * r, 3)

        # check against corner shifts
        for corner in self.corners:
            s = r + corner
            s2 = np.sum(s * s, 3)

            condition = s2 < r2min
            mask = np.array([condition] * 3).T
            rmin = np.where(mask, s, rmin)
            r2min = np.where(condition, s2, r2min)

        return rmin, r2min
