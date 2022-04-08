from typing import Tuple
from numpy import ndarray, newaxis

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
        self.corners = -np.array(list(product(range(2), repeat=3))).dot(at)

    @property
    def label(self) -> float:
        return self.__label

    @label.setter
    def label(self, label: str) -> None:
        """docstring"""
        self.__label = label

    def get_min_distance(
        self,
        origin: ndarray,
        dim: int = 0,
        axis: int = 0,
    ) -> Tuple[ndarray]:
        """docstring"""
        r = self._get_displacement(self.r, origin, dim, axis)
        r, r2 = self._apply_minimum_image_convension(r)
        return r, r2

    @staticmethod
    def _get_displacement(
        r: ndarray,
        origin: ndarray,
        dim: int = 0,
        axis: int = 0,
    ) -> ndarray:
        """docstring"""

        dr = r - origin[:, newaxis, newaxis, newaxis]

        if dim == 0:
            pass
        elif dim == 1:
            dr[:, :, :, axis] = 0.
        elif dim == 2:
            dr[:, :, :, np.arange(3) != axis] = 0.
        else:
            raise ValueError("Dimensions out of bounds")

        return dr

    def _apply_minimum_image_convension(self, r: ndarray) -> Tuple[ndarray]:
        """docstring"""

        # apply minimum image convension
        reciprocal_lattice = self.get_reciprocal().lattice / 2 / np.pi
        s = np.einsum('lijk,lm->lijk', r, reciprocal_lattice)
        s -= np.floor(s)
        r = np.einsum('ml,lijk->lijk', self.lattice, s)

        # pre-corner-check results
        rmin = r
        r2min = np.einsum('i...,i...', r, r)

        # check against corner shifts
        for corner in self.corners[1:]:
            s = r + corner[:, newaxis, newaxis, newaxis]
            s2 = np.einsum('i...,i...', s, s)

            condition = s2 < r2min
            rmin = np.where(condition[newaxis, :, :, :], s, rmin)
            r2min = np.where(condition, s2, r2min)

        return rmin, r2min
