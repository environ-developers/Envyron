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

    def get_min_distance(
        self,
        origin: ndarray,
        dim: int = 0,
        axis: int = 0,
    ) -> Tuple[ndarray, ndarray]:
        """docstring"""
        r = self.r - origin[:, newaxis, newaxis, newaxis]
        r, r2 = self._apply_minimum_image_convension(r, dim, axis)
        return r, r2

    def _get_direction(
        self,
        dim: int = 0,
        axis: int = 0,
    ):
        """docstring"""
        if dim == 0:
            n = np.zeros(3)
        elif dim == 1:
            n = self.cell[axis, :]
        elif dim == 2:
            n1, n2 = self.cell[np.arange(3) != axis, :]
            n = np.cross(n2, n1)
        else:
            raise ValueError("dimensions out of range")
    
        norm = np.linalg.norm(n)
        if norm > 1.e-16 :
            n = n/norm
        return n

    def _reduce_dimension(
        self,
        r: ndarray,
        n: ndarray,
        dim: int = 0,
    ):
        """docstring"""
        if dim == 0:
            pass
        elif dim == 1:
            r = r - np.einsum('jkl,i->ijkl', np.einsum('ijkl,i->jkl', r, n), n)
        elif dim == 2:
            r = np.einsum('jkl,i->ijkl', np.einsum('ijkl,i->jkl', r, n), n)
        else:
            raise ValueError("dimensions out of range")
        return r

    def _apply_minimum_image_convension(
        self,
        r: ndarray,
        dim: int = 0,
        axis: int = 0,
    ) -> Tuple[ndarray, ndarray]:
        """docstring"""

        n = self._get_direction(dim,axis)

        # apply minimum image convension
        reciprocal_lattice = self.get_reciprocal().lattice / 2 / np.pi
        s = np.einsum('lijk,ml->mijk', r, reciprocal_lattice)
        s -= np.floor(s)
        r = np.einsum('lm,lijk->mijk', self.lattice, s)
        r = self._reduce_dimension(r, n, dim)

        # pre-corner-check results
        rmin = r
        r2min = np.einsum('i...,i...', r, r)

        t = r
        # check against corner shifts
        for corner in self.corners[1:]:
            r = t + corner[:, np.newaxis, np.newaxis, np.newaxis]
            r = self._reduce_dimension(r, n, dim)
            r2 = np.einsum('i...,i...', r, r)
            mask = r2 < r2min
            rmin = np.where(mask[newaxis, :, :, :], r, rmin)
            r2min = np.where(mask, r2, r2min)

        return rmin, r2min
