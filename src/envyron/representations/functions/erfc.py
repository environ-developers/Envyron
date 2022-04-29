import numpy as np
import scipy.special as sp

from ...utils.constants import FPI, SQRTPI
from .. import EnvironDensity, EnvironGradient, EnvironHessian
from . import EnvironFunction, FUNC_TOL


class EnvironERFC(EnvironFunction):
    """docstring"""

    def _compute_density(self) -> None:
        """docstring"""

        _, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        self._density = EnvironDensity(self.grid, label=self.label)
        self._density[:] = sp.erfc(arg)

        integral = np.sum(self._density) * \
                   self.grid.volume / self.grid.nnrR * 0.5

        charge = self._charge()
        analytic = self._erfc_volume()

        if np.abs((integral - analytic) / analytic > 1e-4):
            print("\nWARNING: wrong integral of erfc function\n")

        self._density[:] *= charge / analytic * 0.5

    def _compute_gradient(self) -> None:
        """docstring"""

        r, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        mask = dist > FUNC_TOL

        r = r[:, mask]
        arg = arg[mask]

        self._gradient = EnvironGradient(self.grid, label=self.label)
        self._gradient[:, mask] = -np.exp(-arg**2) * r

        charge = self._charge()
        analytic = self._erfc_volume()
        self._gradient[:] *= charge / analytic / SQRTPI / self.spread

    def _compute_laplacian(self) -> None:
        """docstring"""

        _, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        mask = dist > FUNC_TOL

        dist = dist[mask]
        arg = arg[mask]

        self._laplacian = EnvironDensity(self.grid, label=self.label)

        exp = np.exp(-arg**2)

        if self.dim == 0:
            self._laplacian[mask] = -exp * (1 / dist - arg / self.spread) * 2
        elif self.dim == 1:
            self._laplacian[mask] = -exp * (1 / dist - 2 * arg / self.spread)
        elif self.dim == 2:
            self._laplacian[mask] = exp * arg / self.spread * 2
        else:
            raise ValueError("unexpected system dimensions")

        charge = self._charge()
        analytic = self._erfc_volume()
        self._laplacian[:] *= charge / analytic / SQRTPI / self.spread

    def _compute_hessian(self) -> None:
        """docstring"""

        r, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        mask = dist > FUNC_TOL
        count = np.count_nonzero(mask)

        r = r[:, mask]
        dist = dist[mask]
        arg = arg[mask]

        self._hessian = EnvironHessian(self.grid, label=self.label)

        outer = np.reshape(np.einsum('i...,j...->ij...', -r, r), (9, count))
        outer *= 1 / dist + 2 * arg / self.spread
        outer += dist * np.identity(3).flatten()[:, None]
        self._hessian[:, mask] = -np.exp(-arg**2) * outer / dist**2

        charge = self._charge()
        analytic = self._erfc_volume()
        self._hessian[:] *= charge / analytic / SQRTPI / self.spread

    def _compute_derivative(self) -> None:
        """docstring"""

        _, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        mask = dist > FUNC_TOL

        arg = arg[mask]

        self._derivative = EnvironDensity(self.grid, label=self.label)
        self._derivative[mask] = -np.exp(-arg**2)

        integral = np.sum(self._derivative) * \
                   self.grid.volume / self.grid.nnrR * 0.5

        charge = self._charge()
        analytic = self._erfc_volume()

        if np.abs((integral - analytic) / analytic > 1e-4):
            print("\nWARNING: wrong integral of erfc function\n")

        self._derivative[:] *= charge / analytic / SQRTPI / self.spread

    def _charge(self) -> float:
        """docstring"""
        charge = self.volume
        if self.kind == 1: raise ValueError("wrongly set as a gaussian")
        elif self.kind == 2: pass
        elif self.kind == 3: charge *= self._erfc_volume()
        elif self.kind == 4: charge *= -self._erfc_volume()
        else: raise ValueError("unexpected function type")
        return charge

    def _erfc_volume(self) -> float:
        """docstring"""

        spread = self.spread
        width = self.width

        if any(attr < FUNC_TOL for attr in (spread, width)):
            raise ValueError("wrong parameters for erfc function")

        t = spread / width
        invt = width / spread
        f1 = (1 + sp.erf(invt)) * 0.5
        f2 = np.exp(-invt**2) * 0.5 / SQRTPI

        if self.dim == 0:

            volume = FPI / 3 * width**3 * \
                     ((1. + 1.5 * t**2) * f1 + \
                         (1. + t**2) * t * f2)

        elif self.dim == 1:

            volume = np.pi * width**2 * \
                self.grid.lattice[self.axis, self.axis] * \
                     ((1. + 0.5 * t**2) * f1 + t * f2)

        elif self.dim == 2:

            volume = 2. * width * self.grid.volume / \
                self.grid.lattice[self.axis, self.axis]

        else:
            raise ValueError("unexpected system dimensions")

        return volume
