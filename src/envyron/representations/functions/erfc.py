from certifi import where
import numpy as np
import scipy.special as sp

from . import EnvironFunction, EXP_TOL, FUNC_TOL

from .. import EnvironDensity, EnvironGradient, EnvironHessian

from ...utils.constants import FPI, SQRTPI


class EnvironERFC(EnvironFunction):
    """docstring"""

    def density(self) -> EnvironDensity:
        """docstring"""

        _, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        data = sp.erfc(arg)

        integral = np.sum(data) * self.grid.volume / self.grid.nnrR * 0.5

        charge = self._charge()
        analytic = self._erfc_volume()

        if np.abs((integral - analytic) / analytic > 1e-4):
            print("\nWARNING: wrong integral of erfc function\n")

        data *= charge / analytic * 0.5

        return EnvironDensity(self.grid, data=data, label=self.label)

    def gradient(self) -> EnvironGradient:
        """docstring"""

        r, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        mask = dist > FUNC_TOL

        exp = np.zeros(self.grid.nr)
        np.exp(-arg**2, where=mask, out=exp)

        data = np.zeros((3, *self.grid.nr))
        np.multiply(-exp, r, where=mask, out=data)

        charge = self._charge()
        analytic = self._erfc_volume()
        data *= charge / analytic / SQRTPI / self.spread

        return EnvironGradient(self.grid, data=data, label=self.label)

    def laplacian(self) -> EnvironDensity:
        """docstring"""

        _, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        mask = dist > FUNC_TOL

        exp = np.zeros(self.grid.nr)
        exp = np.exp(-arg**2, where=mask, out=exp)

        data = np.zeros(self.grid.nr)

        factor = np.zeros(self.grid.nr)
        ones = np.ones(self.grid.nr)

        if self.dim == 0:

            np.divide(ones, dist, where=mask, out=factor)
            np.subtract(factor, arg / self.spread, where=mask, out=factor)
            np.multiply(-exp, factor * 2, where=mask, out=data)

        elif self.dim == 1:

            np.divide(ones, dist, where=mask, out=factor)
            np.subtract(factor, 2. * arg / self.spread, where=mask, out=factor)
            np.multiply(-exp, factor, where=mask, out=data)

        elif self.dim == 2:

            # TODO not `-exp`?
            factor = arg / self.spread * 2.
            np.multiply(np.exp(-arg**2), factor, where=mask, out=data)

        else:
            raise ValueError("unexpected system dimensions")

        charge = self._charge()
        analytic = self._erfc_volume()
        data *= charge / analytic / SQRTPI / self.spread

        return EnvironDensity(self.grid, data=data, label=self.label)

    def derivative(self) -> EnvironDensity:
        """docstring"""

        _, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        mask = dist > FUNC_TOL

        data = np.zeros(self.grid.nr)
        -np.exp(-arg**2, where=mask, out=data)

        integral = np.sum(sp.erfc(arg, where=mask, out=data))
        integral *= self.grid.volume / self.grid.nnrR * 0.5

        charge = self._charge()
        analytic = self._erfc_volume()

        if np.abs((integral - analytic) / analytic > 1e-4):
            print("\nWARNING: wrong integral of erfc function\n")

        data *= charge / analytic / SQRTPI / self.spread

        return EnvironDensity(self.grid, data=data, label=self.label)

    def _charge(self) -> float:
        """docstring"""
        charge = self.volume
        if self.kind == 1: raise ValueError("wrongly set as a gaussian")
        elif self.kind == 2: pass
        elif self.kind == 3: charge *= self._erfc_volume()
        elif self.kind == 4: charge *= -self._erfc_volume()
        else: raise ValueError("unexpected function type")
        return charge

    def _erfc_volume(self) -> None:
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
