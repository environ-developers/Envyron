import numpy as np

from . import EnvironFunction, EXP_TOL

from .. import EnvironDensity, EnvironGradient

from ...utils.constants import SQRTPI


class EnvironGaussian(EnvironFunction):
    """docstring"""

    def density(self) -> EnvironDensity:
        """docstring"""

        _, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        r2 /= self.spread**2

        mask = r2 <= EXP_TOL

        data = np.zeros(self.grid.nr)
        np.exp(-r2, where=mask, out=data)

        scale = self._get_scale_factor()
        data *= scale

        return EnvironDensity(self.grid, data=data, label=self.label)

    def gradient(self) -> EnvironGradient:
        """docstring"""

        spread2 = self.spread**2

        r, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        r2 /= spread2

        mask = r2 <= EXP_TOL

        exp = np.zeros(self.grid.nr)
        np.exp(-r2, where=mask, out=exp)

        data = np.zeros((3, *self.grid.nr))
        np.multiply(-exp, r, where=mask, out=data)

        scale = self._get_scale_factor()
        data *= scale * 2 / spread2

        return EnvironGradient(self.grid, data=data, label=self.label)

    def _get_scale_factor(self) -> float:
        """docstring"""

        dim = self.dim
        axis = self.axis
        charge = self.volume
        spread = self.spread

        if dim in {1, 2}:
            length = np.abs(self.grid.lattice(axis, axis))

        if dim == 0:
            scale = charge / (SQRTPI * spread)**3
        elif dim == 1:
            scale = charge / length / (SQRTPI * spread)**2
        elif dim == 2:
            scale = charge / length / self.grid.volume / (SQRTPI * spread)
        else:
            raise ValueError("dimensions out of range")

        return scale
