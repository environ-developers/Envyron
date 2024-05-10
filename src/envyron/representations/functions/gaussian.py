import numpy as np

from ...utils.constants import SQRTPI
from .. import EnvironDensity, EnvironGradient
from . import EnvironFunction, EXP_TOL


class EnvironGaussian(EnvironFunction):
    """docstring"""

    def _compute_density(self) -> None:
        """docstring"""

        _, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        r2 /= self.spread**2

        mask = r2 <= EXP_TOL

        r2 = r2[mask]

        self._density = EnvironDensity(self.grid, label=self.label)

        density = np.zeros(self._density.shape)
        density[mask] = np.exp(-r2)

        scale = self._get_scale_factor()

        self._density[mask] += density[mask] * scale

    def _compute_gradient(self) -> None:
        """docstring"""

        spread2 = self.spread**2

        r, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        r2 /= spread2

        mask = r2 <= EXP_TOL

        r = r[:, mask]
        r2 = r2[mask]

        self._gradient = EnvironGradient(self.grid, label=self.label)

        gradient = np.zeros(self._gradient.shape)
        gradient[:, mask] = -np.exp(-r2) * r

        scale = self._get_scale_factor() * 2.0 / spread2

        self._gradient[:, mask] += gradient[:, mask] * scale

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
