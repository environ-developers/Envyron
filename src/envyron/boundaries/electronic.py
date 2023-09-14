from typing import Optional

import numpy as np

from ..domains import EnvironGrid
from ..representations import EnvironDensity, EnvironGradient, EnvironHessian
from ..physical import EnvironElectrons, EnvironIons
from ..cores import CoreContainer
from . import EnvironBoundary

from ..utils.constants import TPI


class ElectronicBoundary(EnvironBoundary):
    """docstring"""

    def __init__(
        self,
        rhomin: float,
        rhomax: float,
        electrons: EnvironElectrons,
        mode: str,
        need_gradient: bool,
        need_laplacian: bool,
        need_hessian: bool,
        deriv_method: str,
        cores: CoreContainer,
        grid: EnvironGrid,
        ions: Optional[EnvironIons] = None,
        label: str = '',
    ) -> None:

        super().__init__(
            mode,
            need_gradient,
            need_laplacian,
            need_hessian,
            deriv_method,
            cores,
            grid,
            label,
        )

        self.rhomax = rhomax
        self.rhomin = rhomin
        self.factor = np.log(rhomax / rhomin)

        self.electrons = electrons
        self.ions = ions

        if self.mode == 'full' and ions is None:
            raise ValueError("missing ions")

        density_label = f"{label}_boundary_density"
        self.density = EnvironDensity(grid, label=density_label)

        density_label = f"{label}_dboundary"
        self.dswitch = EnvironDensity(grid, label=density_label)

        density_label = f"{label}_d2boundary"
        self.d2switch = EnvironDensity(grid, label=density_label)

    def update(self) -> None:
        """docstring"""

        updating = False
        if self.mode == 'full': updating = updating or self.ions.updating
        updating = updating or self.electrons.updating

        if not updating:
            if self.update_status == 2: self.update_status == 0
            return

        if self.mode == 'full':

            if self.ions.updating:
                self.ions.core_density[:] = 0.0

                for core in self.ions.core_electrons:
                    self.ions.core_density += core.density

                self.update_status = 1

            if self.electrons.updating:

                if self.update_status == 0:
                    raise ValueError("missed ionic update step")

                self.density[:] = self.electrons.density + \
                                  self.ions.core_density

                self._build()
                self.update_status = 2

        elif self.mode == 'electronic':

            if self.electrons.updating:
                self.density[:] = self.electrons.density
                self._build()
                self.update_status = 2

            else:
                if self.update_status == 2: self.update_status == 0
                return

        else:
            raise ValueError(f"{self.mode} is not a valid boundary mode")

        self._update_solvent_aware_boundary()

    def dboundary_dions(self, index: int) -> EnvironGradient:
        """docstring"""

        partial = EnvironGradient(self.grid)

        if self.mode == 'electronic': return partial

        if len(self.ions.core_electrons) == 0:
            raise ValueError("missing core electrons")

        partial[:] = self.ions.core_electrons[index].gradient * -self.dswitch

        spurious_force = partial.modulus.charge

        if spurious_force > 1e-5:
            print("non-negligible forces due to core electrons")
            print(f"spurious force on species {index} = {spurious_force}")

        return partial

    def _build(self) -> None:
        """docstring"""

        self._generate_switching_function()

        hessian = None
        if self.deriv_level == 3:
            if self.solvent_aware:
                hessian = self.hessian
            else:
                hessian = EnvironHessian(self.grid)

        if self.deriv_method == 'fft':
            self._compute_derivatives_fft(self.switch, hessian)
        elif self.deriv_method == 'chain':
            self._compute_derivatives_fft(self.density, hessian)

            if self.deriv_level == 3:

                if self.solvent_aware:
                    hessian[:] *= self.dswitch
                    hessian[:] += np.reshape(
                        np.einsum(
                            'i...,j...,...->ji...',
                            self.gradient,
                            self.gradient,
                            self.d2switch,
                        ),
                        (9, *self.grid.nr),
                    )

            if self.deriv_level > 1:
                self.laplacian[:] *= self.dswitch
                self.laplacian[:] += np.sum(self.gradient**2, 0) * self.dswitch

            if self.deriv_level >= 1: self.gradient[:] *= self.dswitch

        else:
            raise ValueError(f"{self.deriv_method} not supported")

        self.switch.compute_charge()
        self.volume = self.switch.charge

        if self.deriv_level >= 1:
            self.gradient.modulus.compute_charge()
            self.surface = self.gradient.modulus.charge

    def _generate_switching_function(self) -> None:
        """docstring"""

        self.switch[:] = 0.
        self.dswitch[:] = 0.
        self.d2switch[:] = 0.

        mask = self.density <= self.rhomin
        if np.any(mask): self.switch[mask] = 1.0

        mask = (self.rhomin < self.density) & (self.density < self.rhomax)
        if not np.any(mask): return

        arg = np.log(self.rhomax / np.abs(self.density[mask])) * \
              TPI / self.factor

        self.switch[mask] = (arg - np.sin(arg)) / TPI
        self.switch[:] = 1.0 - self.switch

        self.dswitch[mask] = -(np.cos(arg) - 1.0) / \
                               np.abs(self.density[mask]) / self.factor

        self.d2switch[mask] = \
            -(TPI * np.sin(arg) + self.factor * (1.0 - np.cos(arg))) / \
             (self.density[mask] * self.factor)**2
