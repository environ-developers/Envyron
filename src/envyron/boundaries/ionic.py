from copy import deepcopy
from typing import Optional

import numpy as np

from envyron.representations import density

from ..domains import EnvironGrid
from ..representations import EnvironDensity, EnvironGradient, EnvironHessian
from ..representations.functions import FunctionContainer, EnvironERFC
from ..physical import EnvironElectrons, EnvironIons
from ..cores import CoreContainer
from .boundary import EnvironBoundary


class IonicBoundary(EnvironBoundary):
    """docstring"""

    def __init__(
        self,
        alpha: float,
        softness: float,
        ions: EnvironIons,
        mode: str,
        need_gradient: bool,
        need_laplacian: bool,
        need_hessian: bool,
        deriv_method: str,
        cores: CoreContainer,
        grid: EnvironGrid,
        electrons: Optional[EnvironElectrons] = None,
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

        self.alpha = alpha
        self.softness = softness

        self.ions = ions

        if self.field_aware:
            if electrons is None: raise ValueError("missing electrons")
            self.electrons = electrons

            nions = self.ions.count
            self.ion_field = np.zeros(nions)
            self.ion_field_partial = np.zeros((3, nions, nions))  # TODO shape?

            self.dion_field_drho = [
                EnvironDensity(self.grid) for _ in range(nions)
            ]

        self._set_soft_spheres()

    def update(self) -> None:
        """docstring"""

        updating = False
        updating = updating or self.ions.updating
        if self.field_aware: updating = updating or self.electrons.updating

        if not updating:
            if self.update_status == 2: self.update_status == 0
            return

        if self.field_aware:

            if self.ions.updating:
                self._calc_dion_field_drho()
                self.update_status = 1

            elif self.electrons.updating:
                self._calc_ion_field()
                self._update_soft_spheres()
                self._build()

        elif self.ions.updating:
            self._build()
            self.update_status = 2

        else:
            if self.update_status == 2: self.update_status == 0
            return

        self._update_solvent_aware_boundary()

    def dboundary_dions(self, index: int) -> EnvironGradient:
        """docstring"""

        if len(self.soft_spheres) == 0:
            raise ValueError("missing soft spheres")

        partial = EnvironGradient(self.grid)
        partial[:] = self.soft_spheres[index].gradient

        density = EnvironDensity(self.grid)
        mask = [*range(len(self.soft_spheres))]
        mask.remove(index)
        density[mask] = self.soft_spheres[mask].density()

        partial[:] *= density

        return partial

    def _build(self) -> None:
        """docstring"""

        self.soft_spheres.reset_derivatives()

        self.switch[:] = 1.0

        for sphere in self.soft_spheres:
            self.switch[:] *= sphere.density

        if self.deriv_level == 3:
            if self.solvent_aware:
                hessian = self.hessian
            else:
                hessian = EnvironHessian(self.grid)

        if self.deriv_method == 'fft':
            self._compute_derivatives_fft(self.switch, hessian)
        elif self.deriv_method == 'lowmem':

            if self.deriv_level >= 1: self._compute_gradient()

            if self.deriv_level == 2: self._compute_laplacian()

            if self.deriv_level == 3: self._compute_dsurface(hessian)

        else:
            raise ValueError(f"{self.deriv_method} not supported")

        self.switch[:] = 1.0 - self.switch
        self.volume = self.switch.charge

        if self.deriv_level >= 1:
            self.gradient[:] *= -1
            self.surface = self.gradient.modulus.charge

            if self.deriv_level >= 2: self.laplacian[:] *= -1

            if self.deriv_level == 3:
                self.dsurface[:] *= -1

                if self.solvent_aware:
                    hessian[:] *= -1

    def _compute_gradient(self) -> None:
        """docstring"""
        for sphere in self.soft_spheres:
            mask = np.abs(sphere.density) > 1e-60
            if not np.any(mask): continue

            self.gradient[:, mask] += sphere.gradient[:, mask] * \
                                      self.switch[mask] / sphere.density[mask]

    def _compute_laplacian(self) -> None:
        """docstring"""
        for sphere in self.soft_spheres:
            mask = np.abs(sphere.density) > 1e-60
            if not np.any(mask): continue

            den = sphere.density[mask]
            grad = sphere.gradient[:, mask]
            lapl = sphere.laplacian[mask]
            s = self.switch[mask]

            self.laplacian[mask] += lapl / den * s
            self.laplacian[mask] -= np.sum(grad**2, 0) / den**2 * s

            self.laplacian[mask] += \
                np.sum(self.gradient[:, mask] * grad, 0) / den

    def _compute_dsurface(self, hessian: EnvironHessian) -> None:
        """docstring"""
        for sphere in self.soft_spheres:
            mask = np.abs(sphere.density) > 1e-60
            if not np.any(mask): continue

            den = sphere.density[mask]
            grad = sphere.gradient[:, mask]
            hess = sphere.hessian[:, mask]
            s = self.switch[mask]

            hessian[:, mask] += hess[:] / den * s

            shape = hessian[:, mask].shape

            hessian[:, mask] -= np.reshape(
                np.einsum(
                    'i...,j...->ij...',
                    grad,
                    grad,
                ) / den**2 * s,
                shape,
            )

            hessian[:, mask] += np.reshape(
                np.einsum(
                    'i...,j...->ij...',
                    self.gradient[:, mask],
                    grad,
                ) / den,
                shape,
            )

        self.laplacian = hessian.trace
        self.dsurface = self._calc_dsurface(self.gradient, hessian)

    def _set_soft_spheres(self) -> None:
        """docstring"""

        self.soft_spheres = FunctionContainer(self.grid)

        for i in range(self.ions.count):
            iontype = self.ions.iontypes[self.ions.itypes[i]]
            sphere = EnvironERFC(
                grid=self.grid,
                kind=4,
                dim=0,
                axis=0,
                width=iontype.solvationrad * self.alpha,
                spread=self.softness,
                volume=1.0,
                pos=self.ions.coords[i],
                label=f"{iontype.label}_soft_sphere",
            )
            self.soft_spheres.append(sphere)

        if self.field_aware:
            self.unscaled_spheres = deepcopy(self.soft_spheres)

    def _calc_dion_field_drho(self) -> None:
        """docstring"""
        pass

    def _calc_ion_field(self) -> None:
        """docstring"""
        pass

    def _update_soft_spheres(self) -> None:
        """docstring"""
        pass
