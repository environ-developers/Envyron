from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..domains import EnvironGrid
from ..representations import EnvironDensity, EnvironGradient, EnvironHessian
from ..representations.functions import EnvironERFC
from ..cores import CoreContainer


class EnvironBoundary(ABC):
    """docstring"""

    def __init__(
        self,
        mode: str,
        need_gradient: bool,
        need_laplacian: bool,
        need_hessian: bool,
        deriv_method: str,
        cores: CoreContainer,
        grid: EnvironGrid,
        label: str = '',
    ) -> None:
        self.mode = mode
        self.label = label
        self.update_status = 0
        self.deriv_method = deriv_method
        self.volume = 0.0
        self.surface = 0.0
        self.solvent_aware = False
        self.field_aware = False

        if need_hessian:
            self.deriv_level = 3
        elif need_laplacian:
            self.deriv_level = 2
        elif need_gradient:
            self.deriv_level = 1
        else:
            self.deriv_level = 0

        self.grid = grid
        self.cores = cores

        boundary_label = f"{label}_boundary"
        self.switch = EnvironDensity(grid, label=boundary_label)

        if self.deriv_level >= 1:
            gradient_label = f"{label}_boundary_gradient"
            self.gradient = EnvironGradient(grid, label=gradient_label)

        if self.deriv_level >= 2:
            laplacian_label = f"{label}_boundary_laplacian"
            self.laplacian = EnvironDensity(grid, label=laplacian_label)

        if self.deriv_level >= 3:
            dsurface_label = f"{label}_boundary_dsurface"
            self.dsurface = EnvironDensity(grid, label=dsurface_label)

    def activate_solvent_awareness(
        self,
        solvent_radius: float,
        radial_scale: float,
        radial_spread: float,
        filling_threshold: float,
        filling_spread: float,
    ) -> None:
        """docstring"""

        self.solvent_aware = True
        self.filling_threshold = filling_threshold
        self.filling_spread = filling_spread

        self.solvent_probe = EnvironERFC(
            grid=self.grid,
            kind=2,
            dim=0,
            axis=0,
            width=solvent_radius * radial_scale,
            spread=radial_spread,
            volume=1.0,
        )

        local_label = f"{self.label}_local"
        self.local = EnvironDensity(self.grid, label=local_label)

        probe_label = f"{self.label}_probe"
        self.probe = EnvironDensity(self.grid, label=probe_label)

        filling_label = f"{self.label}_filling"
        self.filling = EnvironDensity(self.grid, label=filling_label)

        dfilling_label = f"{self.label}_dfilling"
        self.dfilling = EnvironDensity(self.grid, label=dfilling_label)

        if self.deriv_level >= 3:
            hessian_label = f"{self.label}_boundary_hessian"
            self.hessian = EnvironHessian(self.grid, label=hessian_label)

    def activate_field_awareness(
        self,
        field_factor: float,
        field_asymmetry: float,
        field_max: float,
        field_min: float,
    ) -> None:
        """docstring"""
        self.field_aware = True
        self.field_factor = field_factor
        self.field_asymmetry = field_asymmetry
        self.field_max = field_max
        self.field_min = field_min

    def calc_vconfine(
        self,
        confine: float,
    ) -> EnvironDensity:
        """docstring"""
        return confine * (1. - self.switch[:])

    def calc_econfine(
        self,
        rho: EnvironDensity,
        vconfine: EnvironDensity,
    ) -> float:
        """docstring"""
        return rho.scalar_product(vconfine)

    def calc_deconfine_dboundary(
        self,
        confine: float,
        rho: EnvironDensity,
        de_dboundary: EnvironDensity,
    ) -> None:
        """docstring"""
        de_dboundary -= confine * rho[:]

    def calc_evolume(
        self,
        pressure: float,
    ) -> float:
        """docstring"""
        return pressure * self.volume

    def calc_devolume_dboundary(
        self,
        pressure: float,
        de_dboundary: EnvironDensity,
    ) -> None:
        """docstring"""
        de_dboundary += pressure

    def calc_esurface(
        self,
        surface_tension: float,
    ) -> float:
        """docstring"""
        return surface_tension * self.surface

    def calc_desurface_dboundary(
        self,
        surface_tension: float,
        de_dboundary: EnvironDensity,
    ) -> None:
        """docstring"""
        de_dboundary += surface_tension * self.dsurface

    def calc_solvent_aware_de_dboundary():
        """docstring"""
        raise NotImplementedError

    @abstractmethod
    def update(self) -> None:
        """docstring"""

    @abstractmethod
    def dboundary_dions(self, index: int) -> EnvironGradient:
        """docstring"""

    @abstractmethod
    def _build(self) -> None:
        """docstring"""

    def _compute_derivatives_fft(
        self,
        density: EnvironDensity,
        hessian: EnvironHessian,
    ) -> None:
        """docstring"""

        if self.deriv_level == 1:
            self.gradient[:] = self.cores.derivatives.gradient(density)

        if self.deriv_level == 2:
            self.laplacian[:] = self.cores.derivatives.laplacian(density)

        if self.deriv_level == 3:
            self.dsurface[:] = self._calc_dsurface(
                self.gradient,
                hessian,
                pre_compute=True,
                density=density,
                laplacian=self.laplacian,
            )

    def _calc_dsurface(
        self,
        gradient: EnvironGradient,
        hessian: EnvironHessian,
        pre_compute: bool = False,
        density: Optional[EnvironDensity] = None,
        laplacian: Optional[EnvironDensity] = None,
    ) -> EnvironDensity:
        """docstring"""

        if pre_compute:

            for field in (density, laplacian):
                if field is None: raise ValueError(f"missing {field}")

            hessian[:] = self.cores.derivatives.hessian(density)
            laplacian[:] = hessian.trace

        dsurface = EnvironDensity(gradient.grid)

        grad_mod2 = np.sum(gradient**2, 0)
        mask = grad_mod2 >= 1e-50

        g = gradient[:, mask]
        h = hessian.reshape(3, 3, *gradient.grid.nr)[:, :, mask]

        dsurface[mask] += np.einsum('i...,j...,ij...', g, g, h) - \
                          np.einsum('i...,i...,jj...', g, g, h)

        dsurface[mask] /= grad_mod2[mask] / np.sqrt(grad_mod2[mask])

        return dsurface

    def _update_solvent_aware_boundary(self) -> None:
        """docstring"""
        if self.solvent_aware and self.update_status == 2:
            self._build_solvent_aware_boundary()

    def _build_solvent_aware_boundary(self) -> None:
        """docstring"""
        pass
