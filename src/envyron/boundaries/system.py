from typing import Optional

import numpy as np

from ..domains import EnvironGrid
from ..representations import EnvironDensity, EnvironGradient, EnvironHessian
from ..representations.functions import EnvironERFC
from ..physical import EnvironSystem
from ..cores import CoreContainer
from .boundary import EnvironBoundary


class SystemBoundary(EnvironBoundary):
    """docstring"""

    def __init__(
        self,
        distance: float,
        spread: float,
        system: EnvironSystem,
        mode: str,
        need_gradient: bool,
        need_laplacian: bool,
        need_hessian: bool,
        deriv_method: str,
        cores: CoreContainer,
        grid: EnvironGrid,
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

        self.system = system

        self.simple = EnvironERFC(
            grid=self.grid,
            kind=3,
            dim=system.dim,
            axis=system.axis,
            width=distance,
            spread=spread,
            volume=1.0,
            pos=system.com,
        )

    def update(self) -> None:
        """docstring"""

        if self.system.updating:
            self._build()
            self.update_status = 2
        elif self.update_status == 2:
            self.update_status == 0
        else:
            if self.update_status == 2: self.update_status == 0
            return

        self._update_solvent_aware_boundary()

    def dboundary_dions(self, index: int) -> EnvironGradient:
        """docstring"""
        return EnvironGradient(self.grid)

    def _build(self) -> None:
        """docstring"""

        self.switch[:] = self.simple.density

        if self.deriv_level == 3:
            if self.solvent_aware:
                hessian = self.hessian
            else:
                hessian = EnvironHessian(self.grid)

        if self.deriv_method == 'fft':
            self._compute_derivatives_fft(self.switch, hessian)
        elif self.deriv_method == 'chain':

            if self.deriv_level >= 1: self.gradient[:] = self.simple.gradient

            if self.deriv_level == 2: self.laplacian[:] = self.simple.laplacian

            if self.deriv_level == 3:
                hessian[:] = self.simple.hessian
                self.laplacian[:] = hessian.trace
                self.dsurface[:] = self._calc_dsurface(self.gradient, hessian)

        else:
            raise ValueError(f"{self.deriv_method} not supported")

        self.volume = self.switch.charge

        if self.deriv_level >= 1: self.surface = self.gradient.modulus.charge
