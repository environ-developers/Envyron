from numpy import ndarray

import numpy as np

from ..cores import NumericalCore

from ..domains import EnvironGrid
from ..representations import EnvironDensity, EnvironGradient, EnvironHessian
from ..representations.functions import FunctionContainer
from ..utils.constants import FPI, EPS8

from dftpy.field import DirectField, ReciprocalField


class FFTCore(NumericalCore):
    """docstring"""

    def __init__(self, grid: EnvironGrid) -> None:
        super().__init__(grid)
        self.reciprocal_grid = grid.get_reciprocal()

    def gradient(self, density: EnvironDensity) -> EnvironGradient:
        """docstring"""
        density_g = density.fft()

        data = self.reciprocal_grid.g * density_g * 1j

        gradient_g = ReciprocalField(
            self.reciprocal_grid,
            rank=3,
            griddata_3d=data,
        )

        gradient = gradient_g.ifft(force_real=True)
        return EnvironGradient(self.grid, gradient, 'gradient')

    def divergence(self, gradient: EnvironGradient) -> EnvironDensity:
        """docstring"""
        gradient_g = gradient.fft()

        data = np.einsum(
            'l...,l...',
            self.reciprocal_grid.g,
            gradient_g,
        ) * 1j

        divergence_g = ReciprocalField(
            self.reciprocal_grid,
            griddata_3d=data,
        )

        divergence = divergence_g.ifft(force_real=True)
        return EnvironDensity(self.grid, divergence, 'divergence')

    def laplacian(self, density: EnvironDensity) -> EnvironDensity:
        """docstring"""
        density_g = density.fft()

        data = -self.reciprocal_grid.gg * density_g

        laplacian_g = ReciprocalField(
            self.reciprocal_grid,
            griddata_3d=data,
        )

        laplacian = laplacian_g.ifft(force_real=True)
        return EnvironDensity(self.grid, laplacian, 'laplacian')

    def hessian(self, density: EnvironDensity) -> EnvironHessian:
        """docstring"""
        density_g = density.fft()

        hessian_g: np.ndarray = -np.einsum(
            'i...,j...->ji...',
            self.reciprocal_grid.g,
            self.reciprocal_grid.g,
        ) * density_g

        hessian_g = hessian_g.reshape(9, *self.grid.nr)

        hessian = EnvironHessian(self.grid, label='hessian')

        for ipol in np.arange(9):
            aux_g = ReciprocalField(
                self.reciprocal_grid,
                griddata_3d=hessian_g[ipol, :, :, :],
            )
            hessian[ipol, :, :, :] = aux_g.ifft(force_real=True)

        return hessian

    def convolution_density(
        self,
        density_a: EnvironDensity,
        density_b: EnvironDensity,
    ) -> EnvironDensity:
        """docstring"""
        density_a_g = density_a.fft()
        density_b_g = density_b.fft()

        data = density_a_g * density_b_g

        convolution_density_g = ReciprocalField(
            self.reciprocal_grid,
            griddata_3d=data,
        )

        convolution_density = convolution_density_g.iff(force_real=True)
        return convolution_density

    def convolution_gradient(
        self,
        density: EnvironDensity,
        gradient: EnvironGradient,
    ) -> EnvironGradient:
        """docstring"""
        density_g = density.fft()
        gradient_g = gradient.fft()

        data = density_g * gradient_g

        convolution_gradient_g = ReciprocalField(
            self.reciprocal_grid,
            rank=3,
            griddata_3d=data,
        )

        convolution_gradient = convolution_gradient_g.iff(force_real=True)
        return convolution_gradient

    def convolution_hessian(
        self,
        density: EnvironDensity,
        hessian: EnvironHessian,
    ) -> EnvironHessian:
        """docstring"""
        density_g = density.fft()

        convolution_hessian = EnvironHessian(
            self.grid,
            label='convolution_hessian',
        )

        for ipol in np.arange(9):
            aux = DirectField(
                self.grid,
                griddata_3d=hessian[ipol, :, :, :],
            )
            aux_g = aux.fft() * density_g
            convolution_hessian[ipol, :, :, :] = aux_g.ifft(force_real=True)

        return convolution_hessian

    def poisson(self, density: EnvironDensity) -> EnvironDensity:
        """docstring"""
        density_g = density.fft()

        rec = self.reciprocal_grid

        mask = rec.gg < EPS8

        data = np.zeros(rec.gg.shape)
        data[mask] = FPI * density_g[mask] / rec.gg[mask]

        poisson_g = ReciprocalField(rec, griddata_3d=data)

        poisson = poisson_g.ifft(force_real=True)
        return poisson

    def grad_poisson(self, density: EnvironDensity) -> EnvironGradient:
        """docstring"""
        density_g = density.fft()

        rec = self.reciprocal_grid

        mask = rec.gg < EPS8

        data = np.zeros(rec.g.shape)
        data[mask] = FPI * 1j * density_g[mask] * rec.g[:, mask] / rec.gg[mask]

        grad_poisson_g = ReciprocalField(
            rec,
            rank=3,
            griddata_3d=data,
        )

        grad_poisson = grad_poisson_g.ifft(force_real=True)
        return grad_poisson

    def force(self, rho: EnvironDensity, ions: FunctionContainer) -> ndarray:
        """docstring"""
        raise NotImplementedError()

    def hess_v_h_of_rho_r(self, rho: ndarray) -> ndarray:
        """docstring"""
        raise NotImplementedError()

    def field_of_grad_rho(self, grad_rho: ndarray) -> ndarray:
        """docstring"""
        raise NotImplementedError()
