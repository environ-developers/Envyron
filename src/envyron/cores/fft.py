from numpy import ndarray

import numpy as np

from envyron.cores.core import NumericalCore

from ..domains import EnvironGrid
from ..representations import EnvironDensity, EnvironGradient, EnvironHessian
from ..representations.functions import FunctionContainer
from ..utils.constants import FPI, EPS8

from dftpy.field import ReciprocalField
class FFTCore(NumericalCore):
    """docstring"""

    def __init__(self, grid: EnvironGrid) -> None:
        super().__init__(grid)
        self.reciprocal_grid = grid.get_reciprocal()

    def gradient(self, density: EnvironDensity) -> EnvironGradient:
        """docstring"""
        density_g = density.fft()
        imag = 0 + 1j
        gradient_g = self.reciprocal_grid.g * density_g * imag
        gradient_g = ReciprocalField(grid=self.reciprocal_grid, rank=3, griddata_3d=gradient_g)
        gradient = gradient_g.ifft(force_real=True)
        return EnvironGradient(self.grid, gradient, 'gradient')
#        return EnvironGradient(density.grid, density.gradient(), 'gradient')

    def divergence(self, gradient: EnvironGradient) -> EnvironDensity:
        """docstring"""
        gradient_g = gradient.fft()
        imag = 0 + 1j
        divergence_g = np.einsum('l...,l...',self.reciprocal_grid.g,gradient_g) * imag
        divergence_g = ReciprocalField(grid=self.reciprocal_grid, griddata_3d=divergence_g)
        divergence = divergence_g.ifft(force_real=True)
        return EnvironDensity(self.grid, divergence, 'divergence')

    def laplacian(self, density: EnvironDensity) -> EnvironDensity:
        """docstring"""
        density_g = density.fft()
        laplacian_g = -self.reciprocal_grid.gg * density_g
        laplacian_g = ReciprocalField(grid=self.reciprocal_grid, griddata_3d=laplacian_g)
        laplacian = laplacian_g.ifft(force_real=True)
        return EnvironDensity(self.grid, laplacian, 'laplacian')

    def hessian(self, density: EnvironDensity) -> EnvironHessian:
        """docstring"""
        density_g = density.fft()
        hessian_g = -np.einsum('i,j',self.reciprocal_grid.g,self.reciprocal_grid.g) * density_g
        hessian_g = hessian_g.reshape(9, *self.grid.nr)
        hessian = EnvironHessian(self.grid, 'hessian')
        for ipol in np.arange(9):
            aux_g = ReciprocalField(grid=self.reciprocal_grid, griddata_3d=hessian_g[ipol,:,:,:])
            hessian[ipol,:,:,:] = aux_g.ifft(force_real=True)
        return hessian

    def convolution_density(
        self,
        density: EnvironDensity,
        other_density: EnvironDensity,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    def convolution_gradient(
        self,
        density: EnvironDensity,
        gradient: EnvironGradient,
    ) -> EnvironGradient:
        """docstring"""
        raise NotImplementedError()

    def convolution_hessian(
        self,
        density: EnvironDensity,
        hessian: EnvironHessian,
    ) -> EnvironHessian:
        """docstring"""
        raise NotImplementedError()

    def poisson(self, density: EnvironDensity) -> EnvironDensity:
        """docstring"""
        density_g = density.fft()
        mask = self.reciprocal_grid.gg < EPS8
        poisson_g = np.zeros(self.reciprocal_grid.gg.shape)
        poisson_g[mask] = FPI * density_g[mask] / self.reciprocal_grid.gg[mask]
        poisson_g = ReciprocalField(grid=self.reciprocal_grid, griddata_3d=poisson_g)
        poisson = poisson_g.ifft(force_real=True)
        return poisson

    def grad_poisson(self, density: EnvironDensity) -> EnvironGradient:
        """docstring"""
        imag = 0 + 1j
        density_g = density.fft()
        mask = self.reciprocal_grid.gg < EPS8
        grad_poisson_g = np.zeros(self.reciprocal_grid.g.shape)
        grad_poisson_g[:,mask] = FPI * density_g[mask] * imag * self.reciprocal_grid.g[:,mask] / self.reciprocal_grid.gg[mask]
        grad_poisson_g = ReciprocalField(grid=self.reciprocal_grid, rank=3, griddata_3d=grad_poisson_g)
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
