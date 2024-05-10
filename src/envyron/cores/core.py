from numpy import ndarray

from abc import ABC

from multimethod import multimethod

from ..domains import EnvironGrid
from ..representations import EnvironDensity, EnvironGradient, EnvironHessian
from ..representations.functions import FunctionContainer


class NumericalCore(ABC):
    """docstring"""

    def __init__(self, grid: EnvironGrid) -> None:
        self.grid = grid

    def gradient(self, density: EnvironDensity) -> EnvironGradient:
        """docstring"""
        raise NotImplementedError()

    def divergence(self, gradient: EnvironGradient) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    def laplacian(self, density: EnvironDensity) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    def hessian(self, density: EnvironDensity) -> EnvironGradient:
        """docstring"""
        raise NotImplementedError()

    @multimethod
    def convolution(
        self,
        density: EnvironDensity,
        other_density: EnvironDensity,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @multimethod
    def convolution(
        self,
        density: EnvironDensity,
        gradient: EnvironGradient,
    ) -> EnvironGradient:
        """docstring"""
        raise NotImplementedError()

    @multimethod
    def convolution(
        self,
        density: EnvironDensity,
        hessian: EnvironHessian,
    ) -> EnvironHessian:
        """docstring"""
        raise NotImplementedError()

    def poisson(self, rho: EnvironDensity) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    def grad_poisson(self, rho: EnvironDensity) -> EnvironGradient:
        """docstring"""
        raise NotImplementedError()

    def force(self, rho: EnvironDensity, ions: FunctionContainer) -> ndarray:
        """docstring"""
        raise NotImplementedError()

    def grad_v_h_of_rho_r(self, rho: ndarray) -> ndarray:
        """docstring"""
        raise NotImplementedError()

    def hess_v_h_of_rho_r(self, rho: ndarray) -> ndarray:
        """docstring"""
        raise NotImplementedError()

    def field_of_grad_rho(self, grad_rho: ndarray) -> ndarray:
        """docstring"""
        raise NotImplementedError()
