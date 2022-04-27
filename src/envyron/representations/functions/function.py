from numpy import ndarray

import numpy as np
from abc import ABC, abstractmethod

from ...domains import EnvironGrid
from .. import EnvironDensity, EnvironGradient, EnvironHessian

KINDS = {
    1: 'gaussian',
    2: 'erfc',
    3: 'scaled erfc',
    4: 'scaled erf',
}

EXP_TOL = 4e1
FUNC_TOL = 1e-10


class EnvironFunction(ABC):
    """docstring"""

    def __init__(
        self,
        grid: EnvironGrid,
        kind: int,
        dim: int,
        axis: int,
        width: float,
        spread: float,
        volume: float,
        pos: ndarray = np.zeros(3),
        label: str = '',
    ) -> None:
        self.kind = kind
        self.dim = dim
        self.axis = axis
        self.width = width
        self.spread = spread
        self.volume = volume
        self.pos = pos
        self.grid = grid
        self.label = label

    @property
    def kind(self) -> int:
        """docstring"""
        return self.__kind

    @kind.setter
    def kind(self, kind: int) -> None:
        """docstring"""
        if kind not in KINDS:
            raise ValueError(f"{kind} is not a valid kind of function (1-4)")
        self.__kind = kind

    @property
    def dim(self) -> int:
        """docstring"""
        return self.__dim

    @dim.setter
    def dim(self, dim: int) -> None:
        """docstring"""
        if not 0 <= dim <= 2: raise ValueError("dim out of range")
        self.__dim = dim

    @property
    def axis(self) -> int:
        """docstring"""
        return self.__axis

    @axis.setter
    def axis(self, axis: int) -> None:
        """docstring"""
        if not 0 <= axis <= 2: raise ValueError("axis out of range")
        self.__axis = axis

    @property
    def spread(self) -> float:
        """docstring"""
        return self.__spread

    @spread.setter
    def spread(self, spread: float) -> None:
        """docstring"""
        if np.abs(spread < FUNC_TOL):
            raise ValueError(f"wrong spread for {self.kind} function")
        self.__spread = spread

    @abstractmethod
    def density(self) -> EnvironDensity:
        """docstring"""

    @abstractmethod
    def gradient(self) -> EnvironGradient:
        """docstring"""

    def laplacian(self) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError(
            f"not implemented for {KINDS[self.kind]} functions")

    def hessian(self) -> EnvironHessian:
        """docstring"""
        raise NotImplementedError(
            f"not implemented for {KINDS[self.kind]} functions")

    def derivative(self) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError(
            f"not implemented for {KINDS[self.kind]} functions")
