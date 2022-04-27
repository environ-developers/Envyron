from __future__ import annotations

from typing import Iterator, List, Union

from ...domains import EnvironGrid
from .. import EnvironDensity, EnvironGradient, EnvironHessian
from . import EnvironFunction


class FunctionContainer:
    """docstring"""

    def __init__(self, grid: EnvironGrid) -> None:
        self.grid = grid
        self.functions: List[EnvironFunction] = []

    def __getitem__(
        self,
        s: Union[int, List[int], slice],
    ) -> Union[FunctionContainer, EnvironFunction]:

        if isinstance(s, int):
            return self.functions[s]
        else:
            subset = FunctionContainer(self.grid)

            if isinstance(s, list):
                for i in s:
                    subset.functions.append(self.functions[i])
            else:
                for function in self.functions[s]:
                    subset.functions.append(function)

            return subset

    def __iter__(self) -> Iterator[EnvironFunction]:
        return iter(self.functions)

    def __len__(self) -> int:
        return len(self.functions)

    def append(self, function: EnvironFunction) -> None:
        """docstring"""
        self.functions.append(function)

    def density(self) -> EnvironDensity:
        """docstring"""
        density = EnvironDensity(self.grid)

        function: EnvironFunction
        for function in self:
            density[:] += function.density()

        return density

    def gradient(self) -> EnvironGradient:
        """docstring"""
        gradient = EnvironGradient(self.grid)

        function: EnvironFunction
        for function in self:
            gradient[:] += function.gradient()

        return gradient

    def laplacian(self) -> EnvironDensity:
        """docstring"""
        laplacian = EnvironDensity(self.grid)

        function: EnvironFunction
        for function in self:
            laplacian[:] += function.laplacian()

        return laplacian

    def hessian(self) -> EnvironHessian:
        """docstring"""
        hessian = EnvironHessian(self.grid)

        function: EnvironFunction
        for function in self:
            hessian[:] += function.hessian()

        return hessian

    def derivative(self) -> EnvironDensity:
        """docstring"""
        derivative = EnvironDensity(self.grid)

        function: EnvironFunction
        for function in self:
            derivative[:] += function.derivative()

        return derivative
