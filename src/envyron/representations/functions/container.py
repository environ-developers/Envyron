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
        self.count = 0

    def __getitem__(
        self,
        slice: Union[int, List[int], slice],
    ) -> Union[EnvironFunction, FunctionContainer]:

        if isinstance(slice, int):
            return self.functions[slice]
        else:
            subset = FunctionContainer(self.grid)

            if isinstance(slice, list):
                for i in slice:
                    subset.functions.append(self.functions[i])
            else:
                for function in self.functions[slice]:
                    subset.functions.append(function)

            return subset

    def __iter__(self) -> Iterator[EnvironFunction]:
        return iter(self.functions)

    def __len__(self) -> int:
        return self.count

    def append(self, function: EnvironFunction) -> None:
        """docstring"""
        self.functions.append(function)
        self.count += 1

    def reset_derivatives(self) -> None:
        """docstring"""
        for function in self:
            function.reset_derivatives()

    def density(self) -> EnvironDensity:
        """docstring"""
        density = EnvironDensity(self.grid)

        for function in self:
            density[:] += function.density

        return density

    def gradient(self) -> EnvironGradient:
        """docstring"""
        gradient = EnvironGradient(self.grid)

        for function in self:
            gradient[:] += function.gradient

        return gradient

    def laplacian(self) -> EnvironDensity:
        """docstring"""
        laplacian = EnvironDensity(self.grid)

        for function in self:
            laplacian[:] += function.laplacian

        return laplacian

    def hessian(self) -> EnvironHessian:
        """docstring"""
        hessian = EnvironHessian(self.grid)

        for function in self:
            hessian[:] += function.hessian

        return hessian

    def derivative(self) -> EnvironDensity:
        """docstring"""
        derivative = EnvironDensity(self.grid)

        for function in self:
            derivative[:] += function.derivative

        return derivative
