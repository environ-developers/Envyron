from ..representations import EnvironDensity
from ..boundaries import EnvironBoundary


class EnvironDielectric:
    """docstring"""

    def __init__(
        self,
        boundary: EnvironBoundary,
        constant=1.0,
        need_gradient=False,
        need_factsqrt=False,
        need_auxiliary=False,
    ) -> None:
        self.boundary = boundary
        self.constant = constant
        
        self.need_gradient = need_gradient
        if self.need_gradient:
            self.gradient = EnvironDensity(grid=boundary.grid, rank=3)

        self.need_factsqrt = need_factsqrt
        if self.need_factsqrt:
            self.factsqrt = EnvironDensity(grid=boundary.grid)

        self.need_auxiliary = need_auxiliary
        if self.need_auxiliary:
            self.iterative = EnvironDensity(grid=boundary.grid)

        self.epsilon = EnvironDensity(grid=boundary.grid)
        self.depsilon = EnvironDensity(grid=boundary.grid)
        self.gradlogepsilon = EnvironDensity(grid=boundary.grid, rank=3)

        # polarization density
        self.density = EnvironDensity(grid=boundary.grid)

    def update(self) -> None:
        """docstring"""
        raise NotImplementedError

    def of_boundary(self) -> None:
        """docstring"""
        raise NotImplementedError

    def of_potential(self) -> None:
        """docstring"""
        raise NotImplementedError

    def de_dboundary(self) -> None:
        """docstring"""
        raise NotImplementedError

    def dv_dboundary(self) -> None:
        """docstring"""
        raise NotImplementedError
