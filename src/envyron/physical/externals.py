from typing import List, Optional
from numpy import ndarray

from ..domains import EnvironGrid
from ..representations import EnvironDensity
from ..representations.functions import FunctionContainer, EnvironERFC


class EnvironExternals:
    """docstring"""

    def __init__(
        self,
        n: int,
        dims: List[int],
        axes: List[int],
        spreads: List[float],
        charges: List[float],
        grid: EnvironGrid,
        positions: Optional[List[ndarray]] = None,
    ) -> None:
        self.number = n
        self.charge = 0.
        self.density = EnvironDensity(grid, label='externals')

        if n > 0:

            self.functions = FunctionContainer(grid)

            for i in range(n):
                function = EnvironERFC(
                    grid,
                    1,
                    dims[i],
                    axes[i],
                    spreads[i],
                    spreads[i],
                    -charges[i],
                    positions[i],
                )

                self.functions.append(function)

        self.updating = False

    def update(self) -> None:
        """docstring"""
        self.density[:] = self.functions.density()
        self.charge = self.density.charge
