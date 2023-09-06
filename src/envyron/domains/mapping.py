from multimethod import multimethod

import numpy as np

from . import EnvironGrid
from ..representations import EnvironField


class EnvironMapping:
    """docstring"""

    def __init__(
        self,
        nrep: int,
        small: EnvironGrid,
        large: EnvironGrid,
    ) -> None:
        self.nrep = nrep
        self.small = small
        self.large = large

        if self.small is not self.large:
            self.map = np.zeros(self.small.nr)

    def update(self, pos: np.ndarray = None) -> None:
        """docstring"""
        if self.small is self.large: return

    @multimethod
    def to_large(
        self,
        nsmall: int,
        nlarge: int,
        fsmall: np.ndarray,
        flarge: np.ndarray,
    ) -> np.ndarray:
        """docstring"""

    @multimethod
    def to_large(
        self,
        fsmall: EnvironField,
        flarge: EnvironField,
    ) -> EnvironField:
        """docstring"""

    @multimethod
    def to_small(
        self,
        nlarge: int,
        nsmall: int,
        flarge: np.ndarray,
        fsmall: np.ndarray,
    ) -> np.ndarray:
        """docstring"""

    @multimethod
    def to_small(
        self,
        flarge: EnvironField,
        fsmall: EnvironField,
    ) -> EnvironField:
        """docstring"""
