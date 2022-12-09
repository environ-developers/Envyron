from abc import ABC

from .solver import ElectrostaticSolver
from .direct import DirectSolver
from ..cores import CoreContainer


class IterativeSolver(ElectrostaticSolver, ABC):
    """
    docstring
    """

    def __init__(
        self,
        cores: CoreContainer,
        direct: DirectSolver,
        maxiter: int,
        tol: float,
        auxiliary: str = '',
    ) -> None:
        super().__init__(cores)
        self.direct = direct
        self.maxiter = maxiter
        self.tol = tol
        self.auxiliary = auxiliary
