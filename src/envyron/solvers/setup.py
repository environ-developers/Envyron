import numpy as np

from .solver import ElectrostaticSolver
from ..representations import EnvironDensity
from ..physical import EnvironCharges


class ElectrostaticSolverSetup:
    """
    Setup parameters of an electrostatic solver.
    """

    def __init__(
        self,
        problem: str,
        solver: ElectrostaticSolver,
        inner: ElectrostaticSolver,
    ) -> None:
        self.problem = problem
        self.solver = solver
        self.inner = inner

    def solve(self, charges: EnvironCharges) -> EnvironDensity:
        """docstring"""

        if self.problem == 'poisson':
            return self.solver.poisson(charges)
        elif self.problem == 'generalized':
            if not charges.dielectric:
                raise ValueError("missing dielectric")
            return self.solver.generalized(charges)
        elif self.problem in ('linpb', 'linmodpb'):
            if not charges.electrolyte:
                raise ValueError("missing electrolyte")
            return self.solver.linearized_pb(charges)
        elif self.problem in ('pb', 'modpb'):
            if not charges.electrolyte:
                raise ValueError("missing electrolyte")
            if not self.inner:
                if not charges.dielectric:
                    raise ValueError("missing dielectric")
                return self.solver.pb_nested(charges, inner=self.inner)
            else:
                return self.solver.pb_nested(charges)
        else:
            raise ValueError(f'Unsupported problem: {self.problem}')

    def compute_energy(self) -> float:
        """docstring"""
        raise NotImplementedError()

    def compute_force(self) -> np.ndarray:
        """docstring"""
        raise NotImplementedError()
