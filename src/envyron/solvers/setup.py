from typing import Optional

import numpy as np

from .solver import ElectrostaticSolver
from ..representations import EnvironDensity
from ..physical import EnvironCharges
from ..utils.constants import E2, TPI


class ElectrostaticSolverSetup:
    """
    Setup parameters of an electrostatic solver.
    """

    def __init__(
        self,
        problem: str,
        solver: ElectrostaticSolver,
        inner: Optional['ElectrostaticSolverSetup'] = None,
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

    def compute_energy(self, charges: EnvironCharges, \
                       potential: EnvironDensity,
                       reference: bool) -> float:
        """Calculates the electrostatic embedding contribution to the energy"""
        energy = 0.
        eself = 0.
        degauss = 0.

        # Electrons and nuclei and external charges
        energy = energy + 0.5 * charges.density.scalar_product(potential)
        degauss = degauss + charges.charge

        # Include environment contributions
        if charges.dielectric and not reference:
            degauss = degauss + charges.dielectric.charge * 0.5 # polarization charge

        if charges.electrolyte and not reference:

            # note: electrolyte electrostatic interaction should be negative
            energy = energy - 0.5 * \
                charges.electrolyte.density.scalar_product(potential)
            
            degauss = degauss + charges.electrolyte.charge

        # Adding correction for point-like nuclei: only affects simulations of charged
        # systems, it does not affect forces, but shift the energy depending on the
        # fictitious Gaussian spread of the nuclei
        #
        # Compute spurious self-polarization energy
        eself = charges.ions.selfenergy_correction * E2

        if self.solver.cores.has_internal_correction or \
           self.solver.cores.has_corrections:
            degauss = 0.
        else:
            degauss = -degauss * charges.ions.quadrupole_correction * \
            E2 * TPI / charges.density.cell.volume

        energy = energy + eself + degauss

        return energy

    def compute_force(self) -> np.ndarray:
        """docstring"""
        raise NotImplementedError()
