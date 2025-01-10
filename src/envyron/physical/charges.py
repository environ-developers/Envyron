from ..domains import EnvironGrid
from ..representations import EnvironDensity
from . import (
    EnvironElectrons,
    EnvironIons,
    EnvironExternals,
    EnvironDielectric,
    EnvironElectrolyte,
    EnvironSemiconductor,
)


class EnvironCharges:
    """
    docstring
    """

    def __init__(self, grid: EnvironGrid) -> None:
        self.density = EnvironDensity(grid, label='charges')
        self.electrons = None
        self.ions = None
        self.externals = None
        self.dielectric = None
        self.electrolyte = None
        self.semiconductor = None
        self.additional = None

    def add(
        self,
        electrons: EnvironElectrons = None,
        ions: EnvironIons = None,
        externals: EnvironExternals = None,
        dielectric: EnvironDielectric = None,
        electrolyte: EnvironElectrolyte = None,
        semiconductor: EnvironSemiconductor = None,
        additional: EnvironDensity = None,
    ):
        """docstring"""
        if electrons: self.electrons = electrons
        if ions: self.ions = ions
        if externals: self.externals = externals
        if dielectric: self.dielectric = dielectric
        if electrolyte: self.electrolyte = electrolyte
        if semiconductor: self.semiconductor = semiconductor
        if additional: self.additional = additional

    def update(self) -> None:
        """docstring"""
        self.count = 0
        self.charge = 0.
        self.density[:] = 0.

        if self.electrons:
            self.count = self.count + self.electrons.count
            self.charge = self.charge + self.electrons.charge
            self.density[:] = self.density[:] + self.electrons.density[:]

        if self.ions:
            self.count = self.count + self.ions.count
            self.charge = self.charge + self.ions.charge
            self.density[:] = self.density[:] + self.ions.density[:]

        if self.externals:
            self.count = self.count + self.externals.count
            self.charge = self.charge + self.externals.charge
            self.density[:] = self.density[:] + self.externals.density[:]

        if self.additional:
            self.charge = self.charge + self.additional.charge
            self.density[:] = self.density[:] + self.additional[:]

        local_charge = self.density.integral()
        error = abs(local_charge - self.charge)
        if error > 1e-5:
            raise ValueError(f"{error:.2e} error in integrated charges")

    def of_potential(self, potential: EnvironDensity) -> EnvironDensity:
        """docstring"""
        total_charge_density = EnvironDensity(self.density.grid)
        total_charge_density[:] = self.density[:]

        if self.electrolyte:
            self.electrolyte.base.of_potential(potential)
            total_charge_density[:] += self.electrolyte.density[:]

        if self.dielectric:
            self.dielectric.of_potential(total_charge_density, potential)
