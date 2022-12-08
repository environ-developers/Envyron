from typing import Optional
from envyron.domains.cell import EnvironGrid
from envyron.representations.density import EnvironDensity
from envyron.physical import (
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

    def add(
        self,
        electrons: Optional[EnvironElectrons] = None,
        ions: Optional[EnvironIons] = None,
        externals: Optional[EnvironExternals] = None,
        dielectric: Optional[EnvironDielectric] = None,
        electrolyte: Optional[EnvironElectrolyte] = None,
        semiconductor: Optional[EnvironSemiconductor] = None,
        additional: Optional[EnvironDensity] = None,
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
        raise NotImplementedError()

    def of_potential(self, potential: EnvironDensity) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()
