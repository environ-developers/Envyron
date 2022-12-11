from ..domains import EnvironGrid
from ..representations import EnvironDensity
from . import EnvironElectrons
from . import EnvironIons
from . import EnvironExternals
from . import EnvironDielectric
from . import EnvironElectrolyte
from . import EnvironSemiconductor


class EnvironCharges:
    """
    docstring
    """

    def __init__(self, grid: EnvironGrid) -> None:
        self.density = EnvironDensity(grid, label='charges')

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
        raise NotImplementedError()

    def of_potential(self, potential: EnvironDensity) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()
