import numpy as np

from envyron.io.input.base import SemiconductorModel

from ..domains import EnvironGrid
from ..representations import EnvironDensity
from ..representations.functions import EnvironERFC
from ..physical import EnvironSystem


class EnvironSemiconductorBase:
    """docstring"""

    def __init__(
        self,
        semiconductor: SemiconductorModel,
        temperature: float,
        need_flatband: bool,
        naxis: int,
    ) -> None:

        self.temperature = temperature
        self.permittivity = semiconductor.permittivity

        # convert carrier density to internal units (1/bohr**3)
        self.carrier_density = semiconductor.carrier_density * 1.48e-25

        self.electrode_charge = semiconductor.electrode_charge
        self.charge_threshold = semiconductor.charge_threshold

        # parameters of the simple semiconductor interface
        self.distance = semiconductor.distance
        self.spread = semiconductor.spread

        self.need_flatband = need_flatband
        if self.need_flatband:
            self.flatband_potential_planar_avg = np.zeros(naxis)


class EnvironSemiconductor:
    """docstring"""

    def __init__(
        self,
        semiconductor: SemiconductorModel,
        temperature: float,
        need_flatband: bool,
        system: EnvironSystem,
        grid: EnvironGrid,
    ) -> None:

        self.base = EnvironSemiconductorBase(
            semiconductor,
            temperature,
            need_flatband,
            grid.nr[2],
        )

        self.density = EnvironDensity(grid, label='semiconductor')

        self.simple = EnvironERFC(
            grid,
            3,
            system.dim,
            system.axis,
            self.base.distance,
            self.base.spread,
            1.0,
            system.com,
        )
