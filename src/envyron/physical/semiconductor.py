import numpy as np

from ..domains import EnvironGrid
from ..representations import EnvironDensity
from ..representations.functions import EnvironERFC
from ..physical import EnvironSystem


class EnvironSemiconductorBase:
    """docstring"""

    def __init__(
        self,
        temperature: float,
        permittivity: float,
        carrier_density: float,
        electrode_charge: float,
        distance: float,
        spread: float,
        charge_threshold: float,
        need_flatband: bool,
        naxis: int,
    ) -> None:

        self.temperature = temperature
        self.permittivity = permittivity

        # convert carrier density to internal units (1/bohr**3)
        self.carrier_density = carrier_density * 1.48e-25

        self.electrode_charge = electrode_charge
        self.charge_threshold = charge_threshold

        # parameters of the simple semiconductor interface
        self.distance = distance
        self.spread = spread

        self.need_flatband = need_flatband
        if self.need_flatband:
            self.flatband_potential_planar_avg = np.zeros(naxis)


class EnvironSemiconductor:
    """docstring"""

    def __init__(
        self,
        base: EnvironSemiconductorBase,
        system: EnvironSystem,
        grid: EnvironGrid,
    ) -> None:

        self.base = base

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
