from typing import List

from ..utils.constants import FPI, KB_RY, BOHR_RADIUS, AMU
from ..domains import EnvironGrid
from ..representations import EnvironDensity
from ..boundaries import EnvironBoundary


class EnvironIonccType:
    """
    docstring
    """

    def __init__(
        self,
        index,
        cbulk: float,
        charge: int,
        grid: EnvironGrid,
    ) -> None:
        self.index = index
        self.charge = charge
        self.cbulk = cbulk

        self.concentration = \
            EnvironDensity(grid, label=f'c_electrolyte_{index}')

        self.cfactor = \
            EnvironDensity(grid, label=f'cfactor_electrolyte_{index}')


class EnvironElectrolyteBase:
    """docstring"""

    def __init__(
        self,
        temperature: float,
        permittivity: float,
        distance: float,
        spread: float,
        linearized: bool,
        entropy: str,
        ntyp: int,
        cbulk: float,
        formula: List[int],
        grid: EnvironGrid,
        cionmax=0.,
        rion=0.,
    ) -> None:
        self.temperature = temperature
        self.permittivity = permittivity

        self.distance = distance
        self.spread = spread

        self.linearized = linearized
        self.entropy = entropy

        self.ntyp = ntyp
        self.ioncctype = []
        sumcz2 = 0.
        maxcbulk = cbulk

        for i in range(self.ntyp):
            ci = formula[2 * i] * cbulk
            zi = formula[2 * i + 1]
            self.ioncctype.append(EnvironIonccType(i, ci, zi), grid)
            sumcz2 += ci * zi**2
            maxcbulk = max(maxcbulk, ci)

        self.k2 = sumcz2 * FPI / (KB_RY * self.temperature)
        self.cionmax = cionmax * BOHR_RADIUS**3 / AMU

        if cionmax == 0. and rion > 0.:
            self.cionmax = 0.64 * 3. / FPI / rion**3

        if cionmax < maxcbulk:
            raise ValueError(
                "cionmax should be larger than the largest bulk concentration of ions"
            )

    def update(self) -> None:
        """docstring"""
        raise NotImplementedError

    def of_boundary(self) -> None:
        """docstring"""
        raise NotImplementedError

    def of_potential(self) -> None:
        """docstring"""
        raise NotImplementedError

    def energy(self) -> None:
        """docstring"""
        raise NotImplementedError

    def de_dboundary(self) -> None:
        """docstring"""
        raise NotImplementedError


class EnvironElectrolyte:
    """docstring"""

    def __init__(
        self,
        boundary: EnvironBoundary,
        temperature: float,
        permittivity: float,
        distance: float,
        spread: float,
        linearized: bool,
        entropy: str,
        ntyp: int,
        cbulk: float,
        formula: List[int],
        grid: EnvironGrid,
        cionmax=0.,
        rion=0.,
    ) -> None:

        self.base = EnvironElectrolyteBase(
            temperature,
            permittivity,
            distance,
            spread,
            linearized,
            entropy,
            ntyp,
            cbulk,
            formula,
            grid,
            cionmax,
            rion,
        )

        self.boundary = boundary

        self.density = EnvironDensity(grid, label='electrolyte')
        self.charge = 0.

        self.gamma = EnvironDensity(grid, label='gamma')
        self.dgamma = EnvironDensity(grid, label='dgamma')

        if linearized: self.de_dboundary_second_order = EnvironDensity(grid)

        self.energy_second_order = 0.
        self.updating = False

    def update(self) -> None:
        """docstring"""
        raise NotImplementedError

    def of_boundary(self) -> None:
        """docstring"""
        raise NotImplementedError

    def of_potential(self) -> None:
        """docstring"""
        raise NotImplementedError

    def energy(self) -> None:
        """docstring"""
        raise NotImplementedError

    def de_dboundary(self) -> None:
        """docstring"""
        raise NotImplementedError
