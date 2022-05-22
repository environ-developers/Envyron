from typing import List, Optional, Union
from numpy import ndarray

import numpy as np

from ..utils.constants import TPI, E2
from ..domains import EnvironGrid
from ..representations import EnvironDensity
from ..representations.functions import FunctionContainer, EnvironGaussian
from .iontype import EnvironIonType


class EnvironIons:
    """docstring"""

    def __init__(
        self,
        nions: int,
        ntypes: int,
        itypes: List[int],
        ion_ids: List[Union[str, int, float]],
        zv: List[float],
        atomicspread: List[float],
        corespread: List[float],
        solvationrad: List[float],
        radius_mode: str,
        is_soft_cavity: bool,
        smear: bool,
        fill_cores: bool,
        grid: EnvironGrid,
    ) -> None:

        self.count = nions
        self.ntypes = ntypes
        self.itypes = itypes

        self.iontypes: List[EnvironIonType] = []

        self.smeared = smear
        self.filled_cores = fill_cores

        self._set_iontypes(
            ion_ids,
            zv,
            radius_mode,
            atomicspread,
            corespread,
            solvationrad,
            is_soft_cavity,
        )

        self.charge = 0.0

        for i in range(nions):
            self.charge += self.iontypes[self.itypes[i]].zv

        self.coords = np.zeros((nions, 3))

        if smear: self._generate_smeared_ions(grid)

        if fill_cores: self._generate_core_electrons(grid)

        self.com = np.zeros(3)
        self.dipole = np.zeros(3)
        self.quadrupole_pc = np.zeros(3)
        self.quadrupole_gauss = np.zeros(3)
        self.quadrupole_correction = 0.0
        self.selfenergy_correction = 0.0
        self.potential_shift = 0.0

        self.updating = False

    def _set_iontypes(
        self,
        ids: List[Union[str, int, float]],
        zv: List[float],
        radius_mode: str,
        atomicspread: List[float],
        corespread: List[float],
        solvationrad: List[float],
        is_soft_cavity: bool,
    ) -> None:
        """docstring"""

        for i in range(self.ntypes):

            ion = EnvironIonType(
                i,
                ids[i],
                zv[i],
                radius_mode,
                atomicspread[i],
                corespread[i],
                solvationrad[i],
            )

            if not is_soft_cavity and ion.solvationrad == 0.0:
                raise ValueError(f"missing solvation radius for type {i + 1}")

            if self.smeared and ion.atomicspread == 0.0:
                raise ValueError(f"missing atomic spread for type {i + 1}")

            self.iontypes.append(ion)

    def _generate_smeared_ions(self, grid: EnvironGrid) -> None:
        """docstring"""

        self.density = EnvironDensity(grid, label='smeared_ions')

        self.smeared_ions = FunctionContainer(grid)

        for i in range(self.count):

            iontype = self.iontypes[self.itypes[i]]

            ion = EnvironGaussian(
                grid=grid,
                kind=1,
                dim=0,
                axis=0,
                width=0.0,
                spread=iontype.atomicspread,
                volume=iontype.zv,
                pos=self.coords[i],
                label=iontype.label,
            )

            self.smeared_ions.append(ion)

    def _generate_core_electrons(self, grid: EnvironGrid) -> None:
        """docstring"""

        self.core_density = EnvironDensity(grid, label='core_electrons')

        self.core_electrons = FunctionContainer(grid)

        for i in range(self.count):

            iontype = self.iontypes[self.itypes[i]]

            ion = EnvironGaussian(
                grid=grid,
                kind=1,
                dim=0,
                axis=0,
                width=0.0,
                spread=iontype.corespread,
                volume=-iontype.zv,
                pos=self.coords[i],
                label=f"{iontype.label}_core",
            )

            self.core_electrons.append(ion)

    def update(
        self,
        coords: ndarray,
        center: Optional[ndarray] = None,
    ) -> None:
        """docstring"""

        if len(coords) != self.count:
            raise ValueError("mismatch in number of atoms")

        self.coords[:] = coords

        if center is not None:
            self.com = center
        else:
            total_weight = 0.0

            for i in range(self.count):
                weight = self.iontypes[self.itypes[i]].weight
                self.com += self.coords[i] * weight
                total_weight += weight

            self.com /= total_weight

        for i in range(self.count):

            iontype = self.iontypes[self.itypes[i]]

            self.quadrupole_pc += iontype.zv * (self.coords[i] - self.com)**2

            if self.smeared:

                self.quadrupole_correction += \
                    iontype.zv * iontype.atomicspread**2 * 0.5

                self.selfenergy_correction += \
                    iontype.zv**2 / iontype.atomicspread * np.sqrt(2.0 / np.pi)

        if self.smeared:

            for i in range(self.count):
                self.density[:] += self.smeared_ions[i].density

            self.potential_shift = \
                self.quadrupole_correction * \
                TPI * E2 / self.density.grid.volume

            self.quadrupole_gauss = \
                self.quadrupole_pc + self.quadrupole_correction
