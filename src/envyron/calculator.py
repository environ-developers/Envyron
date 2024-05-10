from numpy import ndarray

from envyron import Main
from envyron.representations import EnvironDensity
from envyron.boundaries import ElectronicBoundary


class Calculator:
    """
    Calculation drivers for potentials and forces.
    """

    def __init__(self, main: Main) -> None:
        self.main = main

    def potential(self, update: bool) -> None:
        """
        docstring
        """

        # if not update write existing potentials and exit
        if not update:
            # TODO: write potentials to cubefiles
            return

        # if update compute new potentials
        self.main.dvtot[:] = 0.

        if self.main.setup.lelectrostatic:
            # reference calculation
            self.main.vreference = self.main.setup.reference.solve(
                self.main.charges)
            # environment calculation
            self.main.velectrostatic = self.main.setup.outer.solve(
                self.main.charges)
            # dvtot
            self.main.dvtot[:] = self.main.velectrostatic[:] - self.main.vreference[:]
            # compute charges that depends on potential
            # (polarization and electrolyte)
            self.main.charges.of_potential(self.main.velectrostatic)

        if self.main.setup.lconfine:
            self.main.vconfine = self.main.solvent.calc_vconfine(
                self.main.setup.confine)
            self.main.dvtot[:] += self.main.vconfine[:]

        if self.main.setup.lsoftcavity:
            self.main.vsoftcavity[:] = 0.
            de_dboundary = EnvironDensity(self.main.setup.cell)
            if self.main.setup.lsoftsolvent:
                if self.main.setup.lsurface:
                    self.main.solvent.calc_desurface_dboundary(
                        self.main.setup.surface_tension, de_dboundary)
                if self.main.setup.lvolume:
                    self.main.solvent.calc_devolume_dboundary(
                        self.main.setup.pressure, de_dboundary)
                if self.main.setup.lconfine:
                    self.main.solvent.calc_deconfine_dboundary(
                        self.main.setup.confine,
                        self.main.charges.electrons.density, de_dboundary)
                if self.main.setup.lstatic:
                    self.main.static.de_dboundary(self.main.velectrostatic,
                                                  de_dboundary)
                if self.main.solvent.solvent_aware:
                    self.main.solvent.calc_solvent_aware_de_dboundary(
                        de_dboundary)
                if type(self.main.solvent) == ElectronicBoundary:
                    self.main.vsoftcavity = de_dboundary * self.main.solvent.dswitch
            if self.main.setup.lsoftelectrolyte:
                de_dboundary[:] = 0.


# TODO:                self.main.electrolyte.de_dboundary(de_dboundary)
#                   if self.main.electrolyte.boundary.solvent_aware:
#                    self.main.electrolyte.boundary.calc_solvent_aware_de_dboundary(de_dboundary)
#                   if type(self.main.electrolyte.boundary) == ElectronicBoundary:
#                       self.main.vsoftcavity = de_dboundary * self.main.electrolyte.boundary.dswitch
            self.main.dvtot[:] += self.main.vsoftcavity[:]

    def energy(self) -> float:
        """docstring"""
        self.main.init_energy()

        self.main.setup.niter_scf += 1

        if self.main.setup.lelectrostatic:
            # reference calculation
            ereference = self.main.setup.reference.compute_energy()
            # environment calculation
            self.main.eelectrostatic = self.main.setup.outer.compute_energy()
            # energy correction
            self.main.eelectrostatic -= ereference

        if self.main.setup.lsurface:
            self.main.esurface = self.main.solvent.calc_esurface(
                self.main.setup.surface_tension)

        if self.main.setup.lvolume:
            self.main.evolume = self.main.solvent.calc_evolume(
                self.main.setup.pressure)

        if self.main.setup.lconfine:
            self.main.econfine = self.main.solvent.calc_econfine(
                self.main.electrons.density, self.main.vconfine)

# TODO:
#        if self.main.setup.lelectrolyte:
#            self.main.eelectrolyte = self.main.electrolyte.energy()

    def force(self) -> ndarray:
        """docstring"""
        raise NotImplementedError

    def response_potential(self) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError

    def denergy(self) -> float:
        """docstring"""
        raise NotImplementedError
