from typing import List, Optional, Union
from numpy import ndarray

from envyron import Setup
from envyron.representations import EnvironDensity

from envyron.physical import EnvironIons
from envyron.physical import EnvironElectrons
from envyron.physical import EnvironSystem
from envyron.physical import EnvironCharges
from envyron.physical import EnvironExternals
from envyron.physical import EnvironDielectric

from envyron.boundaries import ElectronicBoundary, IonicBoundary, SystemBoundary


class Main:
    """
    Dynamic quantities of an Environ simulation, including
    physical quantities, densities, and energies.
    """

    def __init__(
        self,
        setup: Setup,
        nions: int,
        ntypes: int,
        itypes: List[int],
        zv: List[float],
        ion_ids: Union[List[str], List[int], List[float]],
    ):
        """
        docstring
        """
        # Save internal copy of setup (processed input values)
        self.setup = setup
        # Merge system properties and setup to generate physical classes
        self._init_physical(nions, ntypes, itypes, zv, ion_ids)

        # Initialize energies and potentials
        self.init_energy()
        self._init_potential()

        # Set initialization flag
        self.initialized = True

    def update_potential(
        self,
        potential: EnvironDensity,
    ):
        """
        docstring
        """
        self.vzero[:] = potential[:]

    def update_cell_dependent_quantities(self):
        """
        docstring
        """
        if self.setup.lstatic: self.static.update()
        if self.setup.loptical: self.optical.update()

#        if self.setup.lelectrolyte: TODO
#        if self.setup.lexternals: TODO
#        if self.setup.lsemiconductor: TODO

    def update_ions(
        self,
        coords: ndarray,
        center: Optional[ndarray] = None,
    ):
        """
        docstring
        """
        self.ions.updating = True
        self.system.updating = True
        # Update counter on ionic steps
        self.setup.niter_ionic += 1
        # Update ionic coordinates
        self.ions.update(coords, center)
        # Update system's properties
        self.system.update(center)
        # Update system's charges
        if self.setup.lconfine or self.setup.lelectrostatic:
            self.charges.update()
        # Update Cores
        if self.setup.l1da:
            self.setup.analytic1d.update_origin(self.system.com)
        # Update properties that depend on rigid (ionic) boundaries
        if self.setup.lrigidcavity:
            if self.setup.lsolvent:
                self.solvent.update()
                if self.setup.lstatic: self.static.update()
                if self.setup.loptical: self.optical.update()
            if self.setup.lelectrolyte:
                raise NotImplementedError
# TODO:          self.electrolyte.boundary.update()
#                self.electrolyte.update()
# Update external charges
        if self.setup.lexternals:
            raise NotImplementedError
# TODO:          self.externals.update()

        self.ions.updating = False
        self.system.updating = False

    def update_electrons(
        self,
        density: EnvironDensity,
        nelec: Optional[int] = None,
    ):
        """
        docstring
        """
        self.electrons.updating = True
        # Update electronic density
        self.electrons.update(density, nelec)
        # Update system's charges
        if self.setup.lconfine or self.setup.lelectrostatic:
            self.charges.update()
        # Update properties that depend on soft (electronic) boundaries
        if self.setup.lsoftcavity:
            if self.setup.lsolvent:
                self.solvent.update()
                if self.setup.lstatic: self.static.update()
                if self.setup.loptical: self.optical.update()
            if self.setup.lelectrolyte:
                raise NotImplementedError


# TODO:           self.electrolyte.boundary.update()
#                 self.electrolyte.update()
        self.electrons.updating = False

    def update_response(
        self,
        drho: EnvironDensity,
    ):
        """
        docstring
        """
        raise NotImplementedError

    def init_energy(self):
        """
        docstring
        """
        self.evolume = 0.
        self.esurface = 0.
        self.econfine = 0.
        self.deenviron = 0.
        self.eelectrolyte = 0.
        self.eelectrostatic = 0.

    def _init_potential(self):
        """
        docstring
        """
        self.vzero = EnvironDensity(self.setup.cell)
        self.dvtot = EnvironDensity(self.setup.cell)
        if self.setup.lelectrostatic:
            self.velectrostatic = EnvironDensity(self.setup.cell)
            self.vreference = EnvironDensity(self.setup.cell)
        if self.setup.lsoftcavity:
            self.vsoftcavity = EnvironDensity(self.setup.cell)
        if self.setup.lconfine:
            self.vconfine = EnvironDensity(self.setup.cell)

    def _init_physical(
        self,
        nions: int,
        ntypes: int,
        itypes: List[int],
        zv: List[float],
        ion_ids: Union[List[str], List[int], List[float]],
    ):
        """
        docstring
        """
        # TODO: initialize response properties

        # Ions
        self.ions = EnvironIons(nions, ntypes, itypes, ion_ids, zv,
                                self.setup.input.ions.atomicspread,
                                self.setup.input.ions.corespread,
                                self.setup.input.ions.solvationrad,
                                self.setup.input.solvent.radius_mode,
                                self.setup.lsoftcavity,
                                self.setup.lsmearedions,
                                self.setup.lcoredensity, self.setup.cell)

        # Electrons
        self.electrons = EnvironElectrons(self.setup.cell)

        # System
        self.system = EnvironSystem(self.setup.input.system.ntyp,
                                    self.setup.input.system.dim,
                                    self.setup.input.system.axis, self.ions)

        # Charges
        self.charges = EnvironCharges(self.setup.cell)
        if self.setup.lconfine or self.setup.lelectrostatic:
            self.charges.add(self.electrons)
        if self.setup.lelectrostatic:
            self.charges.add(self.ions)

        # External Charges TODO: Need to convert input externals for setup

        # Solvent Boundary
        if self.setup.lsolvent:
            if self.setup.input.solvent.mode in ('electronic', 'full'):
                self.solvent = ElectronicBoundary(
                    self.setup.input.solvent.rhomin,
                    self.setup.input.solvent.rhomax,
                    self.electrons,
                    self.setup.input.solvent.mode,
                    self.setup.need_gradient,
                    self.setup.need_factsqrt,
                    self.setup.lsurface,
                    self.setup.input.solvent.deriv_method,
                    self.setup.environment_core,
                    self.setup.cell,
                    self.ions,
                    label='solvent')
            elif self.setup.input.solvent.mode == 'ionic':
                self.solvent = IonicBoundary(
                    self.setup.input.solvent.alpha,
                    self.setup.input.solvent.softness,
                    self.ions,
                    self.setup.input.solvent.mode,
                    self.setup.need_gradient,
                    self.setup.need_factsqrt,
                    self.setup.lsurface,
                    self.setup.input.solvent.deriv_method,
                    self.setup.environment_core,
                    self.setup.cell,
                    self.electrons,
                    label='solvent')
            elif self.setup.input.solvent.mode == 'system':
                self.solvent = SystemBoundary(
                    self.setup.input.solvent.distance,
                    self.setup.input.solvent.spread,
                    self.system,
                    self.setup.input.solvent.mode,
                    self.setup.need_gradient,
                    self.setup.need_factsqrt,
                    self.setup.lsurface,
                    self.setup.input.solvent.deriv_method,
                    self.setup.environment_core,
                    self.setup.cell,
                    self.electrons,
                    label='solvent')
            else:
                raise ValueError('Unexpected value for solvent mode')

        # Boundary Awareness
        if (self.setup.input.solvent.radius > 0.):
            self.solvent.activate_solvent_awareness(
                self.setup.input.solvent.radius,
                self.setup.input.solvent.radial_scale,
                self.setup.input.solvent.radial_spread,
                self.setup.input.solvent.filling_threshold,
                self.setup.input.solvent.filling_spread)

        if (self.setup.input.solvent.field_aware):
            self.solvent.activate_field_awareness(
                self.setup.input.solvent.field_factor,
                self.setup.input.solvent.field_asymmetry,
                self.setup.input.solvent.field_max,
                self.setup.input.solvent.field_min)

        # Electrolyte TODO

        # Semiconductor TODO

        # Dielectric
        if self.setup.lstatic:
            self.static = EnvironDielectric(self.solvent,
                                            self.setup.static_permittivity,
                                            self.setup.need_gradient,
                                            self.setup.need_factsqrt,
                                            self.setup.need_auxiliary)

        if self.setup.loptical:
            self.optical = EnvironDielectric(self.solvent,
                                             self.setup.optical_permittivity,
                                             self.setup.need_gradient,
                                             self.setup.need_factsqrt,
                                             self.setup.need_auxiliary)
