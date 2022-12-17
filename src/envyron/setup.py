from envyron.utils.constants import BOHR_RADIUS, RYDBERG

from envyron.io.input.input import Input


class Setup:
    """
    Simulation details of an Environ calculation, including the
    simulation parameters, simulation cells, cell mappings, numerical
    cores, and electrostatic solvers.
    """

    def __init__(self, user_input: Input) -> None:
        self.input = user_input
        self.niter_scf = 0
        self.niter_ionic = 0

        self.system_cell = None
        self.environment_cell = None
        self.mapping = None

        self.need_gradient = False
        self.need_factsqrt = False
        self.need_auxiliary = False

        self.has_numerical_setup = False

        self._set_flags()

    def _set_flags(self) -> None:
        """docstring"""
        self._set_execution_flags()
        self._set_simulation_flags()
        self._set_environment_flags()
        self._set_derived_flags()
        self._set_numerical_flags()

    def _set_execution_flags(self) -> None:
        """docstring"""
        self.restart = self.input.control.restart
        self.threshold = self.input.control.threshold
        self.nskip = self.input.control.nskip

    def _set_simulation_flags(self) -> None:
        """docstring"""
        self.ldoublecell = sum(self.input.control.nrep) > 0

        correction = self.input.pbc.correction

        self.lperiodic = correction != 'none'

        self.lsemiconductor = False
        self.laddcharges = False
        self.ltddfpt = False
        self.lmsgcs = False

        if correction == 'gcs':
            self.lelectrolyte = True
        elif correction == 'ms':
            self.lsemiconductor = True
        elif correction == 'ms-gcs':
            self.lsemiconductor = True
            self.lmsgcs = True
        else:
            raise ValueError(f"Unexpected correction type: {correction}")

    def _set_environment_flags(self) -> None:
        """docstring"""
        environment = self.input.environment

        factor = 1e-3 / RYDBERG * BOHR_RADIUS**2
        self.surface_tension = environment.surface_tension * factor
        self.lsurface = self.surface_tension != 0  # or input.lsurface

        factor = 1.e9 / RYDBERG * BOHR_RADIUS**3
        self.pressure = environment.pressure * factor
        self.lvolume = self.pressure != 0  # or input.lvolume

        self.confine = environment.confine
        self.lconfine = self.confine != 0

        self.lexternals = len(self.input.externals.functions) > 0

        self.lelectrolyte = \
            bool(self.lelectrolyte or self.input.electrolyte.formula)

        self.static_permittivity = environment.static_permittivity
        self.lstatic = environment.static_permittivity > 1

        self.optical_permittivity = environment.optical_permittivity
        self.loptical = environment.optical_permittivity > 1

        functions = self.input.regions.functions

        if len(functions) > 0:

            stat = opt = None
            for group in functions:
                for function in group:
                    stat = function.static > 1
                    opt = function.optical > 1

            self.lstatic = self.lstatic or stat
            self.loptical = self.loptical or opt

        self.ldielectric = self.lstatic or self.loptical

    def _set_derived_flags(self) -> None:
        """docstring"""
        field_aware = self.input.solvent.field_aware
        solvent_mode = self.input.solvent.mode
        electrolyte_mode = self.input.electrolyte.mode

        self.lsolvent = \
            self.ldielectric or self.lsurface or self.lvolume or self.lconfine

        self.lelectrostatic = (self.ldielectric or self.lelectrolyte
                               or self.lexternals or self.lperiodic
                               or field_aware)  # and (not no_electrostatics)

        self.lsoftsolvent = (self.lsolvent
                             and (solvent_mode == 'electronic'
                                  or solvent_mode == 'full' or field_aware))

        self.lsoftelectrolyte = (self.lelectrolyte
                                 and (electrolyte_mode == 'electronic'
                                      or electrolyte_mode == 'full'
                                      or field_aware))

        self.lsoftcavity = self.lsoftsolvent or self.lsoftelectrolyte

        self.lrigidsolvent = \
            self.lsolvent and solvent_mode != 'electronic'

        self.lrigidelectrolyte = \
            self.lelectrolyte and electrolyte_mode != 'electronic'

        self.lrigidcavity = self.lrigidsolvent or self.lrigidelectrolyte

        self.lcoredensity = ((self.lsolvent and solvent_mode == 'full')
                             or (self.lelectrolyte
                                 and electrolyte_mode == 'full'))

        self.lsmearedions = self.lelectrostatic

        self.lboundary = self.lsolvent or self.lelectrolyte

        self.lgradient = self.ldielectric or field_aware or self.lsurface

    def _set_numerical_flags(self) -> None:
        """docstring"""
        self.lfft_system = self.lelectrostatic

        self.lfft_environment = ((self.lboundary or self.lelectrostatic)
                                 and self.input.solvent.deriv_core == 'fft')

        self.l1da = self.lperiodic and self.input.pbc.core == '1da'

        self.need_inner = self.input.electrostatics.inner_solver != 'none'
