from envyron.utils.constants import BOHR_RADIUS, RYDBERG

from envyron.io.input.input import Input
from envyron.domains import EnvironGrid
from envyron.cores import FFTCore, Analytic1DCore, CoreContainer
from envyron.solvers import DirectSolver, GradientSolver, FixedPointSolver, \
    NewtonSolver, IterativeSolver, ElectrostaticSolverSetup


class Setup:
    """
    Simulation details of an Environ calculation, including the
    simulation parameters, simulation cell, numerical
    cores, and electrostatic solvers.
    """

    def __init__(self, user_input: Input) -> None:
        self.input = user_input
        self.niter_scf = 0
        self.niter_ionic = 0

        self.cell = None

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

        self.lelectrolyte = False
        self.lsemiconductor = False
        self.laddcharges = False
        self.ltddfpt = False
        self.lmsgcs = False

        if self.lperiodic:
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

        self.lexternals = (self.input.externals is not None) and \
            (len(self.input.externals.functions) > 0)

        self.lelectrolyte = \
            bool(self.lelectrolyte or self.input.electrolyte.concentration > 0.)

        self.static_permittivity = environment.static_permittivity
        self.lstatic = environment.static_permittivity > 1

        self.optical_permittivity = environment.optical_permittivity
        self.loptical = environment.optical_permittivity > 1

        self.lregions = (self.input.regions is not None) and \
            (len(self.input.regions.functions) > 0)

        if self.lregions:
            stat = opt = None
            for group in self.input.regions.functions:
                for function in group:
                    stat = stat or function.static > 1
                    opt = opt or function.optical > 1
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
                             and (solvent_mode in ('electronic', 'full')
                                  or field_aware))

        self.lsoftelectrolyte = (self.lelectrolyte and
                                 (electrolyte_mode in ('electronic', 'full')
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
        self.lfft = self.lelectrostatic or \
            (self.lboundary and self.input.solvent.deriv_core == 'fft')

        self.l1da = self.lperiodic and self.input.pbc.core == '1da'

        self.need_inner = self.input.electrostatics.inner_solver != 'none'

    def init_cell(self, cell: EnvironGrid):
        """docstring"""
        self.cell = cell

    def init_numerical(self, use_internal_pbc_corr):
        """docstring"""
        if self.lfft: self.fft = FFTCore(self.cell)
        if self.l1da:
            self.analytic1d = Analytic1DCore(self.cell, self.input.pbc.dim,
                                             self.input.pbc.axis)
        self._set_core_containers(use_internal_pbc_corr)
        if self.lelectrostatic: self._set_electrostatics()
        self.has_numerical_setup = True

    def _set_core_containers(
        self,
        use_internal_pbc_corr: bool,
    ):
        """docstring"""
        self.environment_core = CoreContainer('environment')
        # Derivative core
        if self.lboundary:
            if self.input.solvent.deriv_core == 'fft':
                self.environment_core.derivatives(self.fft)
            else:
                raise ValueError('Unexpected derivative core')
        # Electrostatic cores
        if self.lelectrostatic:
            self.reference_core = CoreContainer('reference',
                                                use_internal_pbc_corr,
                                                electrostatics_core=self.fft)
            if self.input.electrostatics.core == 'fft':
                self.environment_core.electrostatics(self.fft)
            else:
                raise ValueError('Unexpected electrostatic core')
        # Correction cores
        if self.lperiodic:
            if self.input.pbc.core == '1da':
                self.environment_core.corrections(self.analytic1d)

    def _set_electrostatics(self):
        """docstring"""
        # Reference solver
        self.reference_direct = DirectSolver(self.reference_core)
        # Outer solver
        self.direct = DirectSolver(self.environment_core,
                                   self.input.pbc.correction)
        if self.input.electrostatics.solver == 'direct':
            local_outer_solver = self.direct
        elif self.input.electrostatics.solver in ('cg', 'sd'):
            if self.input.electrostatics.solver == 'cg': self.lconjugate = True
            self.gradient = GradientSolver(
                self.environment_core, self.direct,
                self.input.electrostatics.preconditioner, self.lconjugate,
                self.input.electrostatics.maxstep,
                self.input.electrostatics.tol,
                self.input.electrostatics.auxiliary)
            local_outer_solver = self.gradient
        elif self.input.electrostatics.solver == 'fixed-point':
            self.fixedpoint = FixedPointSolver(
                self.environment_core, self.direct,
                self.input.electrostatics.maxstep,
                self.input.electrostatics.tol,
                self.input.electrostatics.auxiliary,
                self.input.electrostatics.mix)
            local_outer_solver = self.fixedpoint
        elif self.input.electrostatics.solver == 'newton':
            self.newton = NewtonSolver(self.environment_core, self.direct,
                                       self.input.electrostatics.maxstep,
                                       self.input.electrostatics.tol,
                                       self.input.electrostatics.auxiliary)
            local_outer_solver = self.newton
        else:
            raise ValueError('Unexpected outer solver')

        if self.need_inner:
            lconjugate = False
            if self.input.electrostatics.solver == 'fixed-point':
                if self.input.electrostatics.auxiliary == 'ioncc':
                    if self.input.electrostatics.inner_solver in ('cg', 'sd'):
                        self.inner_gradient = GradientSolver(
                            self.environment_core, self.direct,
                            self.input.electrostatics.preconditioner,
                            lconjugate,
                            self.input.electrostatics.inner_maxstep,
                            self.input.electrostatics.inner_tol,
                            self.input.electrostatics.auxiliary)
                        local_inner_solver = self.inner_gradient
                    elif self.input.electrostatics.inner_solver == 'fixed-point':
                        local_auxiliary = 'full'
                        self.inner_fixedpoint = FixedPointSolver(
                            self.environment_core, self.direct,
                            self.input.electrostatics.inner_maxstep,
                            self.input.electrostatics.inner_tol,
                            local_auxiliary,
                            self.input.electrostatics.inner_mix)
                        local_inner_solver = self.inner_fixedpoint
                else:
                    raise ValueError(
                        'Unexpected value for auxiliary in nested solver')
            elif self.input.electrostatics.solver == 'newton':
                lconjugate = True
                self.inner_gradient = GradientSolver(
                    self.environment_core, self.direct,
                    self.input.electrostatics.preconditioner, lconjugate,
                    self.input.electrostatics.inner_maxstep,
                    self.input.electrostatics.inner_tol,
                    self.input.electrostatics.auxiliary)
                local_inner_solver = self.inner_gradient
            else:
                raise ValueError('Unexpected inner solver')

        # Reference electrosatic setup
        local_problem = 'poisson'
        self.reference = ElectrostaticSolverSetup(local_problem,
                                                  self.reference_direct)
        # Outer electrostatic setup
        self.outer = ElectrostaticSolverSetup(
            self.input.electrostatics.problem, local_outer_solver)

        # Inner electrostatic setup
        if self.need_inner:
            self.inner = ElectrostaticSolverSetup(self.inner.problem,
                                                  local_inner_solver)
            self.outer.inner = self.inner

        self._set_electrostatic_flags(self.reference)
        self._set_electrostatic_flags(self.outer)
        if self.need_inner: self._set_electrostatic_flags(self.inner)

    def _set_electrostatic_flags(self, setup: ElectrostaticSolverSetup):
        """docstring"""
        if setup.problem in ('none', 'poisson'):
            pass
        elif setup.problem in ('generalized', 'linpb', 'linmodpb', 'pb',
                               'modpb'):
            if type(setup.solver) == GradientSolver:
                if self.input.electrostatics.preconditioner == 'sqrt':
                    self.need_factsqrt = True
                elif self.input.electrostatics.preconditioner in ('left',
                                                                  'none'):
                    self.need_gradient = True
                else:
                    raise ValueError('Unexpected preconditioner')
            if type(setup.solver) == IterativeSolver:
                if setup.solver.auxiliary != 'none':
                    self.need_auxiliary = True
