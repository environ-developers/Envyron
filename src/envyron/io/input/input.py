from typing import (
    Any,
    Dict,
    Optional,
)

from pathlib import Path
from yaml import load, SafeLoader

from .base import (
    BaseModel,
    ControlModel,
    ElectrolyteModel,
    ElectrostaticsModel,
    EnvironmentModel,
    ExternalsContainerModel,
    IonsModel,
    PBCModel,
    RegionsContainerModel,
    SemiconductorModel,
    SolventModel,
    SystemModel,
)


class Input(BaseModel):
    """
    Model for Environ input.
    """
    control: Optional[ControlModel] = None
    environment: Optional[EnvironmentModel] = None
    ions: Optional[IonsModel] = None
    system: Optional[SystemModel] = None
    electrolyte: Optional[ElectrolyteModel] = None
    semiconductor: Optional[SemiconductorModel] = None
    solvent: Optional[SolventModel] = None
    electrostatics: Optional[ElectrostaticsModel] = None
    pbc: Optional[PBCModel] = None
    externals: Optional[ExternalsContainerModel] = None
    regions: Optional[RegionsContainerModel] = None

    def __init__(
        self,
        natoms: int,
        filename: Optional[str] = None,
        **params: Dict[str, Any],
    ) -> None:

        # default parameter dictionary
        param_dict: Dict[str, Dict[str, Any]] = {
            'control': {},
            'environment': {},
            'ions': {},
            'system': {},
            'electrolyte': {},
            'semiconductor': {},
            'solvent': {},
            'electrostatics': {},
            'pbc': {},
        }

        input_dict = {}

        if params:
            input_dict = params
        elif filename is not None:
            input_dict = self.read(filename)

        param_dict.update(input_dict)

        super().__init__(**param_dict)

        self.adjust_ionic_arrays(natoms)

        if input_dict:
            self.apply_smart_defaults()
            self.sanity_check()

    def read(self, filename: str) -> Dict[str, Any]:
        """Read parameter dictionary from a YAML input file.

        Parameters
        ----------
        filename : str
            The name of the YAML input file

        Returns
        -------
        Dict[str, Any]
            A parameter dictionary
        """
        try:
            with open(Path(filename).absolute()) as f:
                return load(f, SafeLoader)
        except Exception:
            raise

    def adjust_ionic_arrays(self, natoms: int) -> None:
        """Scale ionic arrays to size of number of atoms.

        Parameters
        ----------
        natoms : int
            The number of atoms in the simulation

        Raises
        ------
        ValueError
            If `natoms <= 0`
        """

        if natoms <= 0: raise ValueError("number of atoms must be positive")

        for array in (
                self.ions.atomicspread,
                self.ions.corespread,
                self.ions.solvationrad,
        ):

            if len(array) == 1 and natoms != 1: array *= natoms

            if len(array) != natoms:
                raise ValueError("array size not equal to number of atoms")

    def apply_smart_defaults(self) -> None:
        """Adjust input/default parameters based on user input."""
        self._adjust_environment()
        self._adjust_derivatives_method()
        self._adjust_electrostatics()

    def sanity_check(self) -> None:
        """Check for bad input values."""
        self._validate_solvent()
        self._validate_electrolyte()
        self._validate_electrostatics()

    def _adjust_environment(self) -> None:
        """Adjust environment properties according to environment type."""

        # set up vacuum environment
        if self.environment.type == 'vacuum':
            self.environment.static_permittivity = 1.0
            self.environment.optical_permittivity = 1.0
            self.environment.surface_tension = 0.0
            self.environment.pressure = 0.0

        # set up water environment
        elif 'water' in self.environment.type:
            self.environment.static_permittivity = 78.3
            self.environment.optical_permittivity = 1.776

            # non-ionic interfaces
            if self.solvent.mode in {
                    'electronic',
                    'full',
            }:
                if self.environment.type == 'water':
                    self.environment.surface_tension = 50.0
                    self.environment.pressure = -0.35
                    self.solvent.rhomax = 5e-3
                    self.solvent.rhomin = 1e-4

                elif self.environment.type == 'water-cation':
                    self.environment.surface_tension = 50.0
                    self.environment.pressure = -0.35
                    self.solvent.rhomax = 5e-3
                    self.solvent.rhomin = 1e-4

                elif self.environment.type == 'water-anion':
                    self.environment.surface_tension = 50.0
                    self.environment.pressure = -0.35
                    self.solvent.rhomax = 5e-3
                    self.solvent.rhomin = 1e-4

            # ionic interface
            if self.solvent.mode == 'ionic':
                self.environment.surface_tension = 50.0
                self.environment.pressure = -0.35
                self.solvent.softness = 0.5
                self.solvent.radius_mode = 'uff'

                if self.environment.type == 'water':
                    self.solvent.alpha = 1.12

                elif self.environment.type == 'water-cation':
                    self.solvent.alpha = 1.1

                elif self.environment.type == 'water-anion':
                    self.solvent.alpha = 0.98

    def _adjust_derivatives_method(self) -> None:
        """Adjust derivatives method according to solvent mode."""

        if self.solvent.deriv_method == 'default':

            # non-ionic interfaces
            if self.solvent.mode in {
                    'electronic',
                    'full',
                    'system',
            }:
                self.solvent.deriv_method = 'chain'

            # ionic interface
            elif self.solvent.mode == 'ionic':
                self.solvent.deriv_method = 'lowmem'

    def _adjust_electrostatics(self) -> None:
        """Adjust electrostatics according to solvent properties."""
        self._adjust_electrolyte_dependent_electrostatics()
        self._adjust_dielectric_dependent_electrostatics()

    def _adjust_electrolyte_dependent_electrostatics(self) -> None:
        """Adjust electrostatics according to electrolyte input."""

        if self.pbc.correction == 'gcs':

            if self.electrolyte.mode != 'system':
                self.electrolyte.mode = 'system'

            if self.electrolyte.formula is not None:

                # Linearized Poisson-Boltzmann problem
                if self.electrolyte.linearized:

                    if self.electrolyte.cionmax > 0.0 or \
                        self.electrolyte.rion > 0.0:
                        self.electrostatics.problem = 'linmodpb'

                    elif self.electrostatics.problem == 'none':
                        self.electrostatics.problem = 'linpb'

                    if self.electrostatics.solver == 'none':
                        self.electrostatics.solver = 'cg'

                else:  # Poisson-Boltzmann problem

                    if self.electrolyte.cionmax > 0.0 or \
                        self.electrolyte.rion > 0.0:
                        self.electrostatics.problem = 'modpb'

                    elif self.electrostatics.problem == 'none':
                        self.electrostatics.problem = 'pb'

                    if self.electrostatics.solver == 'none':
                        self.electrostatics.solver = 'newton'

                    if self.electrostatics.inner_solver == 'none':
                        self.electrostatics.inner_solver = 'cg'

        if self.pbc.correction == 'gcs' or \
            self.electrolyte.formula is not None:

            if self.electrolyte.deriv_method == 'default':

                # non-ionic interfaces
                if self.electrolyte.mode in {
                        'electronic',
                        'full',
                        'system',
                }:
                    self.electrolyte.deriv_method = 'chain'

                # ionic interface
                elif self.electrolyte.mode == 'ionic':
                    self.electrolyte.deriv_method = 'lowmem'

    def _adjust_dielectric_dependent_electrostatics(self) -> None:
        """Adjust electrostatics according to dielectric input."""

        if self.environment.static_permittivity > 1.0 or \
            self.regions is not None:

            if self.electrostatics.problem == 'none':
                self.electrostatics.problem = 'generalized'

            if self.pbc.correction != 'gcs':

                if self.electrostatics.solver == 'none':
                    self.electrostatics.solver = 'cg'

            elif self.electrostatics.solver != 'fixed-point':
                self.electrostatics.solver = 'fixed-point'

        else:

            if self.electrostatics.problem == 'none':
                self.electrostatics.problem = 'poisson'

            if self.electrostatics.solver == 'none':
                self.electrostatics.solver = 'direct'

        if self.electrostatics.solver == 'fixed-point' and \
            self.electrostatics.auxiliary == 'none':
            self.electrostatics.auxiliary = 'full'

        if self.electrostatics.inner_solver != 'none':

            if self.electrostatics.solver == 'fixed-point':

                if self.electrostatics.auxiliary == 'ioncc':
                    self.electrostatics.inner_problem = 'generalized'

            elif self.electrostatics.solver == 'newton':
                self.electrostatics.inner_problem = 'linpb'

    def _validate_solvent(self) -> None:
        """Check for bad solvent input."""

        # solvent distance
        if self.solvent.mode == 'system' and self.solvent.distance == 0.0:
            raise ValueError(
                "solvent distance must be greater than zero for system interfaces"
            )

        # rhomax/rhomin validation
        if self.solvent.rhomax < self.solvent.rhomin:
            raise ValueError("rhomax < rhomin")

        # non-ionic interfaces
        if self.solvent.mode in {
                'electronic',
                'full',
                'system',
        }:

            if 'mem' in self.solvent.deriv_method:
                raise ValueError(
                    "only 'fft' or 'chain' allowed with electronic interfaces")

        # ionic interface
        elif self.solvent.mode == 'ionic':

            if self.solvent.deriv_method == 'chain':
                raise ValueError(
                    "only 'highmem', 'lowmem', and 'fft' allowed with ionic interfaces"
                )

    def _validate_electrolyte(self):
        """Check for bad electrolyte input."""

        # electrolyte rhomax/rhomin validation
        if self.electrolyte.rhomax < self.electrolyte.rhomin:
            raise ValueError("electrolyte rhomax < electrolyte rhomin")

        # simultaneous cionmax/rion setting
        if self.electrolyte.cionmax > 0 and self.electrolyte.rion > 0:
            raise ValueError("cannot set cionmax and rion simultaneously")

        if self.pbc.correction == 'gcs':

            if self.electrolyte.distance == 0.0:
                raise ValueError(
                    "electrolyte distance must be greater than zero for GCS correction"
                )

        if self.pbc.correction == 'gcs' or \
            self.electrolyte.formula is not None:

            # non-ionic interfaces
            if self.electrolyte.mode in {
                    'electronic',
                    'full',
                    'system',
            }:

                if 'mem' in self.electrolyte.deriv_method:
                    raise ValueError(
                        "only 'fft' or 'chain' allowed with electronic interfaces"
                    )

            # ionic interface
            elif self.electrolyte.mode == 'ionic':

                if self.electrolyte.deriv_method == 'chain':
                    raise ValueError(
                        "only 'highmem', 'lowmem', and 'fft' allowed with ionic interfaces"
                    )

    def _validate_electrostatics(self) -> None:
        """Check for bad electrostatics input."""

        # pbc dim validation
        if self.pbc.dim == 1:
            raise ValueError("1D periodic boundary correction not implemented")

        for problem, solver in zip(
            (self.electrostatics.problem, self.electrostatics.inner_problem),
            (self.electrostatics.solver, self.electrostatics.inner_solver),
        ):

            if problem == 'generalized':
                if solver == 'direct':
                    raise ValueError(
                        "cannot use direct solver for the Generalized Poisson eq."
                    )

            elif "pb" in problem:

                if "lin" in problem:
                    if solver not in {
                            'none',
                            'cg',
                            'sd',
                    }:
                        raise ValueError(
                            "only gradient-based solvers allowed for the linearized Poisson-Boltzmann eq."
                        )

                    if self.pbc.correction != 'parabolic':
                        raise ValueError(
                            "linearized-PB problem requires parabolic PBC correction"
                        )

                else:

                    if solver in {
                            'direct',
                            'cg',
                            'sd',
                    }:
                        raise ValueError(
                            "no direct or gradient-based solver allowed for the full Poisson-Boltzmann eq."
                        )

        if self.electrostatics.inner_solver != 'none' and \
            problem not in {
                'pb',
                'modpb',
                'generalized',
            }:
            raise ValueError("only pb or modpb problems allow inner solver")
