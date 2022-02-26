from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypeAlias
from pydantic import BaseModel, validator
from pydantic.fields import ModelField

IntFloat: TypeAlias = Union[int, float]


class CardModel(BaseModel):
    """
    Model for card input.
    """
    position: List[float] = [0.0, 0.0, 0.0]
    spread = 0.5
    dim = 0
    axis = 3

    @validator('dim')
    def valid_dimension(cls, value: int) -> int:
        """Check that dimensions are reasonable (0<=>3)."""
        assert 0 <= value <= 3, f"expected 0 <= dim <= 3, got {value}"
        return value

    @validator('axis')
    def valid_axis(cls, value: int) -> int:
        """Check that axis is reasonable (1<=>3)."""
        assert 1 <= value <= 3, f"axis out of range (1 <= a <= 3)"
        return value


class ExternalModel(CardModel):
    """
    Model for a single external function.
    """
    charge: float

    @validator('charge')
    def ne_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is not zero."""
        assert value != 0, f"{value} must not be zero"
        return value

    @validator('spread')
    def ge_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, f"{value} must be greater than or equal to zero"
        return value


class RegionModel(CardModel):
    """
    Model for a single region function.
    """
    static = 1.0
    optical = 1.0
    width = 0.0

    @validator(
        'spread',
        'width',
    )
    def ge_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, f"{value} must be greater than or equal to zero"
        return value

    @validator(
        'static',
        'optical',
    )
    def ge_one(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to one."""
        assert value >= 1, f"{value} must be greater than or equal to one"
        return value


class CardContainer(BaseModel):
    """
    Container for card functions.
    """
    units = 'bohr'

    @validator('units')
    def valid_units(cls, value: str) -> str:
        """Check value against acceptable units."""
        acceptable = (
            'bohr',
            'angstrom',
        )
        assert any(value == v for v in acceptable), \
            f"{value} not a valid unit ( bohr | angstrom )"
        return value


class ExternalsContainer(CardContainer):
    """
    Container for external functions.
    """
    functions: List[List[ExternalModel]] = []


class RegionsContainer(CardContainer):
    """
    Container for region functions.
    """
    functions: List[List[RegionModel]] = []


class EnvironInputModel(BaseModel):
    """
    Model for Environ input.
    """

    natoms = 1

    # Environ section

    debug = False
    restart = False
    verbosity = 0
    threshold = 0.1
    nskip = 1
    env_type = 'input'
    nrep: List[int] = [0, 0, 0]
    system_ntyp = 0
    system_dim = 0
    system_axis = 3
    system_pos: List[float] = [0.0, 0.0, 0.0]
    need_electrostatic = False
    atomicspread: List[float] = [0.5]
    corespread: List[float] = [0.5]
    static_permittivity = 1.0
    optical_permittivity = 1.0
    surface_tension = 0.0
    pressure = 0.0
    confine = 0.0
    electrolyte_concentration = 0.0
    electrolyte_formula: Optional[List[int]] = None
    electrolyte_linearized = False
    electrolyte_entropy = 'full'
    cionmax = 0.0
    rion = 0.0
    temperature = 300.0
    sc_permittivity = 1.0
    sc_carrier_density = 0.0
    external_charges = 0
    dielectric_regions = 0

    # Boundary section

    solvent_mode = 'electronic'
    radius_mode = 'uff'
    alpha = 1.0
    softness = 0.5
    solvationrad: List[float] = [0.0]
    stype = 2
    rhomax = 5e-3
    rhomin = 1e-4
    tbeta = 4.8
    solvent_distance = 1.0
    solvent_spread = 0.5
    solvent_radius = 0.0
    radial_scale = 2.0
    radial_spread = 0.5
    filling_threshold = 0.825
    filling_spread = 0.02
    sc_distance = 0.0
    sc_spread = 0.5
    electrolyte_mode = 'electronic'
    electrolyte_distance = 0.0
    electrolyte_spread = 0.5
    electrolyte_rhomax = 5e-3
    electrolyte_rhomin = 1e-4
    electrolyte_tbeta = 4.8
    electrolyte_alpha = 1.0
    electrolyte_softness = 0.5
    electrolyte_deriv_method = 'default'
    deriv_method = 'default'
    deriv_core = 'fft'

    # Electrostatic section

    problem = 'none'
    tol = 1e-5
    solver = 'none'
    auxiliary = 'none'
    step_type = 'optimal'
    step = 0.3
    maxstep = 200
    mix_type = 'linear'
    ndiis = 1
    mix = 0.5
    preconditioner = 'sqrt'
    screening_type = 'none'
    screening = 0.0
    core = 'fft'
    inner_solver = 'none'
    inner_core = 'fft'
    inner_tol = 1e-10
    inner_maxstep = 200
    inner_mix = 0.5
    pbc_correction = 'none'
    pbc_core = '1da'
    pbc_dim = 0
    pbc_axis = 3

    # Card sections

    externals: Optional[ExternalsContainer] = None
    regions: Optional[RegionsContainer] = None

    def __init__(self, **data: Dict[str, Any]) -> None:
        super().__init__(**data)
        self._adjust_input()
        self._final_validation()

    @classmethod
    def set_number_of_atoms(cls, natoms: int) -> None:
        """Set the number of atoms in the current calculation."""
        cls.natoms = natoms

    # Environ section

    @validator(
        'natoms',
        pre=True,
    )
    def _check_number_of_atoms(cls, value: int) -> int:
        """Verify the number of atoms is non-zero."""
        assert value != 0, f"number of atoms must not be zero"
        return value

    @validator(
        'nrep',
        'system_pos',
        'atomicspread',
        'corespread',
        'electrolyte_formula',
        'solvationrad',
        pre=True,
    )
    def _split_string(cls, value: str) -> List[str]:
        """Preprocess string into a list of values."""
        if ' ' in value: return value.split()
        return [value]

    @validator(
        'nrep',
        'system_pos',
        pre=True,
    )
    def _adjust_vector_size(
        cls,
        value: List[str],
        field: ModelField,
    ) -> List[str]:
        """Scale vector input to 3D."""
        if len(value) == 1: value = value * 3
        assert len(value) == 3, f"{field.name} array size should be 3"
        return value

    @validator(
        'atomicspread',
        'corespread',
        'solvationrad',
        pre=True,
    )
    def _adjust_to_natoms(
        cls,
        value: List[str],
        field: ModelField,
    ) -> List[str]:
        """Scale ion input arrays to size of number of atoms."""
        if len(value) == 1 and cls.natoms != 1: value = value * cls.natoms
        assert len(value) == cls.natoms, \
            f"{field.name} array size not equal to number of atoms"
        return value

    @validator(
        'alpha',
        'softness',
        'solvent_distance',
        'solvent_spread',
        'radial_spread',
        'filling_threshold',
        'filling_spread',
        'sc_spread',
        'electrolyte_spread',
        'electrolyte_softness',
        'tol',
        'step',
        'ndiis',
        'inner_tol',
        'inner_mix',
    )
    def _gt_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than zero."""
        assert value > 0, f"{value} must be greater than zero"
        return value

    @validator(
        'verbosity',
        'threshold',
        'nskip',
        'system_ntyp',
        'surface_tension',
        'confine',
        'electrolyte_concentration',
        'cionmax',
        'rion',
        'temperature',
        'sc_carrier_density',
        'external_charges',
        'dielectric_regions',
        'rhomax',
        'rhomin',
        'tbeta',
        'solvent_radius',
        'sc_distance',
        'electrolyte_distance',
        'electrolyte_rhomax',
        'electrolyte_rhomin',
        'electrolyte_tbeta',
        'electrolyte_alpha',
        'screening',
    )
    def _ge_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, f"{value} must be greater than or equal to zero"
        return value

    @validator(
        'nrep',
        'atomicspread',
        'corespread',
        'solvationrad',
        each_item=True,
    )
    def _ge_zero_many(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, f"{value} must be greater than or equal to zero"
        return value

    @validator(
        'maxstep',
        'inner_maxstep',
    )
    def _gt_one(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than one."""
        assert value > 1, f"{value} must be greater than one"
        return value

    @validator(
        'static_permittivity',
        'optical_permittivity',
        'radial_scale',
    )
    def _ge_one(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to one."""
        assert value >= 1, f"{value} must be greater than or equal to one"
        return value

    @validator('stype')
    def _valid_switching_function_type(cls, value: int) -> int:
        """Check that switching function type is valid."""
        assert 0 <= value <= 2, f"expected 0 <= stype <= 2, got {value}"
        return value

    @validator(
        'system_dim',
        'pbc_dim',
    )
    def _valid_dimension(cls, value: int) -> int:
        """Check that the dimensions are reasonable (0<=>3)."""
        assert 0 <= value <= 3, f"expected 0 <= dim <= 3, got {value}"
        return value

    @validator(
        'system_axis',
        'pbc_axis',
    )
    def _valid_axis(cls, value: int) -> int:
        """Check that the axis is reasonable (1<=>3)."""
        assert 1 <= value <= 3, f"axis out of range (1 <= a <= 3)"
        return value

    @validator('env_type')
    def _valid_environment_type(cls, value: str) -> str:
        """Check value against acceptable environment types."""
        acceptable = (
            'input',
            'vacuum',
            'water',
            'water-cation',
            'water-anion',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected environment type"
        return value

    @validator('electrolyte_entropy')
    def _valid_electrolyte_entropy(cls, value: str) -> str:
        """Check value against acceptable electrolyte entropy schemes."""
        acceptable = (
            'ions',
            'full',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected electrolyte entropy scheme"
        return value

    # Boundary section

    @validator(
        'solvent_mode',
        'electrolyte_mode',
    )
    def _valid_solvent_mode(cls, value: str) -> str:
        """Check value against acceptable solvent/electrolyte modes."""
        acceptable = (
            'electronic',
            'ionic',
            'full',
            'system',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected solvent/electrolyte mode"
        return value

    @validator(
        'deriv_method',
        'electrolyte_deriv_method',
    )
    def _valid_deriv_method(cls, value: str) -> str:
        """Check value against acceptable derivative methods."""
        acceptable = (
            'default',
            'fft',
            'chain',
            'highmen',
            'lowmem',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected derivative method"
        return value

    @validator('radius_mode')
    def _valid_radius_mode(cls, value: str) -> str:
        """Check value against acceptable radius modes."""
        acceptable = (
            'pauling',
            'bondi',
            'uff',
            'muff',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected radius mode"
        return value

    @validator('deriv_core')
    def _valid_derivatives_core(cls, value: str) -> str:
        """Check value against acceptable derivatives cores."""
        acceptable = ('fft', )
        assert any(value == v for v in acceptable), \
            f"unexpected derivatives core"
        return value

    # Electrostatic section

    @validator('problem')
    def _valid_electrostatic_problem(cls, value: str) -> str:
        """Check value against acceptable electrostatic problems."""
        acceptable = (
            'none',
            'poisson',
            'generalized',
            'pb',
            'modpb',
            'linpb',
            'linmodpb',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected electrostatic problem"
        return value

    @validator('solver')
    def _valid_electrostatic_solver(cls, value: str) -> str:
        """Check value against acceptable electrostatic solvers."""
        acceptable = (
            'none',
            'cg',
            'sd',
            'fixed-point',
            'newton',
            'nested',
            'direct',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected electrostatic solver"
        return value

    @validator('auxiliary')
    def _valid_auxiliary_scheme(cls, value: str) -> str:
        """Check value against acceptable auxiliary schemes."""
        acceptable = (
            'none',
            'full',
            'ioncc',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected auxiliary scheme"
        return value

    @validator('step_type')
    def _valid_step_type(cls, value: str) -> str:
        """Check value against acceptable step types."""
        acceptable = (
            'optimal',
            'input',
            'random',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected step type"
        return value

    @validator('mix_type')
    def _valid_mix_type(cls, value: str) -> str:
        """Check value against acceptable mix types."""
        acceptable = (
            'linear',
            'anderson',
            'diis',
            'broyden',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected mix type"
        return value

    @validator('preconditioner')
    def _valid_preconditioner(cls, value: str) -> str:
        """Check value against acceptable preconditioners."""
        acceptable = (
            'sqrt',
            'left',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected preconditioner"
        return value

    @validator('screening_type')
    def _valid_screening_type(cls, value: str) -> str:
        """Check value against acceptable screening types."""
        acceptable = (
            'none',
            'input',
            'linear',
            'optimal',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected screening type"
        return value

    @validator(
        'core',
        'inner_core',
    )
    def _valid_electrostatic_core(cls, value: str) -> str:
        """Check value against acceptable electrostatic cores."""
        acceptable = ('fft', )
        assert any(value == v for v in acceptable), \
            f"unexpected electrostatics core"
        return value

    @validator('inner_solver')
    def _valid_electrostatic_inner_solver(cls, value: str) -> str:
        """Check value against acceptable electrostatic inner solvers."""
        acceptable = (
            'none',
            'cg',
            'sd',
            'fixed-point',
            'newton',
            'direct',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected electrostatic inner solver"
        return value

    @validator('pbc_correction')
    def _valid_pbc_correction(cls, value: str) -> str:
        """Check value against acceptable pbc corrections."""
        acceptable = (
            'none',
            'parabolic',
            'gcs',
            'ms',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected pbc correction"
        return value

    @validator('pbc_core')
    def _valid_pbc_core(cls, value: str) -> str:
        """Check value against acceptable pbc correction cores."""
        acceptable = ('1da', )
        assert any(value == v for v in acceptable), \
            f"unexpected pbc core"
        return value

    def _adjust_input(self) -> None:
        """Adjust input/default parameters based on user input."""
        self._adjust_environment()
        self._adjust_derivatives_method()
        self._adjust_electrostatics()

    def _adjust_environment(self) -> None:
        """Adjust environment properties according to environment type."""

        # set up vacuum environment
        if self.env_type == 'vacuum':
            self.static_permittivity = 1.0
            self.optical_permittivity = 1.0
            self.surface_tension = 0.0
            self.pressure = 0.0

        # set up water environment
        elif 'water' in self.env_type:
            self.static_permittivity = 78.3
            self.optical_permittivity = 1.776

            # non-ionic interfaces
            if self.solvent_mode in {'electronic', 'full'}:
                if self.env_type == 'water':
                    self.surface_tension = 50.0
                    self.pressure = -0.35
                    self.rhomax = 5e-3
                    self.rhomin = 1e-4

                elif self.env_type == 'water-cation':
                    self.surface_tension = 50.0
                    self.pressure = -0.35
                    self.rhomax = 5e-3
                    self.rhomin = 1e-4

                elif self.env_type == 'water-anion':
                    self.surface_tension = 50.0
                    self.pressure = -0.35
                    self.rhomax = 5e-3
                    self.rhomin = 1e-4

            # ionic interface
            if self.solvent_mode == 'ionic':
                self.surface_tension = 50.0
                self.pressure = -0.35
                self.softness = 0.5
                self.radius_mode = 'uff'

                if self.env_type == 'water':
                    self.alpha = 1.12

                elif self.env_type == 'water-cation':
                    self.alpha = 1.1

                elif self.env_type == 'water-anion':
                    self.alpha = 0.98

    def _adjust_derivatives_method(self) -> None:
        """Adjust derivatives method according to solvent mode."""

        if self.deriv_method == 'default':

            # non-ionic interfaces
            if self.solvent_mode in {'electronic', 'full', 'system'}:
                self.deriv_method = 'chain'

            # ionic interface
            elif self.solvent_mode == 'ionic':
                self.deriv_method = 'lowmem'

    def _adjust_electrostatics(self) -> None:
        """Adjust electrostatics according to solvent properties."""
        self._check_electrolyte_input()
        self._check_dielectric_input()

    def _check_electrolyte_input(self) -> None:
        """Adjust electrostatics according to electrolyte input."""

        if self.pbc_correction == 'gcs':

            if self.electrolyte_mode != 'system':
                self.electrolyte_mode = 'system'

            if self.electrolyte_formula is not None:

                # Linearized Poisson-Boltzmann problem
                if self.electrolyte_linearized:
                    if self.cionmax > 0.0 or self.rion > 0.0:
                        self.problem = 'linmodpb'
                    elif self.problem == 'none':
                        self.problem = 'linpb'

                    if self.solver == 'none':
                        self.solver = 'cg'

                else:  # Poisson-Boltzmann problem
                    if self.cionmax > 0.0 or self.rion > 0.0:
                        self.problem = 'modpb'
                    elif self.problem == 'none':
                        self.problem = 'pb'

                    if self.solver == 'none':
                        self.solver = 'newton'

        if self.pbc_correction == 'gcs' or \
            self.electrolyte_formula is not None:

            if self.electrolyte_deriv_method == 'default':

                # non-ionic interfaces
                if self.electrolyte_mode in {'electronic', 'full', 'system'}:
                    self.electrolyte_deriv_method = 'chain'

                # ionic interface
                elif self.electrolyte_mode == 'ionic':
                    self.electrolyte_deriv_method = 'lowmem'

    def _check_dielectric_input(self) -> None:
        """Adjust electrostatics according to dielectric input."""

        if self.static_permittivity > 1.0 or \
            self.dielectric_regions > 0:

            if self.problem == 'none':
                self.problem = 'generalized'

            if self.pbc_correction != 'gcs':

                if self.solver == 'none':
                    self.solver = 'cg'

            elif self.solver != 'fixed-point':
                self.solver = 'fixed-point'

        else:

            if self.problem == 'none':
                self.problem = 'poisson'

            if self.solver == 'none':
                self.solver = 'direct'

        if self.solver == 'fixed-point' and \
            self.auxiliary == 'none':

            self.auxiliary = 'full'

    def _final_validation(self) -> None:
        """Check for bad input values."""
        self._validate_derivatives_method()
        self._validate_electrostatics()

    def _validate_derivatives_method(self) -> None:
        """Check for bad derivatives method."""

        # non-ionic interfaces
        if self.solvent_mode in {'electronic', 'full', 'system'}:
            if 'mem' in self.deriv_method:
                raise ValueError(
                    "Only 'fft' or 'chain' are allowed with electronic interfaces"
                )

        # ionic interface
        elif self.solvent_mode == 'ionic':
            if self.deriv_method == 'chain':
                raise ValueError(
                    "Only 'highmem', 'lowmem', and 'fft' are allowed with ionic interfaces"
                )

    def _validate_electrostatics(self) -> None:
        """Check for bad electrostatics input."""

        # rhomax/rhomin validation
        if self.rhomax < self.rhomin:
            raise ValueError("rhomax < rhomin")

        # electrolyte rhomax/rhomin validation
        if self.electrolyte_rhomax < self.electrolyte_rhomin:
            raise ValueError("electrolyte_rhomax < electrolyte_rhomin")

        # pbc_dim validation
        if self.pbc_dim == 1:
            raise ValueError("1D periodic boundary correction not implemented")

        # electrolyte validation

        if self.pbc_correction == 'gcs':

            if self.electrolyte_distance == 0.0:
                raise ValueError(
                    "electrolyte_distance must be greater than zero for gcs correction"
                )

        if self.pbc_correction == 'gcs' or \
            self.electrolyte_formula is not None:

            # non-ionic interfaces
            if self.electrolyte_mode in {'electronic', 'full', 'system'}:
                if 'mem' in self.electrolyte_deriv_method:
                    raise ValueError(
                        "Only 'fft' or 'chain' are allowed with electronic interfaces"
                    )

            # ionic interface
            elif self.electrolyte_mode == 'ionic':
                if self.electrolyte_deriv_method == 'chain':
                    raise ValueError(
                        "Only 'highmem', 'lowmem', and 'fft' are allowed with ionic interfaces"
                    )

        # problem/solver validation

        if self.problem == 'generalized':

            if self.solver == 'direct' or \
                self.inner_solver == 'direct':
                raise ValueError(
                    "Cannot use a direct solver for the Generalized Poisson eq."
                )

        elif "pb" in self.problem:

            if "lin" in self.problem:
                solvers = {'none', 'cg', 'sd'}

                if self.solver not in solvers or \
                    self.inner_solver not in solvers:
                    raise ValueError(
                        "Only gradient-based solver for the linearized Poisson-Boltzmann eq."
                    )

                if self.pbc_correction != 'parabolic':
                    raise ValueError(
                        "Linearized-PB problem requires parabolic pbc correction"
                    )

            else:
                solvers = {'direct', 'cg', 'sd'}

                if self.solver in solvers or \
                    self.inner_solver in solvers:
                    raise ValueError(
                        "No direct or gradient-based solver for the full Poisson-Boltzmann eq."
                    )

        problems = {'pb, modpb, generalized'}

        if self.inner_solver != 'none' and \
            self.problem not in problems:
            raise ValueError("Only pb or modpb problems allow inner solver")
