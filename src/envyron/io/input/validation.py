from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypeAlias
from pydantic import BaseModel, validator
from pydantic_yaml import YamlModelMixin

IntFloat: TypeAlias = Union[int, float]


class CardModel(YamlModelMixin, BaseModel):
    """
    Model for card input.
    """
    pos: List[float] = [0.0, 0.0, 0.0]
    spread = 0.5
    dim = 0
    axis = 3

    @validator('dim')
    def valid_dimension(cls, value: int) -> int:
        """Check that dimensions are reasonable (0<=>3)."""
        assert 0 <= value <= 3, "dimensions out of range (0<=dim<=3)"
        return value

    @validator('axis')
    def valid_axis(cls, value: int) -> int:
        """Check that axis is reasonable (1<=>3)."""
        assert 1 <= value <= 3, "axis out of range (1<=axis<=3)"
        return value


class ExternalModel(CardModel):
    """
    Model for a single external function.
    """
    charge: float

    @validator('charge')
    def ne_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is not zero."""
        assert value != 0, "must be non-zero"
        return value

    @validator('spread')
    def ge_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, "must be greater than or equal to zero"
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
        assert value >= 0, "must be greater than or equal to zero"
        return value

    @validator(
        'static',
        'optical',
    )
    def ge_one(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to one."""
        assert value >= 1, "must be greater than or equal to one"
        return value


class CardContainer(YamlModelMixin, BaseModel):
    """
    Container for card functions.
    """
    units = 'bohr'

    @validator('units')
    def valid_units(cls, value: str) -> str:
        """Check value against acceptable units."""
        valid = (
            'bohr',
            'angstrom',
        )
        assert any(value == v for v in valid), "units must be bohr or angstrom"
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


class ControlModel(YamlModelMixin, BaseModel):
    """
    Model for control parameters.
    """
    debug = False
    restart = False
    verbosity = 0
    threshold = 0.1
    nskip = 1
    nrep: List[int] = [0, 0, 0]
    need_electrostatic = False

    @validator(
        'nrep',
        pre=True,
    )
    def _vectorize(cls, value: IntFloat) -> List[IntFloat]:
        """Cast value as list."""
        if not isinstance(value, list): value = [value]
        return value

    @validator(
        'nrep',
        pre=True,
    )
    def _adjust_vector_size(cls, value: List[IntFloat]) -> List[IntFloat]:
        """Scale vector input to 3D."""
        if len(value) == 1: value = value * 3
        assert len(value) == 3, "array size should be 3"
        return value

    @validator(
        'verbosity',
        'threshold',
        'nskip',
    )
    def _ge_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, "must be greater than or equal to zero"
        return value

    @validator(
        'nrep',
        each_item=True,
    )
    def _ge_zero_many(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, "must be greater than or equal to zero"
        return value


class EnvironmentModel(YamlModelMixin, BaseModel):
    """
    Model for environment parameters.
    """
    type = 'input'
    surface_tension = 0.0
    pressure = 0.0
    confine = 0.0
    static_permittivity = 1.0
    optical_permittivity = 1.0
    temperature = 300.0

    @validator(
        'surface_tension',
        'confine',
        'temperature',
    )
    def _ge_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, "must be greater than or equal to zero"
        return value

    @validator(
        'static_permittivity',
        'optical_permittivity',
    )
    def _ge_one(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to one."""
        assert value >= 1, "must be greater than or equal to one"
        return value

    @validator('type')
    def _valid_environment_type(cls, value: str) -> str:
        """Check value against acceptable environment types."""
        valid = (
            'input',
            'vacuum',
            'water',
            'water-cation',
            'water-anion',
        )
        assert any(value == v for v in valid), "unexpected environment type"
        return value


class IonsModel(YamlModelMixin, BaseModel):
    """
    Model for ions parameters.
    """
    atomicspread: List[float] = [0.5]
    corespread: List[float] = [0.5]
    solvationrad: List[float] = [0.0]

    @validator(
        'atomicspread',
        'corespread',
        'solvationrad',
        pre=True,
    )
    def _vectorize(cls, value: IntFloat) -> List[IntFloat]:
        """Cast value as list."""
        if not isinstance(value, list): value = [value]
        return value

    @validator(
        'atomicspread',
        'corespread',
        'solvationrad',
        each_item=True,
    )
    def _ge_zero_many(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, "must be greater than or equal to zero"
        return value


class SystemModel(YamlModelMixin, BaseModel):
    """
    Model for system parameters.
    """
    ntyp = 0
    dim = 0
    axis = 3
    pos: List[float] = [0.0, 0.0, 0.0]

    @validator(
        'pos',
        pre=True,
    )
    def _vectorize(cls, value: IntFloat) -> List[IntFloat]:
        """Cast value as list."""
        if not isinstance(value, list): value = [value]
        return value

    @validator(
        'pos',
        pre=True,
    )
    def _adjust_vector_size(cls, value: List[IntFloat]) -> List[IntFloat]:
        """Scale vector input to 3D."""
        if len(value) == 1: value = value * 3
        assert len(value) == 3, "array size should be 3"
        return value

    @validator('ntyp')
    def _ge_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, "must be greater than or equal to zero"
        return value

    @validator('dim')
    def _valid_dimension(cls, value: int) -> int:
        """Check that the dimensions are reasonable (0<=>3)."""
        assert 0 <= value <= 3, "dimensions out of range (0<=dim<=3)"
        return value

    @validator('axis')
    def _valid_axis(cls, value: int) -> int:
        """Check that the axis is reasonable (1<=>3)."""
        assert 1 <= value <= 3, "axis out of range (1<=axis<=3)"
        return value


class ElectrolyteModel(YamlModelMixin, BaseModel):
    """
    Model for electrolyte parameters.
    """
    linearized = False
    mode = 'electronic'
    entropy = 'full'
    deriv_method = 'default'
    concentration = 0.0
    formula: Optional[List[int]] = None
    cionmax = 0.0
    rion = 0.0
    distance = 0.0
    spread = 0.5
    rhomax = 5e-3
    rhomin = 1e-4
    tbeta = 4.8
    alpha = 1.0
    softness = 0.5

    @validator(
        'formula',
        pre=True,
    )
    def _vectorize(cls, value: IntFloat) -> List[IntFloat]:
        """Cast value as list."""
        if not isinstance(value, list): value = [value]
        return value

    @validator(
        'spread',
        'softness',
    )
    def _gt_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than zero."""
        assert value > 0, "must be greater than zero"
        return value

    @validator(
        'concentration',
        'cionmax',
        'rion',
        'distance',
        'rhomax',
        'rhomin',
        'tbeta',
        'alpha',
    )
    def _ge_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, "must be greater than or equal to zero"
        return value

    @validator('mode')
    def _valid_mode(cls, value: IntFloat) -> IntFloat:
        """Check value against acceptable solvent/electrolyte modes."""
        valid = (
            'electronic',
            'ionic',
            'full',
            'system',
        )
        assert any(value == v for v in valid), "unexpected electrolyte mode"
        return value

    @validator('deriv_method')
    def _valid_deriv_method(cls, value: str) -> str:
        """Check value against acceptable derivative methods."""
        valid = (
            'default',
            'fft',
            'chain',
            'highmen',
            'lowmem',
        )
        assert any(value == v for v in valid), "unexpected derivative method"
        return value

    @validator('entropy')
    def _valid_entropy(cls, value: str) -> str:
        """Check value against acceptable electrolyte entropy schemes."""
        valid = (
            'ions',
            'full',
        )
        assert any(value == v for v in valid), "unexpected entropy scheme"
        return value


class SemiconductorModel(YamlModelMixin, BaseModel):
    """
    Model for semiconductor parameters.
    """
    permittivity = 1.0
    carrier_density = 0.0
    distance = 0.0
    spread = 0.5

    @validator('spread')
    def _gt_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than zero."""
        assert value > 0, "must be greater than zero"
        return value

    @validator(
        'carrier_density',
        'distance',
    )
    def _ge_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, "must be greater than or equal to zero"
        return value


class SolventModel(YamlModelMixin, BaseModel):
    """
    Model for solvent parameters.
    """
    mode = 'electronic'
    radius_mode = 'uff'
    deriv_method = 'default'
    deriv_core = 'fft'
    distance = 1.0
    spread = 0.5
    radius = 0.0
    alpha = 1.0
    softness = 0.5
    stype = 2
    rhomax = 5e-3
    rhomin = 1e-4
    tbeta = 4.8
    radial_scale = 2.0
    radial_spread = 0.5
    filling_threshold = 0.825
    filling_spread = 0.02

    @validator(
        'alpha',
        'softness',
        'distance',
        'spread',
        'radial_spread',
        'filling_threshold',
        'filling_spread',
    )
    def _gt_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than zero."""
        assert value > 0, "must be greater than zero"
        return value

    @validator(
        'rhomax',
        'rhomin',
        'tbeta',
        'radius',
    )
    def _ge_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, "must be greater than or equal to zero"
        return value

    @validator('radial_scale')
    def _ge_one(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to one."""
        assert value >= 1, "must be greater than or equal to one"
        return value

    @validator('stype')
    def _valid_switching_function_type(cls, value: int) -> int:
        """Check that switching function type is valid."""
        assert 0 <= value <= 2, "stype out of range (0<=stype<=0)"
        return value

    @validator('mode')
    def _valid_mode(cls, value: str) -> str:
        """Check value against acceptable solvent/electrolyte modes."""
        valid = (
            'electronic',
            'ionic',
            'full',
            'system',
        )
        assert any(value == v for v in valid), "unexpected solvent mode"
        return value

    @validator('deriv_method')
    def _valid_deriv_method(cls, value: str) -> str:
        """Check value against acceptable derivative methods."""
        valid = (
            'default',
            'fft',
            'chain',
            'highmen',
            'lowmem',
        )
        assert any(value == v for v in valid), "unexpected derivative method"
        return value

    @validator('radius_mode')
    def _valid_radius_mode(cls, value: str) -> str:
        """Check value against acceptable radius modes."""
        valid = (
            'pauling',
            'bondi',
            'uff',
            'muff',
        )
        assert any(value == v for v in valid), "unexpected radius mode"
        return value

    @validator('deriv_core')
    def _valid_derivatives_core(cls, value: str) -> str:
        """Check value against acceptable derivatives cores."""
        valid = ('fft', )
        assert any(value == v for v in valid), "unexpected derivatives core"
        return value


class ElectrostaticsModel(YamlModelMixin, BaseModel):
    """
    Model for numerical parameters.
    """
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

    @validator(
        'tol',
        'step',
        'ndiis',
        'inner_tol',
        'inner_mix',
    )
    def _gt_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than zero."""
        assert value > 0, "must be greater than zero"
        return value

    @validator('screening')
    def _ge_zero(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than or equal to zero."""
        assert value >= 0, "must be greater than or equal to zero"
        return value

    @validator(
        'maxstep',
        'inner_maxstep',
    )
    def _gt_one(cls, value: IntFloat) -> IntFloat:
        """Check that value is greater than one."""
        assert value > 1, f"{value} must be greater than one"
        return value

    @validator('problem')
    def _valid_problem(cls, value: str) -> str:
        """Check value against acceptable electrostatic problems."""
        valid = (
            'none',
            'poisson',
            'generalized',
            'pb',
            'modpb',
            'linpb',
            'linmodpb',
        )
        assert any(value == v for v in valid), "unexpected problem"
        return value

    @validator('solver')
    def _valid_solver(cls, value: str) -> str:
        """Check value against acceptable electrostatic solvers."""
        valid = (
            'none',
            'cg',
            'sd',
            'fixed-point',
            'newton',
            'nested',
            'direct',
        )
        assert any(value == v for v in valid), "unexpected solver"
        return value

    @validator('auxiliary')
    def _valid_auxiliary_scheme(cls, value: str) -> str:
        """Check value against acceptable auxiliary schemes."""
        valid = (
            'none',
            'full',
            'ioncc',
        )
        assert any(value == v for v in valid), "unexpected auxiliary scheme"
        return value

    @validator('step_type')
    def _valid_step_type(cls, value: str) -> str:
        """Check value against acceptable step types."""
        valid = (
            'optimal',
            'input',
            'random',
        )
        assert any(value == v for v in valid), "unexpected step type"
        return value

    @validator('mix_type')
    def _valid_mix_type(cls, value: str) -> str:
        """Check value against acceptable mix types."""
        valid = (
            'linear',
            'anderson',
            'diis',
            'broyden',
        )
        assert any(value == v for v in valid), "unexpected mix type"
        return value

    @validator('preconditioner')
    def _valid_preconditioner(cls, value: str) -> str:
        """Check value against acceptable preconditioners."""
        valid = (
            'sqrt',
            'left',
        )
        assert any(value == v for v in valid), "unexpected preconditioner"
        return value

    @validator('screening_type')
    def _valid_screening_type(cls, value: str) -> str:
        """Check value against acceptable screening types."""
        valid = (
            'none',
            'input',
            'linear',
            'optimal',
        )
        assert any(value == v for v in valid), "unexpected screening type"
        return value

    @validator(
        'core',
        'inner_core',
    )
    def _valid_core(cls, value: str) -> str:
        """Check value against acceptable electrostatic cores."""
        valid = ('fft', )
        assert any(value == v for v in valid), "unexpected core"
        return value

    @validator('inner_solver')
    def _valid_inner_solver(cls, value: str) -> str:
        """Check value against acceptable electrostatic inner solvers."""
        valid = (
            'none',
            'cg',
            'sd',
            'fixed-point',
            'newton',
            'direct',
        )
        assert any(value == v for v in valid), "unexpected inner solver"
        return value


class PBCModel(YamlModelMixin, BaseModel):
    """
    Model for PBC parameters.
    """
    correction = 'none'
    core = '1da'
    dim = 0
    axis = 3

    @validator('correction')
    def _valid_correction(cls, value: str) -> str:
        """Check value against acceptable pbc corrections."""
        valid = (
            'none',
            'parabolic',
            'gcs',
            'ms',
        )
        assert any(value == v for v in valid), "unexpected correction"
        return value

    @validator('core')
    def _valid_core(cls, value: str) -> str:
        """Check value against acceptable pbc correction cores."""
        valid = ('1da', )
        assert any(value == v for v in valid), "unexpected pbc core"
        return value

    @validator('dim')
    def _valid_dimension(cls, value: int) -> int:
        """Check that the dimensions are reasonable (0<=>3)."""
        assert 0 <= value <= 3, "dimensions out of range (0<=dim<=3)"
        return value

    @validator('axis')
    def _valid_axis(cls, value: int) -> int:
        """Check that the axis is reasonable (1<=>3)."""
        assert 1 <= value <= 3, "axis out of range (1<=axis<=3)"
        return value


class InputModel(YamlModelMixin, BaseModel):
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
    externals: Optional[ExternalsContainer] = None
    regions: Optional[RegionsContainer] = None

    def __init__(self, natoms: int, **data: Dict[str, Any]) -> None:
        super().__init__(**data)
        self._adjust_to_natoms(natoms)
        self._adjust_input()
        self._final_validation()

    def _adjust_to_natoms(self, natoms: int) -> None:
        """Scale ion input arrays to size of number of atoms."""

        for array in (
                self.ions.atomicspread,
                self.ions.corespread,
                self.ions.solvationrad,
        ):

            if len(array) == 1 and natoms != 1: array *= natoms

            if len(array) != natoms:
                raise ValueError("array size not equal to number of atoms")

    def _adjust_input(self) -> None:
        """Adjust input/default parameters based on user input."""
        self._adjust_environment()
        self._adjust_derivatives_method()
        self._adjust_electrostatics()

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
            if self.solvent.mode in {'electronic', 'full'}:
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
            if self.solvent.mode in {'electronic', 'full', 'system'}:
                self.solvent.deriv_method = 'chain'

            # ionic interface
            elif self.solvent.mode == 'ionic':
                self.solvent.deriv_method = 'lowmem'

    def _adjust_electrostatics(self) -> None:
        """Adjust electrostatics according to solvent properties."""
        self._check_electrolyte_input()
        self._check_dielectric_input()

    def _check_electrolyte_input(self) -> None:
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

        if self.pbc.correction == 'gcs' or \
            self.electrolyte.formula is not None:

            if self.electrolyte.deriv_method == 'default':

                # non-ionic interfaces
                if self.electrolyte.mode in {'electronic', 'full', 'system'}:
                    self.electrolyte.deriv_method = 'chain'

                # ionic interface
                elif self.electrolyte.mode == 'ionic':
                    self.electrolyte.deriv_method = 'lowmem'

    def _check_dielectric_input(self) -> None:
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

    def _final_validation(self) -> None:
        """Check for bad input values."""
        self._validate_derivatives_method()
        self._validate_electrostatics()

    def _validate_derivatives_method(self) -> None:
        """Check for bad derivatives method."""

        # non-ionic interfaces
        if self.solvent.mode in {'electronic', 'full', 'system'}:
            if 'mem' in self.solvent.deriv_method:
                raise ValueError(
                    "Only 'fft' or 'chain' are allowed with electronic interfaces"
                )

        # ionic interface
        elif self.solvent.mode == 'ionic':
            if self.solvent.deriv_method == 'chain':
                raise ValueError(
                    "Only 'highmem', 'lowmem', and 'fft' are allowed with ionic interfaces"
                )

    def _validate_electrostatics(self) -> None:
        """Check for bad electrostatics input."""

        # rhomax/rhomin validation
        if self.solvent.rhomax < self.solvent.rhomin:
            raise ValueError("rhomax < rhomin")

        # electrolyte rhomax/rhomin validation
        if self.electrolyte.rhomax < self.electrolyte.rhomin:
            raise ValueError("electrolyte rhomax < electrolyte rhomin")

        # pbc dim validation
        if self.pbc.dim == 1:
            raise ValueError("1D periodic boundary correction not implemented")

        # electrolyte validation

        if self.pbc.correction == 'gcs':

            if self.electrolyte.distance == 0.0:
                raise ValueError(
                    "electrolyte distance must be greater than zero for gcs correction"
                )

        if self.pbc.correction == 'gcs' or \
            self.electrolyte.formula is not None:

            # non-ionic interfaces
            if self.electrolyte.mode in {'electronic', 'full', 'system'}:
                if 'mem' in self.electrolyte.deriv_method:
                    raise ValueError(
                        "Only 'fft' or 'chain' are allowed with electronic interfaces"
                    )

            # ionic interface
            elif self.electrolyte.mode == 'ionic':
                if self.electrolyte.deriv_method == 'chain':
                    raise ValueError(
                        "Only 'highmem', 'lowmem', and 'fft' are allowed with ionic interfaces"
                    )

        # problem/solver validation

        if self.electrostatics.problem == 'generalized':

            if self.electrostatics.solver == 'direct' or \
                self.electrostatics.inner_solver == 'direct':
                raise ValueError(
                    "Cannot use a direct solver for the Generalized Poisson eq."
                )

        elif "pb" in self.electrostatics.problem:

            if "lin" in self.electrostatics.problem:
                solvers = {'none', 'cg', 'sd'}

                if self.electrostatics.solver not in solvers or \
                    self.electrostatics.inner_solver not in solvers:
                    raise ValueError(
                        "Only gradient-based solver for the linearized Poisson-Boltzmann eq."
                    )

                if self.pbc.correction != 'parabolic':
                    raise ValueError(
                        "Linearized-PB problem requires parabolic pbc correction"
                    )

            else:
                solvers = {'direct', 'cg', 'sd'}

                if self.electrostatics.solver in solvers or \
                    self.electrostatics.inner_solver in solvers:
                    raise ValueError(
                        "No direct or gradient-based solver for the full Poisson-Boltzmann eq."
                    )

        problems = {'pb, modpb, generalized'}

        if self.electrostatics.inner_solver != 'none' and \
            self.electrostatics.problem not in problems:
            raise ValueError("Only pb or modpb problems allow inner solver")
