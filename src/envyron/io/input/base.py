from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Annotated

from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    confloat,
    conint,
    conlist,
    validator,
    BaseModel as PydanticBaseModel,
)
from pydantic_yaml import YamlModelMixin

# type aliases/variable
IntFloat = Union[int, float]
IntGT1 = Annotated[int, conint(gt=1)]
IntVector = Annotated[List[int], conlist(int, min_items=1, max_items=3)]
FloatGE1 = Annotated[float, confloat(ge=1)]
FloatList = Annotated[List[float], conlist(float, min_items=1)]
FloatVector = Annotated[List[float], conlist(float, min_items=1, max_items=3)]
Dimensions = Annotated[int, conint(ge=0, le=3)]
Axis = Annotated[int, conint(ge=1, le=3)]


def _valid_option(value: str, valid: Tuple[str, ...]) -> str:
    """Check that value is a valid option."""
    assert any(value == v for v in valid), f"value is not one of {valid}"
    return value


class BaseModel(PydanticBaseModel):
    """
    Class for global configuration of validation mechanics.
    """

    class Config:
        validate_assignment = True


class CardModel(YamlModelMixin, BaseModel):
    """
    Model for card input.
    """
    pos: List[float] = [0.0, 0.0, 0.0]
    spread: NonNegativeFloat = 0.5
    dim: Dimensions = 0
    axis: Axis = 3


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


class RegionModel(CardModel):
    """
    Model for a single region function.
    """
    static: FloatGE1 = 1.0
    optical: FloatGE1 = 1.0
    width: NonNegativeFloat = 0.0


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
        return _valid_option(value, valid)


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
    verbosity: NonNegativeInt = 0
    threshold: NonNegativeFloat = 0.1
    nskip: NonNegativeInt = 1
    nrep: IntVector = [0, 0, 0]
    need_electrostatic = False

    @validator('nrep')
    def _vectorize(cls, value: List[IntFloat]) -> List[IntFloat]:
        """Scale vector input to 3D."""
        if len(value) == 1: value = value * 3
        assert len(value) == 3, "array size should be 3"
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
    surface_tension: NonNegativeFloat = 0.0
    pressure = 0.0
    confine: NonNegativeFloat = 0.0
    static_permittivity: FloatGE1 = 1.0
    optical_permittivity: FloatGE1 = 1.0
    temperature: NonNegativeFloat = 300.0

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
        return _valid_option(value, valid)


class IonsModel(YamlModelMixin, BaseModel):
    """
    Model for ions parameters.
    """
    atomicspread: FloatList = [0.5]
    corespread: FloatList = [0.5]
    solvationrad: FloatList = [0.0]

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
    ntyp: NonNegativeInt = 0
    dim: Dimensions = 0
    axis: Axis = 3
    pos: FloatVector = [0.0, 0.0, 0.0]

    @validator('pos')
    def _vectorize(cls, value: List[IntFloat]) -> List[IntFloat]:
        """Scale vector input to 3D."""
        if len(value) == 1: value = value * 3
        assert len(value) == 3, "array size should be 3"
        return value


class ElectrolyteModel(YamlModelMixin, BaseModel):
    """
    Model for electrolyte parameters.
    """
    linearized = False
    mode = 'electronic'
    entropy = 'full'
    deriv_method = 'default'
    concentration: NonNegativeFloat = 0.0
    formula: Optional[List[int]] = None
    cionmax: NonNegativeFloat = 0.0
    rion: NonNegativeFloat = 0.0
    distance: NonNegativeFloat = 0.0
    spread: PositiveFloat = 0.5
    rhomax: NonNegativeFloat = 5e-3
    rhomin: NonNegativeFloat = 1e-4
    tbeta: NonNegativeFloat = 4.8
    alpha: NonNegativeFloat = 1.0
    softness: PositiveFloat = 0.5

    @validator('mode')
    def _valid_mode(cls, value: str) -> str:
        """Check value against acceptable solvent/electrolyte modes."""
        valid = (
            'electronic',
            'ionic',
            'full',
            'system',
        )
        return _valid_option(value, valid)

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
        return _valid_option(value, valid)

    @validator('entropy')
    def _valid_entropy(cls, value: str) -> str:
        """Check value against acceptable electrolyte entropy schemes."""
        valid = (
            'ions',
            'full',
        )
        return _valid_option(value, valid)


class SemiconductorModel(YamlModelMixin, BaseModel):
    """
    Model for semiconductor parameters.
    """
    permittivity: FloatGE1 = 1.0
    carrier_density: NonNegativeFloat = 0.0
    distance: NonNegativeFloat = 0.0
    spread: PositiveFloat = 0.5


class SolventModel(YamlModelMixin, BaseModel):
    """
    Model for solvent parameters.
    """
    mode = 'electronic'
    radius_mode = 'uff'
    deriv_method = 'default'
    deriv_core = 'fft'
    distance: PositiveFloat = 1.0
    spread: PositiveFloat = 0.5
    radius: NonNegativeFloat = 0.0
    alpha: PositiveFloat = 1.0
    softness: PositiveFloat = 0.5
    stype: Annotated[int, conint(ge=0, le=2)] = 2
    rhomax: NonNegativeFloat = 5e-3
    rhomin: NonNegativeFloat = 1e-4
    tbeta: NonNegativeFloat = 4.8
    radial_scale: FloatGE1 = 2.0
    radial_spread: PositiveFloat = 0.5
    filling_threshold: PositiveFloat = 0.825
    filling_spread: PositiveFloat = 0.02

    @validator('mode')
    def _valid_mode(cls, value: str) -> str:
        """Check value against acceptable solvent/electrolyte modes."""
        valid = (
            'electronic',
            'ionic',
            'full',
            'system',
        )
        return _valid_option(value, valid)

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
        return _valid_option(value, valid)

    @validator('radius_mode')
    def _valid_radius_mode(cls, value: str) -> str:
        """Check value against acceptable radius modes."""
        valid = (
            'pauling',
            'bondi',
            'uff',
            'muff',
        )
        return _valid_option(value, valid)

    @validator('deriv_core')
    def _valid_derivatives_core(cls, value: str) -> str:
        """Check value against acceptable derivatives cores."""
        valid = ('fft', )
        return _valid_option(value, valid)


class ElectrostaticsModel(YamlModelMixin, BaseModel):
    """
    Model for numerical parameters.
    """
    problem = 'none'
    tol: PositiveFloat = 1e-5
    solver = 'none'
    auxiliary = 'none'
    step_type = 'optimal'
    step: PositiveFloat = 0.3
    maxstep: IntGT1 = 200
    mix_type = 'linear'
    ndiis: PositiveInt = 1
    mix: PositiveFloat = 0.5
    preconditioner = 'sqrt'
    screening_type = 'none'
    screening: NonNegativeFloat = 0.0
    core = 'fft'
    inner_solver = 'none'
    inner_core = 'fft'
    inner_tol: PositiveFloat = 1e-10
    inner_maxstep: IntGT1 = 200
    inner_mix: PositiveFloat = 0.5

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
        return _valid_option(value, valid)

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
        return _valid_option(value, valid)

    @validator('auxiliary')
    def _valid_auxiliary_scheme(cls, value: str) -> str:
        """Check value against acceptable auxiliary schemes."""
        valid = (
            'none',
            'full',
            'ioncc',
        )
        return _valid_option(value, valid)

    @validator('step_type')
    def _valid_step_type(cls, value: str) -> str:
        """Check value against acceptable step types."""
        valid = (
            'optimal',
            'input',
            'random',
        )
        return _valid_option(value, valid)

    @validator('mix_type')
    def _valid_mix_type(cls, value: str) -> str:
        """Check value against acceptable mix types."""
        valid = (
            'linear',
            'anderson',
            'diis',
            'broyden',
        )
        return _valid_option(value, valid)

    @validator('preconditioner')
    def _valid_preconditioner(cls, value: str) -> str:
        """Check value against acceptable preconditioners."""
        valid = (
            'sqrt',
            'left',
        )
        return _valid_option(value, valid)

    @validator('screening_type')
    def _valid_screening_type(cls, value: str) -> str:
        """Check value against acceptable screening types."""
        valid = (
            'none',
            'input',
            'linear',
            'optimal',
        )
        return _valid_option(value, valid)

    @validator(
        'core',
        'inner_core',
    )
    def _valid_core(cls, value: str) -> str:
        """Check value against acceptable electrostatic cores."""
        valid = ('fft', )
        return _valid_option(value, valid)

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
        return _valid_option(value, valid)


class PBCModel(YamlModelMixin, BaseModel):
    """
    Model for PBC parameters.
    """
    correction = 'none'
    core = '1da'
    dim: Dimensions = 0
    axis: Axis = 3

    @validator('correction')
    def _valid_correction(cls, value: str) -> str:
        """Check value against acceptable pbc corrections."""
        valid = (
            'none',
            'parabolic',
            'gcs',
            'ms',
        )
        return _valid_option(value, valid)

    @validator('core')
    def _valid_core(cls, value: str) -> str:
        """Check value against acceptable pbc correction cores."""
        valid = ('1da', )
        return _valid_option(value, valid)


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
