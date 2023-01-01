from typing import (
    List,
    Literal,
    Optional,
)

from typing_extensions import Annotated

from pydantic import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    confloat,
    conint,
    BaseModel as PydanticBaseModel,
)

from envyron.io.input.types import (
    AuxiliaryScheme,
    Axis,
    DerivativeCore,
    DerivativeMethod,
    Dimensions,
    ElectrostaticCore,
    ElectrostaticInnerSolver,
    ElectrostaticProblem,
    ElectrostaticSolver,
    EntropyScheme,
    Environment,
    FloatGE1,
    FloatVector,
    IntGT1,
    MixType,
    NonNegativeFloatList,
    NonNegativeIntVector,
    NonZeroFloat,
    PBCCore,
    PBCCorrection,
    PositiveFloatList,
    Preconditioner,
    RadiusMode,
    ScreeningType,
    SolventMode,
    StepType,
)


class BaseModel(PydanticBaseModel):
    """Global configurations of validation mechanics."""

    class Config:
        validate_assignment = True


class CardModel(BaseModel):
    """Card input model."""
    pos: FloatVector = [0.0, 0.0, 0.0]  # type: ignore
    spread: NonNegativeFloat = 0.5
    dim: Dimensions = 0
    axis: Axis = 3


class ExternalModel(CardModel):
    """External function model."""
    charge: NonZeroFloat


class RegionModel(CardModel):
    """Region function model."""
    static: FloatGE1 = 1.0
    optical: FloatGE1 = 1.0
    width: NonNegativeFloat = 0.0


class CardContainerModel(BaseModel):
    """Container for card functions."""
    units: Literal['bohr', 'angstrom'] = 'bohr'
    number: NonNegativeInt = 0


class ExternalsContainerModel(CardContainerModel):
    """Container for external functions."""
    functions: List[List[ExternalModel]] = []


class RegionsContainerModel(CardContainerModel):
    """Container for region functions."""
    functions: List[List[RegionModel]] = []


class ControlModel(BaseModel):
    """Control input model."""
    debug = False
    restart = False
    verbosity: NonNegativeInt = 0
    threshold: NonNegativeFloat = 0.1
    nskip: NonNegativeInt = 1
    ecut: NonNegativeFloat = 0.0
    nrep: NonNegativeIntVector = [0, 0, 0]  # type: ignore
    need_electrostatic = False


class EnvironmentModel(BaseModel):
    """Environment input model."""
    type: Environment = 'input'
    surface_tension: NonNegativeFloat = 0.0
    pressure = 0.0
    confine: NonNegativeFloat = 0.0
    static_permittivity: FloatGE1 = 1.0
    optical_permittivity: FloatGE1 = 1.0
    temperature: NonNegativeFloat = 300.0


class IonsModel(BaseModel):
    """Ions input model."""
    atomicspread: PositiveFloatList = [0.5]  # type: ignore
    corespread: NonNegativeFloatList = [0.5]  # type: ignore
    solvationrad: PositiveFloatList = [0.0]  # type: ignore


class SystemModel(BaseModel):
    """System input model."""
    ntyp: NonNegativeInt = 0
    dim: Dimensions = 0
    axis: Axis = 3
    pos: FloatVector = [0.0, 0.0, 0.0]  # type: ignore


class ElectrolyteModel(BaseModel):
    """Electrolyte input model."""
    linearized = False
    mode: SolventMode = 'electronic'
    entropy: EntropyScheme = 'full'
    deriv_method: DerivativeMethod = 'default'
    concentration: NonNegativeFloat = 0.0
    formula: Optional[List[int]] = None
    cionmax: NonNegativeFloat = 0.0
    rion: NonNegativeFloat = 0.0
    distance: NonNegativeFloat = 0.0
    spread: PositiveFloat = 0.5
    rhomax: NonNegativeFloat = 5e-3
    rhomin: NonNegativeFloat = 1e-4
    tbeta: NonNegativeFloat = 4.8
    alpha: PositiveFloat = 1.0
    softness: PositiveFloat = 0.5


class SemiconductorModel(BaseModel):
    """Semiconductor input model."""
    permittivity: FloatGE1 = 1.0
    carrier_density: NonNegativeFloat = 0.0
    electrode_charge: NonNegativeFloat = 0.0
    charge_threshold: NonNegativeFloat = 1e-4
    distance: NonNegativeFloat = 0.0
    spread: PositiveFloat = 0.5


class SolventModel(BaseModel):
    """Solvent input model."""
    mode: SolventMode = 'electronic'
    radius_mode: RadiusMode = 'uff'
    deriv_method: DerivativeMethod = 'default'
    deriv_core: DerivativeCore = 'fft'
    distance: NonNegativeFloat = 1.0
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
    field_aware = False
    field_factor: NonNegativeFloat = 0.08
    field_asymmetry: Annotated[float, confloat(ge=-1, le=1)] = -0.32
    field_min: NonNegativeFloat = 2.0
    field_max: NonNegativeFloat = 6.0


class ElectrostaticsModel(BaseModel):
    """Electrostatics input model."""
    problem: ElectrostaticProblem = 'none'
    tol: PositiveFloat = 1e-5
    solver: ElectrostaticSolver = 'none'
    auxiliary: AuxiliaryScheme = 'none'
    step_type: StepType = 'optimal'
    step: PositiveFloat = 0.3
    maxstep: IntGT1 = 200
    mix_type: MixType = 'linear'
    ndiis: PositiveInt = 1
    mix: PositiveFloat = 0.5
    preconditioner: Preconditioner = 'sqrt'
    screening_type: ScreeningType = 'none'
    screening: NonNegativeFloat = 0.0
    core: ElectrostaticCore = 'fft'
    inner_solver: ElectrostaticInnerSolver = 'none'
    inner_core: ElectrostaticCore = 'fft'
    inner_tol: PositiveFloat = 1e-10
    inner_maxstep: IntGT1 = 200
    inner_mix: PositiveFloat = 0.5
    inner_problem: ElectrostaticProblem = 'none'


class PBCModel(BaseModel):
    """PBC input model."""
    correction: PBCCorrection = 'none'
    core: PBCCore = '1da'
    dim: Dimensions = 0
    axis: Axis = 3
