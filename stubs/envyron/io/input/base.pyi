from envyron.io.input.types import (
    AuxiliaryScheme as AuxiliaryScheme,
    Axis as Axis,
    DerivativeCore as DerivativeCore,
    DerivativeMethod as DerivativeMethod,
    Dimensions as Dimensions,
    ElectrostaticCore as ElectrostaticCore,
    ElectrostaticInnerSolver as ElectrostaticInnerSolver,
    ElectrostaticProblem as ElectrostaticProblem,
    ElectrostaticSolver as ElectrostaticSolver,
    EntropyScheme as EntropyScheme,
    Environment as Environment,
    FloatGE1 as FloatGE1,
    FloatVector as FloatVector,
    IntGT1 as IntGT1,
    MixType as MixType,
    NonNegativeFloatList as NonNegativeFloatList,
    NonNegativeIntVector as NonNegativeIntVector,
    NonZeroFloat as NonZeroFloat,
    PBCCore as PBCCore,
    PBCCorrection as PBCCorrection,
    PositiveFloatList as PositiveFloatList,
    Preconditioner as Preconditioner,
    RadiusMode as RadiusMode,
    ScreeningType as ScreeningType,
    SolventMode as SolventMode,
    StepType as StepType,
)
from pydantic import (
    BaseModel as PydanticBaseModel,
    NonNegativeFloat as NonNegativeFloat,
    NonNegativeInt as NonNegativeInt,
    PositiveFloat as PositiveFloat,
    PositiveInt as PositiveInt,
    confloat as confloat,
    conint as conint,
)
from typing import List, Literal, Optional
from typing_extensions import Annotated as Annotated


class BaseModel(PydanticBaseModel):

    class Config:
        validate_assignment: bool


class CardModel(BaseModel):
    pos: FloatVector
    spread: NonNegativeFloat
    dim: Dimensions
    axis: Axis


class ExternalModel(CardModel):
    charge: NonZeroFloat


class RegionModel(CardModel):
    static: FloatGE1
    optical: FloatGE1
    width: NonNegativeFloat


class CardContainerModel(BaseModel):
    units: Literal['bohr', 'angstrom']
    number: NonNegativeInt


class ExternalsContainerModel(CardContainerModel):
    functions: List[List[ExternalModel]]


class RegionsContainerModel(CardContainerModel):
    functions: List[List[RegionModel]]


class ControlModel(BaseModel):
    debug: bool
    restart: bool
    verbosity: NonNegativeInt
    threshold: NonNegativeFloat
    nskip: NonNegativeInt
    ecut: NonNegativeFloat
    nrep: NonNegativeIntVector
    need_electrostatic: bool


class EnvironmentModel(BaseModel):
    type: Environment
    surface_tension: NonNegativeFloat
    pressure: float
    confine: NonNegativeFloat
    static_permittivity: FloatGE1
    optical_permittivity: FloatGE1
    temperature: NonNegativeFloat


class IonsModel(BaseModel):
    atomicspread: PositiveFloatList
    corespread: NonNegativeFloatList
    solvationrad: PositiveFloatList


class SystemModel(BaseModel):
    ntyp: NonNegativeInt
    dim: Dimensions
    axis: Axis
    pos: FloatVector


class ElectrolyteModel(BaseModel):
    linearized: bool
    mode: SolventMode
    entropy: EntropyScheme
    deriv_method: DerivativeMethod
    concentration: NonNegativeFloat
    formula: Optional[List[int]]
    cionmax: NonNegativeFloat
    rion: NonNegativeFloat
    distance: NonNegativeFloat
    spread: PositiveFloat
    rhomax: NonNegativeFloat
    rhomin: NonNegativeFloat
    tbeta: NonNegativeFloat
    alpha: PositiveFloat
    softness: PositiveFloat


class SemiconductorModel(BaseModel):
    permittivity: FloatGE1
    carrier_density: NonNegativeFloat
    distance: NonNegativeFloat
    spread: PositiveFloat


class SolventModel(BaseModel):
    mode: SolventMode
    radius_mode: RadiusMode
    deriv_method: DerivativeMethod
    deriv_core: DerivativeCore
    distance: NonNegativeFloat
    spread: PositiveFloat
    radius: NonNegativeFloat
    alpha: PositiveFloat
    softness: PositiveFloat
    stype: Annotated[int, None]
    rhomax: NonNegativeFloat
    rhomin: NonNegativeFloat
    tbeta: NonNegativeFloat
    radial_scale: FloatGE1
    radial_spread: PositiveFloat
    filling_threshold: PositiveFloat
    filling_spread: PositiveFloat
    field_aware: bool
    field_factor: NonNegativeFloat
    field_asymmetry: Annotated[float, None]
    field_min: NonNegativeFloat
    field_max: NonNegativeFloat


class ElectrostaticsModel(BaseModel):
    problem: ElectrostaticProblem
    tol: PositiveFloat
    solver: ElectrostaticSolver
    auxiliary: AuxiliaryScheme
    step_type: StepType
    step: PositiveFloat
    maxstep: IntGT1
    mix_type: MixType
    ndiis: PositiveInt
    mix: PositiveFloat
    preconditioner: Preconditioner
    screening_type: ScreeningType
    screening: NonNegativeFloat
    core: ElectrostaticCore
    inner_solver: ElectrostaticInnerSolver
    inner_core: ElectrostaticCore
    inner_tol: PositiveFloat
    inner_maxstep: IntGT1
    inner_mix: PositiveFloat
    inner_problem: ElectrostaticProblem


class PBCModel(BaseModel):
    correction: PBCCorrection
    core: PBCCore
    dim: Dimensions
    axis: Axis
