from pydantic import BaseModel as PydanticBaseModel, NonNegativeFloat as NonNegativeFloat, NonNegativeInt as NonNegativeInt, PositiveFloat as PositiveFloat, PositiveInt as PositiveInt
from pydantic_yaml import YamlModelMixin
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Annotated

IntFloat = Union[int, float]
IntGT1: Any
IntVector: Any
FloatGE1: Any
FloatList: Any
FloatVector: Any
Dimensions: Any
Axis: Any

def _valid_option(value: str, valid: Tuple[str, ...]) -> str: ...

class BaseModel(PydanticBaseModel):
    class Config:
        validate_assignment: bool

class CardModel(YamlModelMixin, BaseModel):
    pos: List[float]
    spread: NonNegativeFloat
    dim: Dimensions
    axis: Axis

class ExternalModel(CardModel):
    charge: float
    def ne_zero(cls, value: IntFloat) -> IntFloat: ...

class RegionModel(CardModel):
    static: FloatGE1
    optical: FloatGE1
    width: NonNegativeFloat

class CardContainer(YamlModelMixin, BaseModel):
    units: str
    def valid_units(cls, value: str) -> str: ...

class ExternalsContainer(CardContainer):
    functions: List[List[ExternalModel]]

class RegionsContainer(CardContainer):
    functions: List[List[RegionModel]]

class ControlModel(YamlModelMixin, BaseModel):
    debug: bool
    restart: bool
    verbosity: NonNegativeInt
    threshold: NonNegativeFloat
    nskip: NonNegativeInt
    nrep: IntVector
    need_electrostatic: bool
    def _vectorize(cls, value: List[IntFloat]) -> List[IntFloat]: ...
    def _ge_zero_many(cls, value: IntFloat) -> IntFloat: ...

class EnvironmentModel(YamlModelMixin, BaseModel):
    type: str
    surface_tension: NonNegativeFloat
    pressure: float
    confine: NonNegativeFloat
    static_permittivity: FloatGE1
    optical_permittivity: FloatGE1
    temperature: NonNegativeFloat
    def _valid_environment_type(cls, value: str) -> str: ...

class IonsModel(YamlModelMixin, BaseModel):
    atomicspread: FloatList
    corespread: FloatList
    solvationrad: FloatList
    def _ge_zero_many(cls, value: IntFloat) -> IntFloat: ...

class SystemModel(YamlModelMixin, BaseModel):
    ntyp: NonNegativeInt
    dim: Dimensions
    axis: Axis
    pos: FloatVector
    def _vectorize(cls, value: List[IntFloat]) -> List[IntFloat]: ...

class ElectrolyteModel(YamlModelMixin, BaseModel):
    linearized: bool
    mode: str
    entropy: str
    deriv_method: str
    concentration: NonNegativeFloat
    formula: Optional[List[int]]
    cionmax: NonNegativeFloat
    rion: NonNegativeFloat
    distance: NonNegativeFloat
    spread: PositiveFloat
    rhomax: NonNegativeFloat
    rhomin: NonNegativeFloat
    tbeta: NonNegativeFloat
    alpha: NonNegativeFloat
    softness: PositiveFloat
    def _valid_mode(cls, value: str) -> str: ...
    def _valid_deriv_method(cls, value: str) -> str: ...
    def _valid_entropy(cls, value: str) -> str: ...

class SemiconductorModel(YamlModelMixin, BaseModel):
    permittivity: FloatGE1
    carrier_density: NonNegativeFloat
    distance: NonNegativeFloat
    spread: PositiveFloat

class SolventModel(YamlModelMixin, BaseModel):
    mode: str
    radius_mode: str
    deriv_method: str
    deriv_core: str
    distance: PositiveFloat
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
    def _valid_mode(cls, value: str) -> str: ...
    def _valid_deriv_method(cls, value: str) -> str: ...
    def _valid_radius_mode(cls, value: str) -> str: ...
    def _valid_derivatives_core(cls, value: str) -> str: ...

class ElectrostaticsModel(YamlModelMixin, BaseModel):
    problem: str
    tol: PositiveFloat
    solver: str
    auxiliary: str
    step_type: str
    step: PositiveFloat
    maxstep: IntGT1
    mix_type: str
    ndiis: PositiveInt
    mix: PositiveFloat
    preconditioner: str
    screening_type: str
    screening: NonNegativeFloat
    core: str
    inner_solver: str
    inner_core: str
    inner_tol: PositiveFloat
    inner_maxstep: IntGT1
    inner_mix: PositiveFloat
    def _valid_problem(cls, value: str) -> str: ...
    def _valid_solver(cls, value: str) -> str: ...
    def _valid_auxiliary_scheme(cls, value: str) -> str: ...
    def _valid_step_type(cls, value: str) -> str: ...
    def _valid_mix_type(cls, value: str) -> str: ...
    def _valid_preconditioner(cls, value: str) -> str: ...
    def _valid_screening_type(cls, value: str) -> str: ...
    def _valid_core(cls, value: str) -> str: ...
    def _valid_inner_solver(cls, value: str) -> str: ...

class PBCModel(YamlModelMixin, BaseModel):
    correction: str
    core: str
    dim: Dimensions
    axis: Axis
    def _valid_correction(cls, value: str) -> str: ...
    def _valid_core(cls, value: str) -> str: ...

class InputModel(YamlModelMixin, BaseModel):
    control: Optional[ControlModel]
    environment: Optional[EnvironmentModel]
    ions: Optional[IonsModel]
    system: Optional[SystemModel]
    electrolyte: Optional[ElectrolyteModel]
    semiconductor: Optional[SemiconductorModel]
    solvent: Optional[SolventModel]
    electrostatics: Optional[ElectrostaticsModel]
    pbc: Optional[PBCModel]
    externals: Optional[ExternalsContainer]
    regions: Optional[RegionsContainer]
    def __init__(self, natoms: int, **data: Dict[str, Any]) -> None: ...
    def _adjust_to_natoms(self, natoms: int) -> None: ...
    def _adjust_input(self) -> None: ...
    def _adjust_environment(self) -> None: ...
    def _adjust_derivatives_method(self) -> None: ...
    def _adjust_electrostatics(self) -> None: ...
    def _check_electrolyte_input(self) -> None: ...
    def _check_dielectric_input(self) -> None: ...
    def _final_validation(self) -> None: ...
    def _validate_derivatives_method(self) -> None: ...
    def _validate_electrostatics(self) -> None: ...
