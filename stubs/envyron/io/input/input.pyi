from .base import (
    BaseModel as BaseModel,
    ControlModel as ControlModel,
    ElectrolyteModel as ElectrolyteModel,
    ElectrostaticsModel as ElectrostaticsModel,
    EnvironmentModel as EnvironmentModel,
    ExternalsContainerModel as ExternalsContainerModel,
    IonsModel as IonsModel,
    PBCModel as PBCModel,
    RegionsContainerModel as RegionsContainerModel,
    SemiconductorModel as SemiconductorModel,
    SolventModel as SolventModel,
    SystemModel as SystemModel,
)
from typing import Any, Dict, Optional


class Input(BaseModel):
    control: Optional[ControlModel]
    environment: Optional[EnvironmentModel]
    ions: Optional[IonsModel]
    system: Optional[SystemModel]
    electrolyte: Optional[ElectrolyteModel]
    semiconductor: Optional[SemiconductorModel]
    solvent: Optional[SolventModel]
    electrostatics: Optional[ElectrostaticsModel]
    pbc: Optional[PBCModel]
    externals: Optional[ExternalsContainerModel]
    regions: Optional[RegionsContainerModel]

    def __init__(self,
                 natoms: int,
                 filename: Optional[str] = ...,
                 **params: Dict[str, Any]) -> None:
        ...

    def read(self, filename: str) -> Dict[str, Any]:
        ...

    def adjust_ionic_arrays(self, natoms: int) -> None:
        ...

    def apply_smart_defaults(self) -> None:
        ...

    def sanity_check(self) -> None:
        ...

    def _adjust_environment(self) -> None:
        ...

    def _adjust_derivatives_method(self) -> None:
        ...

    def _adjust_electrostatics(self) -> None:
        ...

    def _adjust_electrolyte_dependent_electrostatics(self) -> None:
        ...

    def _adjust_dielectric_dependent_electrostatics(self) -> None:
        ...

    def _validate_solvent(self) -> None:
        ...

    def _validate_electrolyte(self) -> None:
        ...

    def _validate_electrostatics(self) -> None:
        ...
