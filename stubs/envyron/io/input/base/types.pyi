from collections.abc import Generator
from typing import Any, List, Union

IntFloat = Union[int, float]
IntGT1: Any
FloatGE1: Any
Dimensions: Any
Axis: Any
Environment: Any
SolventMode: Any
DerivativeMethod: Any
EntropyScheme: Any
RadiusMode: Any
DerivativeCore: Any
ElectrostaticProblem: Any
ElectrostaticSolver: Any
AuxiliaryScheme: Any
StepType: Any
MixType: Any
Preconditioner: Any
ScreeningType: Any
ElectrostaticCore: Any
ElectrostaticInnerSolver: Any
PBCCorrection: Any
PBCCore: Any


def int_list(value: List[Any]) -> List[int]:
    ...


def float_list(value: List[Any]) -> List[float]:
    ...


def list_ge_zero(value: List[IntFloat]) -> List[IntFloat]:
    ...


def list_gt_zero(value: List[IntFloat]) -> List[IntFloat]:
    ...


def ne_zero(value: IntFloat) -> IntFloat:
    ...


class NonZeroFloat(float):

    @classmethod
    def __get_validators__(cls) -> Generator[Any, None, None]:
        ...


class NonNegativeFloatList(list):

    @classmethod
    def __get_validators__(cls) -> Generator[Any, None, None]:
        ...


class PositiveFloatList(list):

    @classmethod
    def __get_validators__(cls) -> Generator[Any, None, None]:
        ...


class Vector(list):

    @classmethod
    def __get_validators__(cls) -> Generator[Any, None, None]:
        ...

    @classmethod
    def vectorize(cls, value: List[Any]) -> List[Any]:
        ...


class FloatVector(Vector):

    @classmethod
    def __get_validators__(cls) -> Generator[Any, None, None]:
        ...


class NonNegativeIntVector(Vector):

    @classmethod
    def __get_validators__(cls) -> Generator[Any, None, None]:
        ...
