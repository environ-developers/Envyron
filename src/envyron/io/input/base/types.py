from typing import (
    Any,
    List,
    Literal,
    Union,
)

from typing_extensions import Annotated

from pydantic import (
    confloat,
    conint,
)

from pydantic.validators import (
    int_validator,
    float_validator,
)

from pydantic.errors import (
    NumberNotGeError,
    NumberNotGtError,
)

# numerical type aliases
IntFloat = Union[int, float]
IntGT1 = Annotated[int, conint(gt=1)]
FloatGE1 = Annotated[float, confloat(ge=1)]
Dimensions = Annotated[int, conint(ge=0, le=3)]
Axis = Annotated[int, conint(ge=1, le=3)]

# yapf: disable

# literal type aliases
Environment = Literal[
    'input',
    'vacuum',
    'water',
    'water-cation',
    'water-anion',
]

SolventMode = Literal[
    'electronic',
    'ionic',
    'full',
    'system',
]

DerivativeMethod = Literal[
    'default',
    'fft',
    'chain',
    'highmen',
    'lowmem',
]

EntropyScheme = Literal[
    'ions',
    'full',
]

RadiusMode = Literal[
    'pauling',
    'bondi',
    'uff',
    'muff',
]

DerivativeCore = Literal[
    'fft',
]

ElectrostaticProblem = Literal[
    'none',
    'poisson',
    'generalized',
    'pb',
    'modpb',
    'linpb',
    'linmodpb'
]

ElectrostaticSolver = Literal[
    'none',
    'cg',
    'sd',
    'fixed-point',
    'newton',
    'nested',
    'direct',
]

AuxiliaryScheme = Literal[
    'none',
    'full',
    'ioncc',
]

StepType = Literal[
    'optimal',
    'input',
    'random',
]

MixType = Literal[
    'linear',
    'anderson',
    'diis',
    'broyden',
]

Preconditioner = Literal[
    'sqrt',
    'left',
]

ScreeningType = Literal[
    'none',
    'input',
    'linear',
    'optimal',
]

ElectrostaticCore = Literal[
    'fft',
]

ElectrostaticInnerSolver = Literal[
    'none',
    'cg',
    'sd',
    'fixed-point',
    'newton',
    'direct',
]

PBCCorrection = Literal[
    'none',
    'parabolic',
    'gcs',
    'ms',
]

PBCCore = Literal[
    '1da',
]

# yapf: enable


def int_list(value: List[Any]) -> List[int]:
    """Convert items to integers."""
    return [int_validator(v) for v in value]


def float_list(value: List[Any]) -> List[float]:
    """Convert items to floats."""
    return [float_validator(v) for v in value]


def list_ge_zero(value: List[IntFloat]) -> List[IntFloat]:
    """Check that all array values are greater than or equal to zero."""
    for v in value:
        if v < 0: raise NumberNotGeError(limit_value=0)
    return value


def list_gt_zero(value: List[IntFloat]) -> List[IntFloat]:
    """Check that all array values are greater than zero."""
    for v in value:
        if v <= 0: raise NumberNotGtError(limit_value=0)
    return value


def ne_zero(value: IntFloat) -> IntFloat:
    """Check that value is not zero."""
    if value == 0: raise ValueError("ensure this value is not zero")
    return value


class NonZeroFloat(float):
    """
    docstring
    """

    @classmethod
    def __get_validators__(cls):
        yield float_validator
        yield ne_zero


class NonNegativeFloatList(list):
    """
    docstring
    """

    @classmethod
    def __get_validators__(cls):
        yield float_list
        yield list_ge_zero


class PositiveFloatList(list):
    """
    docstring
    """

    @classmethod
    def __get_validators__(cls):
        yield float_list
        yield list_gt_zero


class Vector(list):
    """
    docstring
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.vectorize

    @classmethod
    def vectorize(cls, value: List[Any]) -> List[Any]:
        """Scale vector input to 3D."""
        if len(value) == 1: value = value * 3
        assert len(value) == 3, "array size should be 3"
        return value


class FloatVector(Vector):
    """
    docstring
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.vectorize
        yield float_list


class NonNegativeIntVector(Vector):
    """
    docstring
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.vectorize
        yield int_list
        yield list_ge_zero
