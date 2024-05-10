from typing import Any, List
from pydantic import NumberNotGeError, NumberNotGtError
from pytest import mark, raises

from envyron.io.input.types import (
    IntFloat,
    Vector,
    int_list,
    float_list,
    list_ge_zero,
    list_gt_zero,
    ne_zero,
)


@mark.parametrize(
    'numbers, expected',
    [
        ([], []),
        ([1, 2, 3], [1, 2, 3]),
        ([1.0, 2.0, 3.0], [1, 2, 3]),
        (['1', '2', '3'], [1, 2, 3]),
    ],
)
def test_int_list(numbers: List[Any], expected: List[int]) -> None:
    """Test integer list validator."""
    converted = int_list(numbers)
    assert all(isinstance(n, int) for n in converted)
    assert converted == expected


@mark.parametrize(
    'numbers, expected',
    [
        ([], []),
        ([1, 2, 3], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        (['1', '2', '3'], [1.0, 2.0, 3.0]),
    ],
)
def test_float_list(numbers: List[Any], expected: List[float]) -> None:
    """Test float list validator."""
    converted = float_list(numbers)
    assert all(isinstance(n, float) for n in converted)
    assert converted == expected


def test_list_ge_zero() -> None:
    """Test greater than or equal to zero list validator."""
    assert list_ge_zero([0, 1, 2, 3]) == [0, 1, 2, 3]
    with raises(NumberNotGeError):
        assert list_ge_zero([-1, 0, 1])


def test_list_gt_zero() -> None:
    """Test greater than zero list validator."""
    assert list_gt_zero([1, 2, 3]) == [1, 2, 3]
    with raises(NumberNotGtError):
        assert list_gt_zero([0, 1, 2])


def test_ne_zero() -> None:
    """Test non-zero number validator."""
    assert ne_zero(1)
    with raises(ValueError):
        assert ne_zero(0)


@mark.parametrize(
    'entry',
    [
        ([1]),
        ([2.0]),
        ([0, 0, 0]),
        ([1.0, -2.0, 3.0]),
    ],
)
def test_vectorize(entry: List[IntFloat]) -> None:
    """Test vector validator."""
    vector = Vector.vectorize(entry)
    assert len(vector) == 3
    if len(entry) == 1: assert vector == entry * 3
