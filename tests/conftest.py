from os import chdir
from pathlib import Path
from pytest import fixture, FixtureRequest

import numpy as np
from envyron.domains.cell import EnvironGrid
from envyron.representations.density import EnvironDensity


@fixture
def datadir(request: FixtureRequest) -> None:
    """Change over to data directory."""

    # absolute path of running test module
    test_dir = Path(request.fspath.dirname).resolve()

    marker = request.node.get_closest_marker('datadir')

    if marker is None:
        raise ValueError("Missing datadir marker.")
    elif len(marker.args) == 0:
        raise ValueError("Missing name of data directory.")
    else:
        # path to data directory
        data_dir = test_dir.joinpath(marker.args[0])

    if data_dir.exists():
        if not data_dir.is_dir():
            raise ValueError(f"{data_dir} not a directory.")
    else:
        raise ValueError(f"{data_dir} not found.")

    chdir(data_dir)


@fixture
def minimal_cell() -> EnvironGrid:
    """Create a minimal unitary cell with 2 gridpoints per side"""
    at = np.eye(3)
    nr = np.array([2, 2, 2])
    return EnvironGrid(at, nr)


@fixture
def unitary_cell(request: FixtureRequest) -> EnvironGrid:
    """Create a unitary cell"""
    at = np.eye(3)
    nr = np.array([request.param, request.param, request.param])
    return EnvironGrid(at, nr)


@fixture
def cubic_cell(request: FixtureRequest) -> EnvironGrid:
    """Create a cubic cell"""
    at = np.eye(3) * request.param[1]
    nr = np.array([request.param[0], request.param[0], request.param[0]])
    return EnvironGrid(at, nr)


@fixture
def hexagonal_cell(request: FixtureRequest) -> EnvironGrid:
    """Create an hexagonal cell"""
    at = np.eye(3) * request.param[1]
    at[1, 0] = request.param[1] * 0.5
    at[1, 1] *= np.sqrt(3) * 0.5
    at[2, 2] *= request.param[2]
    nr = np.array([
        request.param[0], request.param[0],
        int(request.param[0] * request.param[2])
    ])
    return EnvironGrid(at, nr)


@fixture
def uniform_density():
    """Create a uniform density on a given cell"""

    def _uniform_density(cell: EnvironGrid, u: float):
        data = np.ones(cell.nr) * u
        return EnvironDensity(cell, data=data, label='uniform')

    return _uniform_density
