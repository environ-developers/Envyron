import pytest
import numpy as np
from envyron.representations import EnvironDensity
from envyron.domains.cell import EnvironGrid
from envyron.physical import EnvironExternals

grid = EnvironGrid(dimensions=(10, 10, 10), lattice_vectors=np.eye(3))

def test_init_empty():
    externals = EnvironExternals(0, [], [], [], [], grid)

    assert externals.number == 0
    assert externals.charge == 0.0
    assert externals.density == EnvironDensity(grid, label='externals')
    assert externals.functions == None
    assert len(externals.density) == 0.0

def test_init_with_functions():
    n = 2
    dims = [1,2, 3]
    axes = [0, 1, 2]
    spreads = [0.1, 0.2]
    charges = [0.5, 0.6]
    positions = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])]

    externals = EnvironExternals(n, dims, axes, spreads, charges, grid, positions)

    assert externals.number == n

    density_array = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])

    for function in externals.functions:
        function.set(density_array)

    externals.update()

    assert np.array_equal(externals.density, density_array)
    assert np.allclose(externals.density[:], density_array)
    assert np.isclose(externals.charge, density_array.sum())

    


