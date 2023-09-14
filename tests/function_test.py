import pytest
import numpy as np
from envyron.representations.functions.function import EnvironFunction,  KINDS, FUNC_TOL
from envyron.domains.cell import EnvironGrid


environ_grid = EnvironGrid(dimensions=(10, 10, 10), lattice_vectors=np.eye(3))

def test_environ_function_init():
    kind = 1
    dim = 2
    axis = 0
    width = 1.0
    spread = 0.5
    volume = 10.0
    pos = np.array([0.0, 0.0, 0.0])
    label = 'test function'

    func = EnvironFunction(environ_grid, kind=kind, dim=dim, axis=axis, width=width, spread=spread, volume=volume, pos=pos, label=label)

    assert func.kind == kind
    assert func.dim == dim
    assert func.axis == axis
    assert func.width == width
    assert func.spread == spread
    assert func.volume == volume
    assert np.array_equal(func.pos, pos)
    assert func.label == label
    assert func.grid == environ_grid

    

