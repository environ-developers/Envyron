import pytest
import numpy as np
from envyron.representations.functions import EnvironFunction,  KINDS, FUNC_TOL
from envyron.domains.cell import EnvironGrid
from envyron.representations import EnvironDensity
from envyron.representations import EnvironGradient
from envyron.representations import EnvironHessian


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

def test_environ_function_property_kind():
    func = EnvironFunction(environ_grid, kind=1, dim=2, axis=0, width=1.0, spread=0.5, volume=10.0, pos=np.array([0.0, 0.0, 0.0]), label='test function')

    assert func.kind == 1

    with pytest.raises(ValueError):
        func.kind = 0

def test_environ_function_property_dim():
    func = EnvironFunction(environ_grid, kind=1, dim=2, axis=0, width=1.0, spread=0.5, volume=10.0, pos=np.array([0.0, 0.0, 0.0]), label='test function')
    assert func.dim == 2.0

    with pytest.raises(ValueError):
        func.dim = -1


def test_environ_function_proporty_axis():
    func = EnvironFunction(environ_grid, kind=1, dim=2, axis=0, width=1.0, spread=0.5, volume=10.0, pos=np.array([0.0, 0.0, 0.0]), label='test function')
    assert func.axis == 0

    with pytest.raises(ValueError):
        func.axis = -1

def test_environ_function_property_spread():
    func = EnvironFunction(environ_grid, kind=1, dim=2, axis=0, width=1.0, spread=0.5, volume=10.0, pos=np.array([0.0, 0.0, 0.0]), label='test function')
    assert func.spread == 0.5

    with pytest.raises(ValueError):
        func.spread = -1.0
        
def test_environ_function_property_density():

    func = EnvironFunction(environ_grid, kind=1, dim=2, axis=0, width=1.0, spread=0.5, volume=10.0, pos=np.array([0.0, 0.0, 0.0]), label='test function')
    density = func.density
    assert isinstance(density, EnvironDensity)


def test_environ_function_property_gradient():
    func = EnvironFunction(environ_grid, kind=1, dim=2, axis=0, width=1.0, spread=0.5, volume=10.0, pos=np.array([0.0, 0.0, 0.0]), label='test function')
    gradient = func.gradient
    assert isinstance(gradient, EnvironGradient)

def test_environ_function_property_laplacian():
    func = EnvironFunction(environ_grid, kind=1, dim=2, axis=0, width=1.0, spread=0.5, volume=10.0, pos=np.array([0.0, 0.0, 0.0]), label='test function')
    laplacian = func.laplacian
    assert isinstance(laplacian, EnvironDensity)

def test_environ_function_property_hessian():
    func = EnvironFunction(environ_grid, kind=1, dim=2, axis=0, width=1.0, spread=0.5, volume=10.0, pos=np.array([0.0, 0.0, 0.0]), label='test function')
    hessian = func.hessian
    assert isinstance(hessian, EnvironHessian)

def test_environ_function_property_derivative():
    func = EnvironFunction(environ_grid, kind=1, dim=2, axis=0, width=1.0, spread=0.5, volume=10.0, pos=np.array([0.0, 0.0, 0.0]), label='test function')
    derivative = func.derivative
    assert isinstance(derivative, EnvironDensity)


def test_environ_funcvtion_reset_derivative():

    func = EnvironFunction(environ_grid, kind=1, dim=2, axis=0, width=1.0, spread=0.5, volume=10.0)
    func.density[:] = 1.0
    func.gradient[:] = 1.0
    func.laplacian[:] = 1.0
    func.hessian[:] = 1.0
    func.derivative[:] = 1.0

    func.reset_derivative()

    assert np.all(func.density.data == 0.0)
    assert np.all(func.gradient.data == 0.0)
    assert np.all(func.laplacian.data == 0.0)
    assert np.all(func.hessian.data == 0.0)
    assert np.all(func.derivative.data == 0.0)



if __name__ == '__main__':
    pytest.main()







    




