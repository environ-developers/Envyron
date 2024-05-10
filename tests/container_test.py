import pytest
import numpy as np
from envyron.representations.functions.container import FunctionContainer
from envyron.representations.functions.gaussian import EnvironGaussian
from envyron.domains.cell import EnvironGrid


environ_grid = EnvironGrid(dimensions=(10, 10, 10), lattice_vectores=np.eye(3))

sample_gaussian = EnvironGaussian(environ_grid, pos=np.array([0.0, 0.0, 0.0]), spread=1.0, width=1.0)

def test_function_container_append():
    container = FunctionContainer(environ_grid)
    container.append(sample_gaussian)
    assert len(container) == 1
    assert container.functions[0] is sample_gaussian

def test_function_container_reset_derivatives():
    container = FunctionContainer(environ_grid)
    container.append(sample_gaussian)
    sample_gaussian.gradient = np.ones(environ_grid.shape)
    container.reset_derivatives()
    assert np.all(sample_gaussian.gradient.data == 0.0)


def test_function_container_density():
    container = FunctionContainer(environ_grid)
    container.append(sample_gaussian)
    density = container.density()
    assert np.all(density.data == sample_gaussian.density.data)

def test_fucntion_container_gradient():
    container = FunctionContainer(environ_grid)
    container.append(sample_gaussian)
    gradient = container.gradient()
    assert np.all(gradient.data == sample_gaussian.gradient.data)


def test_function_container_laplacian():
    container = FunctionContainer(environ_grid)
    container.append(sample_gaussian)
    laplacian = container.laplacian()
    assert np.all(laplacian.data == sample_gaussian.laplacian.data)

def test_function_container_hessian():
    container = FunctionContainer(environ_grid)
    container.append(sample_gaussian)
    hessian = container.hessian()
    assert np.all(hessian.data == sample_gaussian.hessian.data)

def test_function_container_derivative():
    container = FunctionContainer(environ_grid)
    container.append(sample_gaussian)
    derivative = container.derivative()
    assert np.all(derivative.data == sample_gaussian.derivative.data)



if __name__ == '__main__':
    pytest.main()






