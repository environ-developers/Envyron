import pytest
import numpy as np
from numpy import ndarray
from multimethod import multimethod
from envyron.domains.cell import EnvironGrid
from envyron.representations import EnvironField
from envyron.representations import EnvironDensity
from envyron.representations import EnvironGradient

@pytest.fixture
def environ_grid():
    return EnvironGrid

def test_environ_gradient_creation(environ_grid):
    environ_gradient = EnvironGradient(environ_grid)

    assert isinstance(environ_gradient, EnvironGradient)

    assert environ_gradient.grid == environ_grid
    assert environ_gradient.data is None
    assert environ_gradient.label == ''
    assert environ_gradient._modulus is None

def test_modulus_property(environ_grid):
    environ_gradient = EnvironGradient(environ_grid)

    sample_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    environ_gradient.data = sample_data
    environ_grid.nnr = 2
    environ_gradient._compute_modulus()
    assert isinstance(environ_gradient.modulus, EnvironDensity)
    assert environ_gradient.modulus.grid == environ_grid
    assert environ_gradient.modulus.data.shape == (2, 2, 2)
    assert np.array_equal(environ_gradient.modulus.data, np.array([[[np.sqrt(5), np.sqrt(20)], [np.sqrt(45), np.sqrt(80)]], [[np.sqrt(125), np.sqrt(180)], [np.sqrt(245), np.sqrt(320)]]]))

def test_scalar_product_method_with_gradient(environ_grid):
    environ_gradient1 = EnvironGradient(environ_grid)
    environ_gradient2 = EnvironGradient(environ_grid)
    
    sample_data1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    sample_data2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    environ_gradient1.data = sample_data1
    environ_gradient2.data = sample_data2
    environ_grid.nnr = 2
    result = environ_gradient1.scalar_product(environ_gradient2)

    expected_data = np.einsum('l...,l...', sample_data1, sample_data2)
    assert isinstance(result, EnvironDensity)
    assert result.grid == environ_grid
    assert result.data.shape == (2, 2, 2)
    assert np.array_equal(result.data, expected_data)

def test_scalar_product_environ_density(environ_grid):
    environ_gradient =  EnvironGradient(environ_grid)

    data_self = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    data_density = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    environ_gradient.data = data_self

    density = EnvironDensity(environ_grid, data=data_density)

    result = environ_gradient.scalar_product(density)

    expected_data = np.einsum('lijk,ijk', data_self, data_density) * environ_grid.dV

    assert isinstance(result, ndarray)
    assert result.shape == (2, 2, 2)
    assert np.array_equal(result, expected_data)


if __name__ == '__main__':
    pytest.main()

    





