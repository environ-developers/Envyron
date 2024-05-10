import pytest
import numpy as np
from numpy import ndarray
from typing import Optional
from envyron.domains.cell import EnvironGrid
from envyron.representations.hessian import EnvironHessian
from envyron.representations import EnvironField, EnvironDensity, EnvironGradient



@pytest.fixture
def envrion_grid():
    return EnvironGrid


def test_environ_hessian_creation(environ_grid):
    sample_data = np.zeros((9,9,9))
    environ_hessian = EnvironHessian(environ_grid, data=sample_data, label='test hessian')

    assert isinstance(environ_hessian, EnvironHessian)
    assert environ_hessian.grid == environ_grid
    assert np.array_equal(environ_hessian.data, sample_data)
    assert environ_hessian.label == 'test hessian'

def test_trace_property(environ_grid):
    sample_data = np.array([[[1,0,0], [0,2,0], [0,0,3]], [[4,0,0], [0,5,0], [0,0,6]], [[7,0,0], [0,8,0], [0,0,9]]])
    
    environ_hessian = EnvironHessian(environ_grid, data=sample_data, label='test hessian')
    trace = environ_hessian.trace

    expected_tace = sample_data[0] + sample_data[4] + sample_data[8]

    assert trace == expected_tace
    assert np.array_equal(trace.data, expected_tace)

def test_scalar_gradient_product(environ_grid):
    sample_data = np.array([[[1,0,0], [0,2,0], [0,0,3]], [[4,0,0], [0,5,0], [0,0,6]], [[7,0,0], [0,8,0], [0,0,9]]])
    environ_hessian = EnvironHessian(environ_grid, data=sample_data, label='test hessian')
    sample_gradient_data = np.ones((3,3,3))

    environ_gradient = EnvironGradient(environ_grid, data=sample_gradient_data)

    result = environ_hessian.scalar_gradient_product(environ_gradient)

    reshaped_hessian = sample_data.reshape(3, 3, *environ_grid.nr)
    expected_data = np.einsum('ml...,l...->m...', reshaped_hessian, sample_gradient_data)
    expected_result = EnvironGradient(environ_grid, data=expected_data)

    assert result == expected_result
    assert np.array_equal(result.data, expected_data)


if __name__ == "__main__":
    pytest.mai()









