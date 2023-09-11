import pytest
import numpy as np
from numpy import ndarray
from envyron.domains.cell import EnvironGrid
from envyron.representations.density import EnvironDensity 

@pytest.fixture
def environ_grid() -> EnvironGrid:
    """Fixture for EnvironGrid."""
    return EnvironGrid(
        shape=(10, 10, 10),
        spacing=0.1,
        origin=(0.0, 0.0, 0.0),
    )

@pytest.fixture
def sample_data() -> ndarray:
    """Fixture for sample data."""
    return np.random.rand(10, 10, 10)

def test_environ_density_creation(environ_grid, sample_data):
    # Test if an EnvironDensity object can be created
    environ_density = EnvironDensity(environ_grid, data=sample_data, label='test label')
    assert isinstance(environ_density, EnvironDensity)

def test_environ_density_default_values(environ_grid):
    # Test if default values are correctly set when creating an EnvironDensity object
    environ_density = EnvironDensity(environ_grid)
    assert environ_density.dipole.tolist() == [0.0, 0.0, 0.0]
    assert environ_density.quadrupole.tolist() == [0.0, 0.0, 0.0]
    assert environ_density.label == ''
    assert environ_density._charge is None

def test_environ_densitu_data_attribute(environ_grid, sample_data):
    # Test if the 'data' attribute is correctly set when creating an EnvironDensity object
    environ_density = EnvironDensity(environ_grid, data=sample_data, label='test label')
    assert isinstance(environ_density.data, ndarray)
    assert np.array_equal(environ_density.data, sample_data)

def test_environ_density_label_attribute(environ_grid):
    # Test if the 'label' attribute is correctly set when creating an EnvironDensity object
    environ_density = EnvironDensity(environ_grid, label='test label')
    assert environ_density.label == 'test label'

def test_environ_density_charge_attribute(environ_grid, sample_data):
    # Test if the 'charge' attribute is correctly set when creating an EnvironDensity object
    environ_density = EnvironDensity(environ_grid, data=sample_data)
    assert environ_density.charge == environ_density.integral()


def test_charge_property_with_integral_mock(environ_grid, sample_data, mocker):
    environ_density = EnvironDensity(environ_grid)


    def mock_itegral():
        return 1.0 # A mock value for the integral method
    environ_density.integral = mock_itegral

    assert environ_density.charge == 1.0

def test_compute_multipoles(environ_grid):
    environ_density = EnvironDensity(environ_grid)
    origin = np.array([0.0, 0.0, 0.0])
    environ_density.compute_multipoles(origin)

     #Check the values of dipole and quadrupole attributes
    
    assert isinstance(environ_density.dipole, np.ndarray)  
    assert environ_density.dipole.shape == (3,) 
    assert np.array_equal(environ_density.dipole, np.zeros(3))  

    # Check quadrupole
    assert isinstance(environ_density.quadrupole, np.ndarray) 
    assert environ_density.quadrupole.shape == (3,)
    assert np.array_equal(environ_density.quadrupole, np.zeros(3))  

def test_euclidean_norm(environ_grid, sample_data):
    environ_density = EnvironDensity(environ_grid, data=sample_data)
    assert environ_density.euclidean_norm() == np.einsum('ijk,ijk', environ_density, environ_density)

def test_quadratic_mean(environ_grid, sample_data):
    environ_density = EnvironDensity(environ_grid, data=sample_data)
    assert environ_density.quadratic_mean() == np.sqrt(environ_density.euclidean_norm() / environ_density.grid.nnrR)

def test_scalar_product(environ_grid, sample_data):
    environ_density = EnvironDensity(environ_grid, data=sample_data)
    assert environ_density.scalar_product(environ_density) == np.einsum('ijk,ijk', environ_density, environ_density) * environ_density.grid.dV
