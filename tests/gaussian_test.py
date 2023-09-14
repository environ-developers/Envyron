import pytest
import numpy as np
from envyron.representations.functions import EnvironFunction
from envyron.representations.functions import EnvironGaussian
from envyron.domains.cell import EnvironGrid
from envyron.representations import EnvironGradient
from envyron.representations import EnvironDensity
from envyron.utils.constants import SQRTPI , EXP_TOL

@pytest.fixture
def environ_grid():
    return EnvironGrid 

def test_compute_density(environ_grid):
    sample_pos = np.array([0.0, 0.0, 0.0])
    sample_spread = 1.0
    environ_gaussian = EnvironGaussian(environ_grid, sample_pos, sample_spread)
    environ_gaussian._compute_density()


    _, r2 = environ_grid.get_min_distance(sample_pos, dim=3, axis=0)
    r2 /= sample_spread**2

    mask = r2 <= EXP_TOL

    r2 = r2[mask]

    expected_density_data = np.zeros(environ_gaussian.density.shape)
    expected_density_data[mask] = np.exp(-r2)

    scale = environ_gaussian._get_scale_factor()

    expected_density_data[mask] += expected_density_data[mask] * mask

    density = environ_gaussian.density.data

    expected_density = EnvironDensity(environ_grid, data=expected_density_data, label='test gaussian density')

    assert isinstance(density, EnvironDensity)
    assert density.grid == environ_grid
    assert np.array_equal(density.data, expected_density.data)


def test_compute_gradient(environ_grid):
    sample_pos = np.array([0.0, 0.0, 0.0])
    sample_spread = 1.0
    
    sample_volume = 10.0

    environ_gaussian = EnvironGaussian(environ_grid, sample_pos, sample_spread, volume=sample_volume)

    environ_gaussian._compute_gradient()

    spread2= sample_spread**2

    _, r2 = environ_grid.get_min_distance(sample_pos, dim=3, axis=0)

    r2 /= spread2
    
    mask = r2 <= EXP_TOL
    r = environ_grid.get_min_distance(sample_pos, dim=3, axis=0)[0]

    r = r[:, mask]

    r2 = r2[mask]

    expected_gradient = EnvironGradient(environ_grid, label='test gaussian gradient')

    gradient = np.zeros(expected_gradient.shape)

    gradient[: mask ]  = -np.exp(r2)*r

    scale = environ_gaussian._get_scale_factor() *2.0 * spread2

    expected_gradient[:, mask] +=gradient[:, mask] * scale

    assert isinstance(environ_gaussian.gradient, EnvironGradient)
    assert environ_gaussian.gradient.grid == environ_grid
    assert np.array_equal(environ_gaussian.gradient.data, expected_gradient.data)


def test_get_scale_factor_dim0(environ_grid):
    environ_gaussian = EnvironGaussian(environ_grid, dim=0, charge=1.0, spread=1.0)
    scale_factor = environ_gaussian._get_scale_factor()
    assert scale_factor == 1.0/(SQRTPI * 0.5)**3

def test_get_scale_factor_dim1(environ_grid):
    environ_gaussian = EnvironGaussian(environ_grid, dim=1, charge=2.0, spread=0.5)
    environ_gaussian.grid.lattice = lambda axis1 , axis2 : 2.0
    scale_factor = environ_gaussian._get_scale_factor()
    assert scale_factor == 2.0/(2.0* SQRTPI * 1.0)**2

def test_get_scale_factor_dim2(environ_grid):
    environ_gaussian = EnvironGaussian(environ_grid, dim=2, charge=3.0, spread=0.25)
    environ_gaussian.grid.lattice = lambda axis1 , axis2 : 3.0
    environ_gaussian.grid.volume = lambda : 4.0
    scale_factor = environ_gaussian._get_scale_factor() 

    assert scale_factor == 3.0/(3.0* 4.0 * SQRTPI * 2.0)

def test_get_scale_factor_out_of_range_dim(environ_grid):
    environ_gaussian = EnvironGaussian(environ_grid, dim=3, charge=1.0, spread=1.0)
    with pytest.raises(ValueError, match="dimensions out of range"):
        environ_gaussian._get_scale_factor()
    

if __name__ == '__main__':
    pytest.main()




















    


