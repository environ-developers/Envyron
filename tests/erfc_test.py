import pytest
import numpy as np
import scipy.special as sp
from numpy import ndarray
from envyron.domains.cell import EnvironGrid
from envyron.representations.functions import EnvironFunction, FUNC_TOL
from envyron.representations.functions import EnvironERFC
from envyron.representations.density import EnvironDensity
from envyron.representations.gradient import EnvironGradient
from envyron.utils.constants import SQRTPI

class ERFCGradientWarning(Exception):
     pass
class ERFCGHessianWarning(Exception):
     pass
class ERFCDericativeWarning(Exception):
     pass

class UnexpectedFunctionType(Exception):
     pass


@pytest.fixture
def environ_erfc(environ_grid):
    sample_pos = np.array([0.0, 0.0, 0.0])
    sample_spread = 1.0
    sample_width = 0.5
    sample_volume = 10.0

    environ_erfc = EnvironERFC(
        environ_grid,
        kind=2,
        dim=3,
        axis=0,
        width=sample_width,
        spread=sample_spread,
        volume=sample_volume,
        pos=sample_pos,
        label='test erfc gradient'
    )
    return environ_erfc

def test_compute_density(environ_grid):
    environ_erfc._compute_density()

    _, r2 = environ_grid.get_min_distance(environ_erfc.pos, environ_erfc.dim, environ_erfc.axis)
    dist = np.sqrt(r2)
    arg = (dist - environ_erfc.width) / environ_erfc.spread

    mask = dist > FUNC_TOL

    r = r[:, mask]
    dist = dist[mask]
    arg = arg[mask]

    expected_density_data = np.zeros(environ_erfc.density.shape)
    expected_density_data[mask] = sp.erfc(arg)

    scale = environ_erfc._get_scale_factor()

    expected_density_data[mask] += expected_density_data[mask] * scale

    density = environ_erfc.density.data

    expected_density = EnvironDensity(environ_grid, data=expected_density_data, label='test erfc density')

    assert isinstance(density, EnvironDensity)
    assert density.grid == environ_grid

    assert np.array_equal(density.data, expected_density.data)


def test_compute_gradient(environ_grid):

    environ_erfc._compute_gradient()

    _, r2 = environ_grid.get_min_distance(environ_erfc.pos, environ_erfc.dim, environ_erfc.axis)
    dist = np.sqrt(r2)
    arg = (dist - environ_erfc.width) / environ_erfc.spread

    mask = dist > FUNC_TOL

    r = r[:, mask]

    r2 = r2[mask]

    expected_gradient = EnvironGradient(environ_grid, label='test erfc gradient')

    gradient = np.zeros(expected_gradient.shape)

    gradient[:, mask] = -np.exp(-arg**2) * r / dist

    charge = environ_erfc._charge()
    analytic = environ_erfc._erfc_volume()
    scale = charge / analytic / SQRTPI / environ_erfc.spread

    expected_gradient[:, mask] += gradient[:, mask] * scale

    with pytest.warns(ERFCGradientWarning):
        environ_erfc._compute_gradient()

    assert isinstance(environ_erfc.gradient, EnvironGradient)
    assert environ_erfc.gradient.grid == environ_grid
    assert np.array_equal(environ_erfc.gradient.data, expected_gradient.data)


def test_compute_laplacian(environ_grid):
    environ_erfc._compute_laplacian()

    _, r2 = environ_grid.get_min_distance(environ_erfc.pos, environ_erfc.dim, environ_erfc.axis)
    dist = np.sqrt(r2)
    arg = (dist - environ_erfc.width) / environ_erfc.spread

    mask = dist > FUNC_TOL

    dist = dist[mask]
    arg = arg[mask]

    expected_laplacian = EnvironDensity(environ_grid, label='test erfc laplacian')

    laplacian = np.zeros(expected_laplacian.shape)

    exp = np.exp(-arg**2)

    if environ_erfc.dim == 0:

        laplacian[mask] = -exp * (1 / dist - arg / environ_erfc.spread) * 2

    elif environ_erfc.dim == 1:

        laplacian[mask] = -exp * (1 / dist - 2 * arg / environ_erfc.spread)

    elif environ_erfc.dim == 2:

        laplacian[mask] = exp * arg / environ_erfc.spread * 2

    else:
            
            raise ValueError("unexpected system dimensions")
    
    charge = environ_erfc._charge()

    analytic = environ_erfc._erfc_volume()

    scale = charge / analytic / SQRTPI / environ_erfc.spread

    expected_laplacian[mask] += laplacian[mask] * scale

    assert isinstance(environ_erfc.laplacian, EnvironDensity)

    assert environ_erfc.laplacian.grid == environ_grid

    assert np.array_equal(environ_erfc.laplacian.data, expected_laplacian.data)

def test_compute_hessian(environ_grid):
    environ_erfc._compute_hessian()

    _, r2 = environ_grid.get_min_distance(environ_erfc.pos, environ_erfc.dim, environ_erfc.axis)
    dist = np.sqrt(r2)
    arg = (dist - environ_erfc.width) / environ_erfc.spread
    mask =  dist > FUNC_TOL
    r = r[:, mask]
    dist  = dist[mask]
    arg = arg[mask]

    expected_hessian = EnvironGradient(environ_grid, label='test erfc hessian')
    
    hessian = np.zeros(expected_hessian.shape)

    shape = np.reshape(np.einsum('i..., j...->ij...', -r, r), (9, shape))
    outer *= 1 / dist + 2 * arg / environ_erfc.spread
    outer += dist * np.identity(3).flatten()[:, None]

    expected_hessian_data = -np.exp(-arg**2) * outer / dist**2

    charge = environ_erfc._charge()
    analytic = environ_erfc._erfc_volume()
    scale = charge / analytic / SQRTPI / environ_erfc.spread
    
    expected_hessian_data[:, mask] += expected_hessian_data[:, mask] * scale

    with pytest.warns(ERFCGHessianWarning):
        environ_erfc._compute_hessian()
        
    assert np.allclose(environ_erfc.hessian.data, expected_hessian_data, atol=1e-8, rtol=1e-8)

    



def test_compute_derivative(environ_grid):
    environ_erfc._compute_derivative()
     
    _, r2 = environ_grid.get_min_distance(environ_erfc.pos, environ_erfc.dim, environ_erfc.axis)
    dist = np.sqrt(r2)
    arg = (dist - environ_erfc.width) / environ_erfc.spread
    mask =  dist > FUNC_TOL 
    arg = arg[mask]

    expected_derivative = EnvironDensity(environ_grid, label='test erfc derivative')

    derivative = np.zeros(expected_derivative.shape)

    derivative[mask] = -np.exp(-arg**2)

    integral = np.sum(environ_erfc._derivative) * \
               environ_erfc.grid.volume / environ_erfc.grid.nnrR * 0.5

    charge = environ_erfc._charge()
    analytic = environ_erfc._erfc_volume()

    if np.abs((integral - analytic) / analytic > 1e-4):
        raise ERFCDericativeWarning("integral and analytic values are not close")
    
    scale = charge / analytic / SQRTPI / environ_erfc.spread

    derivative[mask] += derivative[mask] * scale

    with pytest.warns(ERFCDericativeWarning):
        environ_erfc._compute_derivative()

    assert isinstance(environ_erfc.derivative, EnvironDensity)
    assert environ_erfc.derivative.grid == environ_grid
    assert np.array_equal(environ_erfc.derivative.data, expected_derivative.data)


def test_charge(environ_grid):
    charge= environ_erfc._charge()

    if environ_erfc.kind == 1:
        with pytest.raises(ValueError):
            environ_erfc._charge()
    elif environ_erfc.kind == 2:
        expected_charge = environ_erfc.volume

        assert charge == expected_charge

    elif environ_erfc.kind == 3:

        expected_charge *= environ_erfc._erfc_volume()
        assert charge == expected_charge

    elif environ_erfc.kind == 4:
            
        expected_charge *= -environ_erfc._erfc_volume()
        assert charge == expected_charge

    else:
        with pytest.raises(UnexpectedFunctionType):
            environ_erfc._charge()


environ_grid = EnvironGrid(dimensions= (10,10,10), lattice_vectors= np.eye(3))

def test_erfc_volume_dim0():
    environ_erfc = EnvironERFC(environ_grid, kind= 2, dim=0, axis=0, width=1.0, spread=1.0, volume=1.0, pos= np.array([0.0, 0.0, 0.0]), label='test erfc volume')
    expected_volume = (4.0 / 3.0) * np.pi * 1.0**3
    assert environ_erfc._erfc_volume() == expected_volume

def test_erfc_volume_dim1():
    environ_erfc = EnvironERFC(environ_grid, kind= 2, dim=1, axis=0, width=1.0, spread=1.0, volume=1.0, pos= np.array([0.0, 0.0, 0.0]), label='test erfc volume')
    expected_volume = np.pi * 1.0**2 * 1.0
    assert environ_erfc._erfc_volume() == expected_volume

def test_erfc_volume_dim2():
    environ_erfc = EnvironERFC(environ_grid, kind= 2, dim=2, axis=0, width=1.0, spread=1.0, volume=1.0, pos= np.array([0.0, 0.0, 0.0]), label='test erfc volume')
    expected_volume = 2.0 * 1.0 * 1.0
    assert environ_erfc._erfc_volume() == expected_volume

def test_erfc_volume_invalid_parameters():
    environ_erfc = EnvironERFC(environ_grid, kind= 2, dim=3, axis=0, width=0.0, spread=0.0, volume=1.0)
    try:
        environ_erfc._erfc_volume()
        assert False
    except ValueError:
        assert True

def test_erfc_volume_invalid_dim():
    environ_erfc = EnvironERFC(environ_grid, kind= 2, dim=4, axis=0, width=1.0, spread=1.0, volume=1.0, pos= np.array([0.0, 0.0, 0.0]), label='test erfc volume')
    try:
        environ_erfc._erfc_volume()
        assert False
    except ValueError:
        assert True

        






    




















