import pytest
import numpy as np
from envyron.domains.cell import EnvironGrid
from envyron.physical.electrons import EnvironElectrons
from envyron.representations import EnvironDensity


grid = EnvironGrid(dimensions=(10, 10, 10), lattice_vectors=np.eye(3))

def test_EnvironElectrons_initialization():
    electrons = EnvironElectrons(grid)
    
    assert isinstance(electrons.density, EnvironDensity)

    assert electrons.updating is False

def test_EnvironElectrons_update():
    electrons = EnvironElectrons(grid)

    rho_3d = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
    nelec = 4  
    
    electrons.update(rho_3d, nelec=nelec)
   
    assert np.allclose(electrons.density[:], rho_3d)
    
  
    assert np.isclose(electrons.charge, rho_3d.sum())
    assert electrons.count == nelec

def test_EnvironElectrons_update_with_error():
    electrons = EnvironElectrons(grid)
    
    
    rho_3d = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
    wrong_nelec = 5  
    
    with pytest.raises(ValueError) as exc_info:
        electrons.update(rho_3d, nelec=wrong_nelec)
    
    error_message = str(exc_info.value)
    expected_error_message = "1.00e+00 error in integrated electronic charge"
    assert error_message == expected_error_message