import pytest
import numpy as np
from envyron.domains.cell import EnvironGrid
from envyron.representations import EnvironDensity
from envyron.physical import (
    EnvironCharges,
    EnvironDielectric,
    EnvironElectrolyte,
    EnvironElectrons,
    EnvironExternals,
    EnvironIons,
    EnvironSemiconductor,
)

environ_grid = EnvironGrid(dimensions=(10, 10, 10), lattice_vectors=np.eye(3))

def test_environ_charges_init():
    charges = EnvironCharges(environ_grid)

    assert isinstance(charges.density, EnvironDensity)


def test_environ_charges_add():
    charges = EnvironCharges(environ_grid)

    ions = EnvironIons(environ_grid)
    electrons = EnvironElectrons(environ_grid)
    externals = EnvironExternals(environ_grid)
    dielectric = EnvironDielectric(environ_grid)
    electrolyte = EnvironElectrolyte(environ_grid)
    semiconductor = EnvironSemiconductor(environ_grid)
    additional = EnvironDensity(environ_grid)

    charges.add(
        electrons=electrons,
        ions=ions,
        externals=externals,
        dielectric=dielectric,
        electrolyte=electrolyte,
        semiconductor=semiconductor,
        additional=additional,
    )
    assert charges.electrons == electrons
    assert charges.ions == ions
    assert charges.externals == externals
    assert charges.dielectric == dielectric
    assert charges.electrolyte == electrolyte
    assert charges.semiconductor == semiconductor
    assert charges.additional == additional

def test_environ_charges_update():
    charges = EnvironCharges(environ_grid)

    with pytest.raises(NotImplementedError):
        charges.update()

def test_environ_charges_of_potential():
    charges = EnvironCharges(environ_grid)

    potential = EnvironDensity(environ_grid)

    result = charges.of_potential(potential)
    assert isinstance(result, EnvironDensity)

    with pytest.raises(NotImplementedError):
        charges.of_potential(potential)

