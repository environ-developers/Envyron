import pytest
import numpy as np
from envyron.physical.electrolyte import EnvironElectrolyteBase, EnvironElectrolyte, EnvironIonccType
from envyron.domains.cell import EnvironGrid
from envyron.boundaries import EnvironBoundary

# Define test data
def generate_test_data():
    grid = EnvironGrid(dimensions=(10, 10, 10), lattice_vectors=np.eye(3))
    boundary = EnvironBoundary(grid)

    test_data = [
        {
        'temperature': 298.15,
            'permittivity': 80.4,
            'distance': 0.5,
            'spread': 0.05,
            'linearized': True,
            'entropy': 'entropy_value',
            'ntyp': 2,
            'cbulk': 0.1,
            'formula': [2, 1, 3, -1],  # Example: [ci1, zi1, ci2, zi2]
            'grid': grid,
            'cionmax': 0.5,
            'rion': 0.0,
            'boundary': boundary
        },
    ]
    return test_data

@pytest.mark.parametrize(
    "test_input", 
    generate_test_data()

)
def test_EnvironIonccType(test_input):
    ioncc_type = EnvironIonccType(**test_input)
    assert ioncc_type.temperature == test_input['temperature']
    assert ioncc_type.permittivity == test_input['permittivity']
    assert ioncc_type.distance == test_input['distance']
    assert ioncc_type.spread == test_input['spread']
    assert ioncc_type.linearized == test_input['linearized']
    assert ioncc_type.entropy == test_input['entropy']
    assert ioncc_type.ntyp == test_input['ntyp']
    assert ioncc_type.cbulk == test_input['cbulk']
    assert ioncc_type.k2 == pytest.approx(
        sum(ci * zi ** 2 for ci, zi in zip(test_input['formula'][::2], test_input['formula'][1::2]))* (3.0/ 4.0)
    )

    assert ioncc_type.cionmax == test_input['cionmax'] * (1.0 / 4.0)



@pytest.mark.parametrize(
    "test_input",
    generate_test_data()
)
def test_EnvironElectrolyteBase(test_input):
    electrolyte_base = EnvironElectrolyteBase(**test_input)
    assert electrolyte_base.temperature == test_input['temperature']
    assert electrolyte_base.permittivity == test_input['permittivity']
    assert electrolyte_base.distance == test_input['distance']
    assert electrolyte_base.spread == test_input['spread']
    assert electrolyte_base.linearized == test_input['linearized']
    assert electrolyte_base.entropy == test_input['entropy']
    assert electrolyte_base.ntyp == test_input['ntyp']
    assert electrolyte_base.cbulk == test_input['cbulk']
    assert electrolyte_base.k2 == pytest.approx(
        sum(ci * zi ** 2 for ci, zi in zip(test_input['formula'][::2], test_input['formula'][1::2]))* (3.0/ 4.0)
    )

    assert electrolyte_base.cionmax == test_input['cionmax'] * (1.0 / 4.0)
    assert electrolyte_base.rion == test_input['rion']
    assert electrolyte_base.boundary == test_input['boundary']

@pytest.mark.parametrize(
    "test_input",
    generate_test_data()
)
def test_EnvironElectrolyte(test_input):
    electrolyte = EnvironElectrolyte(**test_input)
    assert electrolyte.temperature == test_input['temperature']
    assert electrolyte.permittivity == test_input['permittivity']
    assert electrolyte.distance == test_input['distance']
    assert electrolyte.spread == test_input['spread']
    assert electrolyte.linearized == test_input['linearized']
    assert electrolyte.entropy == test_input['entropy']
    assert electrolyte.ntyp == test_input['ntyp']
    assert electrolyte.cbulk == test_input['cbulk']
    assert electrolyte.k2 == pytest.approx(
        sum(ci * zi ** 2 for ci, zi in zip(test_input['formula'][::2], test_input['formula'][1::2]))* (3.0/ 4.0)
    )

    assert electrolyte.cionmax == test_input['cionmax'] * (1.0 / 4.0)
    assert electrolyte.rion == test_input['rion']
    assert electrolyte.boundary == test_input['boundary']
    assert electrolyte.ioncctype == [EnvironIonccType(**test_input)]



