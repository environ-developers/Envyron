import pytest
from envyron.physical.iontype import EnvironIonType

# Define test data
test_data = [

    (1, "H", 1.0, "pauling", 0.5, 0.5, 1.0),
    (2, 4.002602, 2.0, "bondi", 1.0, 1.0, 2.0),
    (3, "Li", 3.0, "uff", 0.8, 0.7, 1.5)

]

@pytest.mark.parametrize("index, ion_id, zv, radius_mode, atomicspread, corespread, solvationrad", test_data)
def test_initialization(index, ion_id, zv, radius_mode, atomicspread, corespread, solvationrad):
    iontype = EnvironIonType(index, ion_id, zv, radius_mode, atomicspread, corespread, solvationrad)

    assert iontype.index == index
    assert iontype.zv == -zv
    assert iontype.label in  EnvironIonType.elements
    assert iontype.number == EnvironIonType.elements.index(iontype.label) + 1
    assert iontype.weight == EnvironIonType.weights[iontype.number - 1]
    assert iontype.atomicspread == atomicspread
    assert iontype.corespread == corespread
    assert iontype.solvationrad == pytest.approx(solvationrad)


    if iontype.label == 'H':
        assert iontype.corespread == pytest.approx(1e-10)
    else:
        assert iontype.corespread == pytest.approx(corespread)

    if radius_mode == 'pauling':
        assert iontype.solvationrad == pytest.approx(EnvironIonType.pauling[index - 1])
    elif radius_mode == 'bondi':
        assert iontype.solvationrad == pytest.approx(EnvironIonType.bondi[index - 1])
    elif radius_mode == 'uff':
        assert iontype.solvationrad == pytest.approx(EnvironIonType.uff[index - 1])
    elif radius_mode == 'muff':
        assert iontype.solvationrad == pytest.approx(EnvironIonType.muff[index - 1])

def test_ion_id_by_label():
    ion = EnvironIonType(1, 'H', 1.0, 'pauling', 0.1, 0.2, 0.3)
    assert ion.label == 'H'
    assert ion.number == 1

def test_ion_id_by_index():
    ion = EnvironIonType(1, 1, 1.0, 'pauling', 0.1, 0.2, 0.3)
    assert ion.label == 'H'
    assert ion.number == 1

def test_ion_id_by_weight():
    ion = EnvironIonType(1, 1.00794, 1.0, 'pauling', 0.1, 0.2, 0.3)
    assert ion.label == 'H'
    assert ion.number == 1

def test_invalid_ion_id():
    try:
        ion = EnvironIonType(1, 'X', 1.0, 'pauling', 0.1, 0.2, 0.3)
    except TypeError as e:
        assert str(e) == "ion id must be given as number, label, or weight"

def test_solvation_radius_pauling():
    ion = EnvironIonType(1, 'H', 1.0, 'pauling', 0.1, 0.2, 0.3)
    assert ion.solvationrad == 1.2
                                                 

