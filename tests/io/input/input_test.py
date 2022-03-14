from pathlib import Path
from pytest import mark, raises

from envyron.io.input import Input


@mark.datadir('data')
def test_read(datadir: Path) -> None:
    """Test input file reading."""
    test = Input(natoms=1)

    # check default value
    assert test.environment.type == 'input'

    # read file
    test = Input(natoms=1, filename='water.yml')

    # verify parameter was read correctly
    assert test.environment.type == 'water'


@mark.parametrize('natoms', [-1, 0, 5])
def test_adjust_to_atoms(natoms: int) -> None:
    """Test scaling ion input arrays to size of number of atoms."""
    params = Input(1, **{'ions': {}})

    for array in (
            params.ions.atomicspread,
            params.ions.corespread,
            params.ions.solvationrad,
    ):
        assert len(array) == 1

    if natoms > 0:

        params.adjust_ionic_arrays(natoms=natoms)

        for array in (
                params.ions.atomicspread,
                params.ions.corespread,
                params.ions.solvationrad,
        ):
            assert len(array) == natoms

    else:

        with raises(ValueError):
            params.adjust_ionic_arrays(natoms=natoms)
