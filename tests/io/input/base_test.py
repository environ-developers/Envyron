from pytest import mark, raises

from envyron.io.input.base import *


@mark.parametrize('natoms', [-1, 0, 5])
def test_adjust_to_atoms(natoms: int) -> None:
    """Test scaling ion input arrays to size of number of atoms."""
    params = InputModel(**{'ions': {}})

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
