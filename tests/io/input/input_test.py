from pytest import mark
from pathlib import Path

from envyron.io.input import *


@mark.datadir('data')
def test_read(datadir: Path) -> None:
    """Test input file reading."""
    test = Input(natoms=1)

    # check default value
    assert 'type' not in test.param_dict['environment']

    # read file
    test.read('water.yml')

    # verify parameter was read correctly
    assert 'type' in test.param_dict['environment']
    assert test.param_dict['environment'].get('type') == 'water'
