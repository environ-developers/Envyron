from types import LambdaType
from typing import Any, Sequence, Tuple
from pytest import mark, raises
from pathlib import Path
from envyron.io.input import *

ENTRY = {
    'section': 'section',
    'name': 'name',
    'dtype': 'type',
}

SECTIONS = (
    'Environ',
    'Boundary',
    'Electrostatic',
    'Externals',
    'Regions',
)


@mark.parametrize(
    'value, result',
    [
        ('true', True),
        ('false', False),
    ],
)
def test_entry_boolean(value: str, result: bool) -> None:
    """Test Entry class boolean converter."""
    entry = Entry(**ENTRY)

    assert entry._boolean(value) == result
    assert entry._boolean(value.capitalize()) == result
    assert entry._boolean(value.upper()) == result

    with raises(TypeError):
        entry._boolean('')
        entry._boolean('blah')


def test_entry_set_validator() -> None:
    """Test Entry class validator setting."""
    entry = Entry(**ENTRY)

    entry._set_validator('x != 0')

    assert isinstance(entry.valid, LambdaType)

    assert entry.valid(1)
    assert not entry.valid(0)


@mark.parametrize(
    'condition, valid, invalid',
    [
        (
            "0 <= x < 3",
            (0, 1, 2),
            (-1, 3),
        ),
        (
            "x != 0",
            (-1, 1),
            (0, ),
        ),
        (
            "any(x == s for s in ('this', 'that'))",
            ('this', 'that'),
            ('these', 'those'),
        ),
    ],
)
def test_entry_validate(condition: str, valid: Tuple, invalid: Tuple) -> None:
    """Test Entry class validator."""
    entry = Entry(**ENTRY)

    entry._set_validator(condition)

    for value in valid:
        assert entry._validate(value)

    with raises(ValueError):
        for value in invalid:
            entry._validate(value)


def test_array_entry_validate() -> None:
    """Test ArrayEntry class validator."""
    entry = ArrayEntry(**ENTRY, size=3)

    entry._set_validator("x > 0")

    valid = (1, 2, 3)
    invalid = (-2, -1, 0)

    assert entry._validate(valid)

    with raises(ValueError):
        entry._validate(invalid)


@mark.parametrize(
    'dtype, value, result, bad_value',
    [
        ('str', 'some_value', 'some_value', ''),
        ('int', '4', 4, 'hello'),
        ('float', '4.0', 4.0, 'hello'),
        ('bool', 'true', True, '5'),
    ],
)
def test_entry_convert(
    dtype: str,
    value: Any,
    result: Any,
    bad_value: Any,
) -> None:
    """Test Entry class data type converter."""
    entry = Entry(**ENTRY)

    entry.dtype = dtype
    converted = entry._convert(value)
    assert converted == result

    if bad_value:
        with raises(TypeError):
            converted = entry._convert(bad_value)


@mark.parametrize(
    'dtype, values, result',
    [
        ('str', '1 1 1', ('1', '1', '1')),
        ('int', '4 4 4', (4, 4, 4)),
        ('float', '4.0', (4.0, 4.0, 4.0)),
        ('bool', True, (True, True, True))
    ],
)
def test_array_entry_convert(
    dtype: str,
    values: Sequence[Any],
    result: Sequence[Any],
) -> None:
    """Test ArrayEntry data type converter."""
    entry = ArrayEntry(**ENTRY, size=3)

    entry.dtype = dtype
    assert entry._convert(values) == result


def test_reading_parameters() -> None:
    "Test reading parameters file."
    test = Input()

    assert test.params is not None, "failed to read parameters"

    for section in SECTIONS:
        assert section in test.params, f"missing {section} section"

        # check that all parameters have a type attribute
        for option, attrs in test.params[section].items():
            assert 'dtype' in attrs, f"missing 'dtype' attribute for '{option}'"


def test_reading_defaults() -> None:
    "Test reading and processing defaults file."
    test = Input()

    assert test.defaults is not None, "failed to read defaults"
    assert test.entries is not None, "failed to generate entries dictionary"

    for entry in test.entries.values():
        assert entry.value is not None, f"missing value for '{entry.name}'"

        # check array input format and conversion
        if isinstance(entry, ArrayEntry):

            assert isinstance(entry.value, tuple), \
                f"'{entry.name}' array not properly converted"

            assert len(entry.value) == entry.size, \
                f"wrong array size for '{entry.name}'"

            assert all(isinstance(i, eval(entry.dtype)) for i in entry.value), \
                f"'{entry.name}' array items not properly converted"


def test_conversion_to_dictionary() -> None:
    """Test conversion of input parameters to dictionary."""
    test = Input()

    params = test.to_dict()
    assert isinstance(params, dict), "failed to convert input to dictionary"

    for option in test.entries:
        assert option in params, f"missing '{option}' parameter"


@mark.datadir('data')
def test_reading_user_input_file(datadir: Path) -> None:
    """Test reading from user input file."""
    test = Input()

    test.read()  # environ.ini by default
    section = 'Environ'

    assert section in test.parser.sections(), \
        f"failed to read {section} section"

    assert test.parser.get(section, 'verbosity') == '1', \
        f"failed to read 'verbosity' option"

    del test.parser[section]

    test.read('custom.ini')  # custom input file name

    assert section in test.parser.sections(), \
        f"failed to read {section} section"

    assert test.parser.get(section, 'verbosity') == '3', \
        f"failed to read 'verbosity' option"


@mark.datadir('data')
def test_processing_user_input(datadir: Path) -> None:
    """Test processing user input."""
    test = Input()

    test.read()
    params = test.to_dict()

    assert isinstance(params['restart'], bool), f"wrong type for 'restart'"
    assert isinstance(params['nskip'], int), f"wrong type for 'nskip'"
    assert isinstance(params['pressure'], float), f"wrong type for 'pressure'"
    assert isinstance(params['env_type'], str), f"wrong type for 'env_type'"


@mark.datadir('data')
def test_processing_user_array_input(datadir: Path) -> None:
    """Test processing user array input."""
    natoms = 5
    test = Input(natoms)

    test.read('spreads.ini')
    params = test.to_dict()

    # check array input
    value = params['atomicspread']
    dtype = eval(test.entries['atomicspread'].dtype)

    assert isinstance(value, tuple), "array not properly converted"
    assert len(value) == natoms, "wrong array size"
    assert all(isinstance(i, dtype) for i in value), "wrong data type"

    # check size extrapolation
    value = params['corespread']
    assert len(value) == natoms, "wrong array size"


@mark.datadir('data')
def test_processing_externals(datadir: Path) -> None:
    """Test processing externals section."""
    test = Input()

    test.read('externals.ini')
    params = test.to_dict()

    externals = params['externals']
    assert len(externals) == 2, "wrong number of external groups"

    group = externals[0]
    assert len(group) == 3, "wrong number of external charges in group 1"

    external: ExternalCard = group[0]
    assert external.units == 'bohr', "wrong units for external charges"
    assert external.charge == 2.0, "wrong charge for external charge 1"
    assert len(external.pos) == 3, "external position not properly converted"

    count = 0
    for group in externals:
        for external in group:
            count += 1

    assert count == 4, "wrong number of external charges"


@mark.datadir('data')
def test_processing_regions(datadir: Path) -> None:
    """Test processing regions section."""
    test = Input()

    test.read('regions.ini')
    params = test.to_dict()

    regions = params['regions']
    assert len(regions) == 2, "wrong number of region groups"

    group = regions[0]
    assert len(group) == 3, "wrong number of regions in group 1"

    region: RegionCard = group[0]
    assert region.units == 'bohr', "wrong units for external charges"
    assert region.static == 100.0, "wrong static permittivity for region 1"
    assert len(region.pos) == 3, "region position not properly converted"

    count = 0
    for group in regions:
        for region in group:
            count += 1

    assert count == 4, "wrong number of regions"


@mark.datadir('data')
def test_simultaneous_cionmax_rion_setting(datadir: Path) -> None:
    """Test that cionmax and rion are not both set in input."""
    test = Input()

    try:
        test.read('cionmax_rion.ini')
    except ValueError as err:
        assert str(err) == "Cannot set both cionmax and rion"


@mark.datadir('data')
def test_smart_environment(datadir: Path) -> None:
    """Test smart environment settings."""
    test = Input()

    test.read('environment.ini')
    params = test.to_dict()

    # check for water environment with ionic interface
    assert params['static_permittivity'] == 78.3
    assert params['optical_permittivity'] == 1.776
    assert params['surface_tension'] == 50.0
    assert params['pressure'] == -0.35
    assert params['softness'] == 0.5
    assert params['radius_mode'] == 'uff'
    assert params['alpha'] == 1.12

    assert params['deriv_method'] == 'lowmem'

    assert params['problem'] == 'generalized'
    assert params['solver'] == 'cg'


@mark.datadir('data')
def test_smart_electrolyte(datadir: Path) -> None:
    """Test smart electrostatics settings w.r.t electrolyte."""
    test = Input()

    test.read('electrolyte.ini')
    params = test.to_dict()

    assert params['electrolyte_mode'] == 'system'
    assert params['electrolyte_deriv_method'] == 'chain'
    assert params['problem'] == 'pb'
    assert params['solver'] == 'fixed-point'
    assert params['auxiliary'] == 'full'
