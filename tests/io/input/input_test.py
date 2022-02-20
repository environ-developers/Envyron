from pytest import mark
from pathlib import Path
from envyron.io.input import *

ENTRY = {
    'section': 'Environ',
    'name': 'verbosity',
    'type': 'int',
    'condition': 'x >= 0',
    'description': "",
}

SECTIONS = (
    'Environ',
    'Boundary',
    'Electrostatic',
    'Externals',
    'Regions',
)


def test_reading_parameters() -> None:
    "Test reading parameters file."
    test = Input()

    assert test.params is not None, "failed to read parameters"

    for section in SECTIONS:
        assert section in test.params, f"missing {section} section"

        # check that all parameters have a type attribute
        for option, attrs in test.params[section].items():
            assert 'type' in attrs, f"missing 'type' attribute for '{option}'"


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

            assert all(isinstance(i, eval(entry.type)) for i in entry.value), \
                f"'{entry.name}' array items not properly converted"


def test_conversion_to_dictionary() -> None:
    """Test conversion of input parameters to dictionary."""
    test = Input()

    params = test.to_dict()
    assert isinstance(params, dict), "failed to convert input to dictionary"

    for option in test.entries:
        assert option in params, f"missing '{option}' parameter"


def test_conversion_to_class() -> None:
    """Test conversion of input parameters to dynamic class."""
    test = Input()

    params = test.get_parameters()
    assert isinstance(params, Params), "failed to convert input to class"

    for option in test.entries:
        assert option in params.__dict__, f"missing '{option}' parameter"


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
    params = test.get_parameters()

    assert isinstance(params.debug, bool), f"wrong type for 'debug'"
    assert isinstance(params.verbosity, int), f"wrong type for 'verbosity'"
    assert isinstance(params.threshold, float), f"wrong type for 'threshold'"
    assert isinstance(params.env_type, str), f"wrong type for 'env_type'"


@mark.datadir('data')
def test_processing_user_array_input(datadir: Path) -> None:
    """Test processing user array input."""
    natoms = 5
    test = Input(natoms)

    test.read('spreads.ini')
    params = test.get_parameters()

    # check array input
    val = params.atomicspread
    t = eval(test.entries['atomicspread'].type)

    assert isinstance(val, tuple), "array not properly converted"
    assert len(val) == natoms, "wrong array size"
    assert all(isinstance(i, t) for i in val), "wrong data type"

    # check size extrapolation
    val = params.corespread
    assert len(val) == natoms, "wrong array size"


@mark.datadir('data')
def test_processing_externals(datadir: Path) -> None:
    """Test processing externals section."""
    test = Input()

    test.read('externals.ini')
    params = test.get_parameters()

    externals = params.externals
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
    params = test.get_parameters()

    regions = params.regions
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
    params = test.get_parameters()

    # check for water environment with ionic interface
    assert params.static_permittivity == 78.3
    assert params.optical_permittivity == 1.776
    assert params.surface_tension == 50.0
    assert params.pressure == -0.35
    assert params.softness == 0.5
    assert params.radius_mode == 'uff'
    assert params.alpha == 1.12

    assert params.deriv_method == 'lowmem'

    assert params.problem == 'generalized'
    assert params.solver == 'cg'


@mark.datadir('data')
def test_smart_electrolyte(datadir: Path) -> None:
    """Test smart electrostatics settings w.r.t electrolyte."""
    test = Input()

    test.read('electrolyte.ini')
    params = test.get_parameters()

    assert params.electrolyte_mode == 'system'
    assert params.electrolyte_deriv_method == 'chain'
    assert params.problem == 'pb'
    assert params.solver == 'fixed-point'
    assert params.auxiliary == 'full'
