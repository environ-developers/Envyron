from configparser import ConfigParser
from pathlib import Path
from typing import List, Union
from typing_extensions import TypeAlias
from pydantic import BaseModel, validator, validate_model

IntFloat: TypeAlias = Union[int, float]


class ValidatorModel(BaseModel):
    """
    Genearl validation model for Environ input.
    """


class ControlModel(BaseModel):
    """
    Model for the Control input section.
    """
    debug = False
    restart = False
    verbosity = 0
    threshold = 0.1
    nskip = 1
    env_type = 'input'
    nrep: List[int] = [0, 0, 0]
    system_ntyp = 0
    system_dim = 0
    system_axis = 3
    system_pos: List[float] = [0.0, 0.0, 0.0]
    need_electrostatic = False
    atomicspread: List[float] = [0.5]
    corespread: List[float] = [0.5]
    static_permittivity = 1.0
    optical_permittivity = 1.0
    surface_tension = 0.0
    pressure = 0.0
    confine = 0.0
    electrolyte_concentration = 0.0
    electrolyte_formula: List[int] = [0]
    electrolyte_linearized = False
    electrolyte_entropy = 'full'
    cionmax = 0.0
    rion = 0.0
    temperature = 300.0
    sc_permittivity = 1.0
    sc_carrier_density = 0.0
    external_charges = 0
    dielectric_regions = 0

    @validator(
        'nrep',
        'system_pos',
        'atomicspread',
        'corespread',
        'electrolyte_formula',
        pre=True,
    )
    def split_string(cls, value: str) -> List[IntFloat]:
        """Preprocess string into a list of values."""
        if ' ' in value: return value.split()
        return [value]

    @validator(
        'verbosity',
        'threshold',
        'nskip',
        'system_ntyp',
        'surface_tension',
        'confine',
        'electrolyte_concentration',
        'cionmax',
        'rion',
        'temperature',
        'sc_carrier_density',
        'external_charges',
        'dielectric_regions',
    )
    def ge_zero(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than or equal to zero."""
        assert value >= 0, f"{value} must be greater than or equal to zero"
        return value

    @validator(
        'nrep',
        'atomicspread',
        'corespread',
        each_item=True,
    )
    def ge_zero_many(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than or equal to zero."""
        assert value >= 0, f"{value} must be greater than or equal to zero"
        return value

    @validator(
        'static_permittivity',
        'optical_permittivity',
    )
    def ge_one(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than or equal to one."""
        assert value >= 1, f"{value} must be greater than or equal to one"
        return value

    @validator('env_type')
    def valid_environment_type(cls, value: str) -> str:
        """Checks value against acceptable environment types."""
        acceptable = (
            'input',
            'vacuum',
            'water',
            'water-cation',
            'water-anion',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected environment type"
        return value

    @validator('system_dim')
    def valid_dimension(cls, value: int) -> int:
        """Checks that the dimensions are reasonable (0<=>3)."""
        assert 0 <= value <= 3, f"expected 0 <= dim <= 3, got {value}"
        return value

    @validator('system_axis')
    def valid_axis(cls, value: int) -> int:
        """Checks that the axis is reasonable (1<=>3)."""
        assert 1 <= value <= 3, f"axis out of range (1 <= a <= 3)"
        return value

    @validator('electrolyte_entropy')
    def valid_electrolyte_entropy(cls, value: str) -> str:
        """Checks value against acceptable electrolyte entropy schemes."""
        acceptable = (
            'ions',
            'full',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected electrolyte entropy scheme"
        return value


class BoundaryModel(BaseModel):
    """
    Model for the Boundary input section.
    """
    solvent_mode = 'electronic'
    radius_mode = 'uff'
    alpha = 1.0
    softness = 0.5
    solvationrad: List[float] = [0.0]
    stype = 2
    rhomax = 5e-3
    rhomin = 1e-4
    tbeta = 4.8
    solvent_distance = 1.0
    solvent_spread = 0.5
    solvent_radius = 0.0
    radial_scale = 2.0
    radial_spread = 0.5
    filling_threshold = 0.825
    filling_spread = 0.02
    sc_distance = 0.0
    sc_spread = 0.5
    electrolyte_mode = 'electronic'
    electrolyte_distance = 0.0
    electrolyte_spread = 0.5
    electrolyte_rhomax = 5e-3
    electrolyte_rhomin = 1e-4
    electrolyte_tbeta = 4.8
    electrolyte_alpha = 1.0
    electrolyte_softness = 0.5
    electrolyte_deriv_method = 'default'
    deriv_method = 'default'
    deriv_core = 'fft'

    @validator(
        'alpha',
        'softness',
        'solvent_distance',
        'solvent_spread',
        'radial_spread',
        'filling_threshold',
        'filling_spread',
        'sc_spread',
        'electrolyte_spread',
        'electrolyte_softness',
    )
    def gt_zero(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than zero."""
        assert value > 0, f"{value} must be greater than zero"
        return value

    @validator(
        'rhomax',
        'rhomin',
        'tbeta',
        'solvent_radius',
        'sc_distance',
        'electrolyte_distance',
        'electrolyte_rhomax',
        'electrolyte_rhomin',
        'electrolyte_tbeta',
        'electrolyte_alpha',
    )
    def ge_zero(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than or equal to zero."""
        assert value >= 0, f"{value} must be greater than or equal to zero"
        return value

    @validator(
        'solvationrad',
        pre=True,
    )
    def split_string(cls, value: str) -> List[IntFloat]:
        """Preprocess string into a list of values."""
        if ' ' in value: return value.split()
        return [value]

    @validator(
        'solvationrad',
        each_item=True,
    )
    def ge_zero_many(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than or equal to zero."""
        assert value >= 0, f"{value} must be greater than or equal to zero"
        return value

    @validator('radial_scale')
    def ge_one(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than or equal to one."""
        assert value >= 1, f"{value} must be greater than or equal to one"
        return value

    @validator('stype')
    def valid_switching_function_type(cls, value: int) -> int:
        """Checks that switching function type is valid."""
        assert 0 <= value <= 2, f"expected 0 <= stype <= 2, got {value}"
        return value

    @validator(
        'solvent_mode',
        'electrolyte_mode',
    )
    def valid_solvent_mode(cls, value: str) -> str:
        """Checks value against acceptable solvent/electrolyte modes."""
        acceptable = (
            'electronic',
            'ionic',
            'full',
            'system',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected solvent/electrolyte mode"
        return value

    @validator(
        'deriv_method',
        'electrolyte_deriv_method',
    )
    def valid_deriv_method(cls, value: str) -> str:
        """Checks value against acceptable derivative methods."""
        acceptable = (
            'default',
            'fft',
            'chain',
            'highmen',
            'lowmem',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected derivative method"
        return value

    @validator('radius_mode')
    def valid_radius_mode(cls, value: str) -> str:
        """Checks value against acceptable radius modes."""
        acceptable = (
            'pauling',
            'bondi',
            'uff',
            'muff',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected radius mode"
        return value

    @validator('deriv_core')
    def valid_derivatives_core(cls, value: str) -> str:
        """Checks value against acceptable derivatives cores."""
        acceptable = ('fft', )
        assert any(value == v for v in acceptable), \
            f"unexpected derivatives core"
        return value


class ElectrostaticModel(BaseModel):
    """
    Model for the Electrostatic input section.
    """
    problem = 'none'
    tol = 1e-5
    solver = 'none'
    auxiliary = 'none'
    step_type = 'optimal'
    step = 0.3
    maxstep = 200
    mix_type = 'linear'
    ndiis = 1
    mix = 0.5
    preconditioner = 'sqrt'
    screening_type = 'none'
    screening = 0.0
    core = 'fft'
    inner_solver = 'none'
    inner_core = 'fft'
    inner_tol = 1e-10
    inner_maxstep = 200
    inner_mix = 0.5
    pbc_correction = 'none'
    pbc_core = '1da'
    pbc_dim = 0
    pbc_axis = 3

    @validator(
        'tol',
        'step',
        'ndiis',
        'inner_tol',
        'inner_mix',
    )
    def gt_zero(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than zero."""
        assert value > 0, f"{value} must be greater than zero"
        return value

    @validator(
        'maxstep',
        'inner_maxstep',
    )
    def gt_one(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than one."""
        assert value > 1, f"{value} must be greater than one"
        return value

    @validator('screening')
    def ge_zero(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than or equal to zero."""
        assert value >= 0, f"{value} must be greater than or equal to zero"
        return value

    @validator('problem')
    def valid_electrostatic_problem(cls, value: str) -> str:
        """Checks value against acceptable electrostatic problems."""
        acceptable = (
            'none',
            'poisson',
            'generalized',
            'pb',
            'modpb',
            'linpb',
            'linmodpb',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected electrostatic problem"
        return value

    @validator('solver')
    def valid_electrostatic_solver(cls, value: str) -> str:
        """Checks value against acceptable electrostatic solvers."""
        acceptable = (
            'none',
            'cg',
            'sd',
            'fixed-point',
            'newton',
            'nested',
            'direct',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected electrostatic solver"
        return value

    @validator('auxiliary')
    def valid_auxiliary_scheme(cls, value: str) -> str:
        """Checks value against acceptable auxiliary schemes."""
        acceptable = (
            'none',
            'full',
            'ioncc',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected auxiliary scheme"
        return value

    @validator('step_type')
    def valid_step_type(cls, value: str) -> str:
        """Checks value against acceptable step types."""
        acceptable = (
            'optimal',
            'input',
            'random',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected step type"
        return value

    @validator('mix_type')
    def valid_mix_type(cls, value: str) -> str:
        """Checks value against acceptable mix types."""
        acceptable = (
            'linear',
            'anderson',
            'diis',
            'broyden',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected mix type"
        return value

    @validator('preconditioner')
    def valid_preconditioner(cls, value: str) -> str:
        """Checks value against acceptable preconditioners."""
        acceptable = (
            'sqrt',
            'left',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected preconditioner"
        return value

    @validator('screening_type')
    def valid_screening_type(cls, value: str) -> str:
        """Checks value against acceptable screening types."""
        acceptable = (
            'none',
            'input',
            'linear',
            'optimal',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected screening type"
        return value

    @validator('core', 'inner_core')
    def valid_electrostatic_core(cls, value: str) -> str:
        """Checks value against acceptable electrostatic cores."""
        acceptable = ('fft', )
        assert any(value == v for v in acceptable), \
            f"unexpected electrostatics core"
        return value

    @validator('inner_solver')
    def valid_electrostatic_inner_solver(cls, value: str) -> str:
        """Checks value against acceptable electrostatic inner solvers."""
        acceptable = (
            'none',
            'cg',
            'sd',
            'fixed-point',
            'newton',
            'direct',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected electrostatic inner solver"
        return value

    @validator('pbc_correction')
    def valid_pbc_correction(cls, value: str) -> str:
        """Checks value against acceptable pbc corrections."""
        acceptable = (
            'none',
            'parabolic',
            'gcs',
            'ms',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected pbc correction"
        return value

    @validator('pbc_core')
    def valid_pbc_core(cls, value: str) -> str:
        """Checks value against acceptable pbc correction cores."""
        acceptable = ('1da', )
        assert any(value == v for v in acceptable), \
            f"unexpected pbc core"
        return value

    @validator('pbc_dim')
    def valid_pbc_dimension(cls, value: int) -> int:
        """Checks that dimensions are reasonable (0<=>3)."""
        assert 0 <= value <= 3, f"expected 0 <= dim <= 3, got {value}"
        return value

    @validator('pbc_axis')
    def valid_axis(cls, value: int) -> int:
        """Checks that axis is reasonable (1<=>3)."""
        assert 1 <= value <= 3, f"axis out of range (1 <= a <= 3)"
        return value


class CardModel(BaseModel):
    """
    Model for Card input.
    """
    units = 'bohr'
    group = 1
    position = [0.0, 0.0, 0.0]
    spread = 0.5
    dim = 0
    axis = 3

    @validator('group')
    def gt_zero(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than zero."""
        assert value > 0, f"{value} must be greater than zero"
        return value

    @validator('units')
    def valid_units(cls, value: str) -> str:
        """Checks value against acceptable units."""
        acceptable = (
            'bohr',
            'angstrom',
        )
        assert any(value == v for v in acceptable), \
            f"unexpected units"
        return value

    @validator('dim')
    def valid_dimension(cls, value: int) -> int:
        """Checks that dimensions are reasonable (0<=>3)."""
        assert 0 <= value <= 3, f"expected 0 <= dim <= 3, got {value}"
        return value

    @validator('axis')
    def valid_axis(cls, value: int) -> int:
        """Checks that axis is reasonable (1<=>3)."""
        assert 1 <= value <= 3, f"axis out of range (1 <= a <= 3)"
        return value


class ExternalModel(CardModel):
    """
    Model for the Externals input section.
    """
    charge: float

    @validator('charge')
    def ne_zero(cls, value: IntFloat) -> IntFloat:
        """Checks that value is not zero."""
        assert value != 0, f"{value} must not be zero"
        return value

    @validator('spread')
    def ge_zero(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than or equal to zero."""
        assert value >= 0, f"{value} must be greater than or equal to zero"
        return value


class RegionModel(CardModel):
    """
    Model for the Regions input section.
    """
    static = 1.0
    optical = 1.0
    width = 0.0

    @validator(
        'spread',
        'width',
    )
    def ge_zero(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than or equal to zero."""
        assert value >= 0, f"{value} must be greater than or equal to zero"
        return value

    @validator(
        'static',
        'optical',
    )
    def ge_one(cls, value: IntFloat) -> IntFloat:
        """Checks that value is greater than or equal to one."""
        assert value >= 1, f"{value} must be greater than or equal to one"
        return value


def main():

    natoms = 5

    parser = ConfigParser()

    file = Path(__file__).parent.joinpath('test.ini')

    if file.exists():
        parser.read(file)

    params = {}

    for section in parser.sections():
        section_params = dict(parser.items(section))
        params.update(section_params)

    control = ControlModel(**params)
    boundary = BoundaryModel(**params)
    electrostatic = ElectrostaticModel(**params)



if __name__ == '__main__':
    main()
