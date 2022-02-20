from typing import Any, Dict, List, Tuple, Union
from configparser import ConfigParser
from pathlib import Path
from json import load


class Params:
    """Collection of input parameters."""
    def __init__(self, params: dict) -> None:
        for param, value in params.items():
            setattr(self, param, value)


class Entry:
    """Representation of an input entry."""
    def __init__(
        self,
        section: str,
        name: str,
        type: str,
        condition: str = 'True',
        description: str = "",
    ) -> None:
        """Entry constructor."""
        self.section = section
        self.name = name
        self.type = type
        self.valid = eval(f"lambda x: {condition}")
        self.description = description
        self.__value = None

    @property
    def value(self) -> Any:
        """docstring"""
        return self.__value

    @value.setter
    def value(self, value: Any) -> None:
        """Setter for Entry value."""
        value = self._convert(value)
        self._validate(value)
        self.__value = value

    def _convert(self, value: Any) -> Any:
        """Convert value to expected data type."""
        if not isinstance(value, str) or self.type == 'str': return value
        if self.type == 'int': return int(value)
        if self.type == 'float': return float(value)
        if self.type == 'bool': return self._boolean(value)
        raise TypeError(f"Unexpected {self.type} type")

    def _validate(self, user_input: Any) -> None:
        """Check if value is within criteria."""
        if not self.valid(user_input):
            raise ValueError(f"{user_input} is invalid for {self.name}")

    def _boolean(self, value: str) -> bool:
        """Convert value to boolean."""
        boolean_states = {'true': True, 'false': False}
        boolean = boolean_states.get(value.lower())
        if boolean is None: raise ValueError(f"{value} not a boolean")
        return boolean

    def __str__(self) -> str:
        return f"{self.name} -> {self.description}"


class ArrayEntry(Entry):
    """Representation of an input array entry."""
    def __init__(
        self,
        section: str,
        name: str,
        type: str,
        size: int,
        condition: str = 'True',
        description: str = "",
    ) -> None:
        """ArrayEntry constructor."""
        super().__init__(section, name, type, condition, description)
        self.size = size

    def _convert(self, value: Any) -> Tuple:
        """Convert value to expected data type."""

        if isinstance(value, list) or isinstance(value, tuple):
            pre_conversion = value
        elif isinstance(value, str):
            pre_conversion = value.split()
        else:
            pre_conversion = [value]

        values = []
        for val in pre_conversion:
            values.append(super()._convert(val))

        n = len(values)
        if n == 1:
            values = [values[0]] * self.size  # extrapolate to size
        else:
            if n != self.size: raise ValueError("Not enough values")

        return tuple(values)

    def _validate(self, user_input: Any) -> None:
        """Check if value is within criteria."""
        for v in user_input:
            super()._validate(v)


class Card:
    """Parent class for card input."""
    def __repr__(self) -> str:
        return f"{self.__dict__}"

    def __str__(self) -> str:
        return f"{self.__dict__}"


class ExternalInput(Card):
    """Input representation of an external charge input."""

    units = ''

    def __init__(
        self,
        charge: float,
        pos: List[str],
        spread: float,
        dim: int,
        axis: int,
    ) -> None:
        self.charge = charge
        self.pos = pos
        self.spread = spread
        self.dim = dim
        self.axis = axis


class RegionInput(Card):
    """Input representation of a dielectric region."""

    units = ''

    def __init__(
        self,
        static: float,
        optical: float,
        pos: List[str],
        width: float,
        spread: float,
        dim: int,
        axis: int,
    ) -> None:
        self.static = static
        self.optical = optical
        self.pos = pos
        self.width = width
        self.spread = spread
        self.dim = dim
        self.axis = axis


class Input:
    """Container for Environ input."""
    def __init__(self, natoms: int = 0) -> None:
        self.natoms = natoms

        self.parser = ConfigParser()  # user-input parser

        self.entries: Dict[str, Union[Entry, ArrayEntry]] = {}

        self.externals: List[List[ExternalInput]] = []
        self.regions: List[List[RegionInput]] = []

        self._read_defaults()

    def read(self, filename: str = 'environ.ini') -> None:
        """Overwrite defaults with user input."""
        if not Path(filename).exists():
            raise FileNotFoundError(f"Missing {filename} in working directory")

        self.parser.read(filename)

        self._process_user_input()

    def to_dict(self) -> Dict[str, Any]:
        """"""
        param_dict = {opt.name: opt.value for opt in self.entries.values()}

        if self.externals: param_dict['externals'] = self.externals
        if self.regions: param_dict['regions'] = self.regions

        return param_dict

    def get_parameters(self) -> Params:
        """Return input parameters as a class."""
        parameters = self.to_dict()
        return Params(parameters)

    def _read_defaults(self) -> None:
        """Read and process default values."""
        here = Path(__file__).parent  # io directory

        # read input parameter attributes
        with open(here / 'params.json') as f:
            self.params: Dict[str, Dict[str, dict]] = load(f)

        # read input parameter default values
        with open(here / 'defaults.json') as f:
            self.defaults: Dict[str, Dict[str, dict]] = load(f)

        self._build_entry_dictionary()

    def _build_entry_dictionary(self) -> None:
        """Generate dictionary of entry objects from param/default data."""

        for section in self.params:

            # skip template sections
            if section in {'Externals', 'Regions'}: continue

            # instantiate entries
            for param in self.params[section]:

                attrs = self.params[section][param]

                # check for array input
                entry = ArrayEntry(section, param, **attrs) \
                    if 'size' in attrs else Entry(section, param, **attrs)

                # set default value (validated internally)
                entry.value = self.defaults[section][param]

                self.entries[param] = entry

    def _process_user_input(self) -> None:
        """Convert user input to expected data types."""

        # check for simultaneous cionmax/rion setting
        cionmax = self.parser.get('Environ', 'cionmax', fallback=None)
        rion = self.parser.get('Environ', 'rion', fallback=None)

        if cionmax is not None and rion is not None:
            raise ValueError("Cannot set both cionmax and rion")

        self._process_input_sections()
        self._adjust_input()
        self._validate_input()

    def _process_input_sections(self) -> None:
        """Process input sections in user input file."""

        for section in self.parser.sections():

            if section not in self.params:
                raise ValueError(f"Unexpected {section} section")

            if section == 'Externals':
                self._process_externals()
            elif section == 'Regions':
                self._process_regions()
            else:
                self._process_input_options(section)

    def _process_input_options(self, section: str) -> None:
        """Process input options for given section."""

        for opt, val in self.parser.items(section):

            # verify that option belongs to this section
            if opt not in self.params[section]:
                raise ValueError(
                    f"Unexpected {opt} option for {section} section")

            # get entry object
            param = self.entries[opt]

            if isinstance(param, ArrayEntry):
                self._allocate_array_sizes(param, val)

            param.value = val

    def _process_externals(self) -> None:
        """Process Externals input section."""

        section = 'Externals'

        # set up template for external object
        defaults: dict = self.params[section]

        template: Dict[str, Entry] = {
            'units': Entry(section, 'units', **defaults['units']),
            'group': Entry(section, 'group', **defaults['group']),
            'charge': Entry(section, 'charge', **defaults['charge']),
            'pos': ArrayEntry(section, 'position', **defaults['position']),
            'spread': Entry(section, 'spread', **defaults['spread']),
            'dim': Entry(section, 'dim', **defaults['dim']),
            'axis': Entry(section, 'axis', **defaults['axis']),
        }

        # set units for externals
        template['units'].value = self.parser.get(section, 'units')
        ExternalInput.units = template['units'].value

        group = -1

        # iterate over external functions
        for function in self.parser.get(section, 'functions').split('\n'):

            # get values
            g, c, x, y, z, s, d, a = function.split()

            # convert, validate, and set values
            template['group'].value = g
            template['charge'].value = c
            template['pos'].value = " ".join((x, y, z))
            template['spread'].value = s
            template['dim'].value = d
            template['axis'].value = a

            # add new externals group if necessary
            if template['group'].value != group + 1:
                self.externals.append([])
                group += 1

            # set external values
            keys = (
                'charge',
                'pos',
                'spread',
                'dim',
                'axis',
            )
            attrs = {k: template[k].value for k in keys}
            external = ExternalInput(**attrs)

            # add external to list
            self.externals[group].append(external)

    def _process_regions(self) -> None:
        """Process Regions input section."""

        section = 'Regions'

        # set up template for region object
        defaults: dict = self.params[section]

        template: Dict[str, Entry] = {
            'units': Entry(section, 'units', **defaults['units']),
            'group': Entry(section, 'group', **defaults['group']),
            'static': Entry(section, 'static', **defaults['static']),
            'optical': Entry(section, 'optical', **defaults['optical']),
            'pos': ArrayEntry(section, 'position', **defaults['position']),
            'width': Entry(section, 'width', **defaults['width']),
            'spread': Entry(section, 'spread', **defaults['spread']),
            'dim': Entry(section, 'dim', **defaults['dim']),
            'axis': Entry(section, 'axis', **defaults['axis']),
        }

        # set units for regions
        template['units'].value = self.parser.get(section, 'units')
        RegionInput.units = template['units'].value

        group = -1

        # iterate over region functions
        for function in self.parser.get(section, 'functions').split('\n'):

            # get values
            g, eps, opt, x, y, z, w, s, d, a = function.split()

            # convert and validate values
            template['group'].value = g
            template['static'].value = eps
            template['optical'].value = opt
            template['pos'].value = " ".join((x, y, z))
            template['width'].value = w
            template['spread'].value = s
            template['dim'].value = d
            template['axis'].value = a

            # add new regions group if necessary
            if template['group'].value != group + 1:
                self.regions.append([])
                group += 1

            # set region values
            keys = (
                'static',
                'optical',
                'pos',
                'width',
                'spread',
                'dim',
                'axis',
            )
            attrs = {k: template[k].value for k in keys}
            region = RegionInput(**attrs)

            # add region to list
            self.regions[group].append(region)

    def _allocate_array_sizes(self, param: ArrayEntry, value: str) -> None:
        """Assign array sizes for allocatable arrays."""

        if param.size == 0:

            # assign allocation size for ion values
            if param.name in {'atomicspread', 'corespread', 'solvationrad'}:
                param.size = self.natoms

            # assign allocation size for electrolyte values
            elif param.name == 'electrolyte_formula':
                parts = value.split()
                n = len(parts)

                # check for sufficient charges
                if n < 4: raise ValueError("Multiplicity/charge sets < 2")

                # check for complete formula
                if n % 2 != 0: raise ValueError("Missing multiplicity/charge")

                # check for charge neutrality
                m = [int(i) for i in parts[::2]]  # multiplicities
                z = [int(i) for i in parts[1::2]]  # charges
                s = sum(i * j for i, j in zip(m, z))

                if s != 0: raise ValueError("Electrolyte is not neutral")

                param.size = n

            # missing pre-defined size in defaults
            else:
                raise ValueError(f"Unexpected allocatable array: {param.name}")

    def _adjust_input(self) -> None:
        """Adjust input parameters based on user input."""
        self._adjust_environment()
        self._adjust_derivatives_method()
        self._adjust_electrostatics()

    def _adjust_environment(self) -> None:
        """Adjust environment properties according to environment type."""
        environment_type = self._get_value('env_type')

        # set up vacuum environment
        if environment_type == 'vacuum':
            self.entries['static_permittivity'].value = 1.0
            self.entries['optical_permittivity'].value = 1.0
            self.entries['surface_tension'].value = 0.0
            self.entries['pressure'].value = 0.0

        # set up water environment
        elif 'water' in environment_type:
            self.entries['static_permittivity'].value = 78.3
            self.entries['optical_permittivity'].value = 1.776

            solvent_mode = self._get_value('solvent_mode')

            # non-ionic interfaces
            if solvent_mode in {'electronic', 'full'}:
                if environment_type == 'water':
                    self.entries['surface_tension'].value = 50.0
                    self.entries['pressure'].value = -0.35
                    self.entries['rhomax'].value = 5e-3
                    self.entries['rhomin'].value = 1e-4

                elif environment_type == 'water-cation':
                    self.entries['surface_tension'].value = 50.0
                    self.entries['pressure'].value = -0.35
                    self.entries['rhomax'].value = 5e-3
                    self.entries['rhomin'].value = 1e-4

                elif environment_type == 'water-anion':
                    self.entries['surface_tension'].value = 50.0
                    self.entries['pressure'].value = -0.35
                    self.entries['rhomax'].value = 5e-3
                    self.entries['rhomin'].value = 1e-4

            # ionic interface
            if solvent_mode == 'ionic':
                self.entries['surface_tension'].value = 50.0
                self.entries['pressure'].value = -0.35
                self.entries['softness'].value = 0.5
                self.entries['radius_mode'].value = 'uff'

                if environment_type == 'water':
                    self.entries['alpha'].value = 1.12

                elif environment_type == 'water-cation':
                    self.entries['alpha'].value = 1.1

                elif environment_type == 'water-anion':
                    self.entries['alpha'].value = 0.98

    def _adjust_derivatives_method(self) -> None:
        """Adjust derivatives method according to solvent mode."""
        mode = self._get_value('solvent_mode')
        method = self._get_value('deriv_method')

        if method == 'default':

            # non-ionic interfaces
            if mode in {'electronic', 'full', 'system'}:
                self.entries['deriv_method'].value = 'chain'

            # ionic interface
            elif mode == 'ionic':
                self.entries['deriv_method'].value = 'lowmem'

    def _adjust_electrostatics(self) -> None:
        """Adjust electrostatics according to solvent properties."""
        correction = self._get_value('pbc_correction')
        self._check_electrolyte_input(correction)
        self._check_dielectric_input(correction)

    def _check_electrolyte_input(self, correction: str) -> None:
        """Adjust electrostatics according to electrolyte input."""
        mode = self._get_value('electrolyte_mode')
        formula = self._get_entry('electrolyte_formula')

        size = formula.size if isinstance(formula, ArrayEntry) else 0

        if correction == 'gcs':
            if mode != 'system':
                self.entries['electrolyte_mode'].value = 'system'

            if size != 0:
                rion = self._get_value('rion')
                cionmax = self._get_value('cionmax')
                linearized = self._get_value('electrolyte_linearized')
                solver = self._get_value('solver')

                if linearized:  # Linearized Poisson-Boltzmann problem
                    if cionmax > 0.0 or rion > 0.0:
                        self.entries['problem'].value = 'linmodpb'
                    elif self.entries['problem'].value == 'none':
                        self.entries['problem'].value = 'linpb'

                    if solver == 'none':
                        self.entries['solver'].value = 'cg'

                else:  # Poisson-Boltzmann problem
                    if cionmax > 0.0 or rion > 0.0:
                        self.entries['problem'].value = 'modpb'
                    elif self.entries['problem'].value == 'none':
                        self.entries['problem'].value = 'pb'

                    if solver == 'none':
                        self.entries['solver'].value = 'newton'

        if correction == 'gcs' or size != 0:
            method = self._get_value('electrolyte_deriv_method')

            if method == 'default':

                # non-ionic interfaces
                if mode in {'electronic', 'full', 'system'}:
                    self.entries['electrolyte_deriv_method'].value = 'chain'

                # ionic interface
                elif mode == 'ionic':
                    self.entries['electrolyte_deriv_method'].value = 'lowmem'

    def _check_dielectric_input(self, correction: str) -> None:
        """Adjust electrostatics according to dielectric input."""
        static_permittivity = self._get_value('static_permittivity')
        num_of_regions = self._get_value('dielectric_regions')
        problem = self._get_value('problem')
        solver = self._get_value('solver')
        auxiliary = self._get_value('auxiliary')

        if static_permittivity > 1.0 or num_of_regions > 0:
            if problem == 'none': self.entries['problem'].value = 'generalized'

            if correction != 'gcs':
                if solver == 'none': self.entries['solver'].value = 'cg'
            elif solver != 'fixed-point':
                self.entries['solver'].value = 'fixed-point'

        else:
            if problem == 'none': self.entries['problem'].value = 'poisson'
            if solver == 'none': self.entries['solver'].value = 'direct'

        if self._get_value('solver') == 'fixed-point' and auxiliary == 'none':
            self.entries['auxiliary'].value = 'full'

    def _validate_input(self) -> None:
        """Check for bad input values."""
        self._validate_derivatives_method()
        self._validate_electrostatics()

    def _validate_derivatives_method(self) -> None:
        """Check for bad derivatives method."""

        # derivatives method validation
        mode = self._get_value('solvent_mode')
        method = self._get_value('deriv_method')

        # non-ionic interfaces
        if mode in {'electronic', 'full', 'system'}:
            if 'mem' in method:
                raise ValueError(
                    "Only 'fft' or 'chain' are allowed with electronic interfaces"
                )

        # ionic interface
        elif mode == 'ionic':
            if method == 'chain':
                raise ValueError(
                    "Only 'highmem', 'lowmem', and 'fft' are allowed with ionic interfaces"
                )

    def _validate_electrostatics(self) -> None:
        """Check for bad electrostatics input."""

        # rhomax/rhomin validation
        rhomax = self._get_value('rhomax')
        rhomin = self._get_value('rhomin')

        if rhomax < rhomin: raise ValueError("rhomax < rhomin")

        # electrolyte rhomax/rhomin validation
        rhomax = self._get_value('electrolyte_rhomax')
        rhomin = self._get_value('electrolyte_rhomin')

        if rhomax < rhomin:
            raise ValueError("electrolyte_rhomax < electrolyte_rhomin")

        # pbc_dim validation
        pbc_dim = self._get_value('pbc_dim')

        if pbc_dim == 1:
            raise ValueError("1D periodic boundary correction not implemented")

        # electrolyte validation
        correction = self._get_value('pbc_correction')
        formula = self._get_entry('electrolyte_formula')

        size = formula.size if isinstance(formula, ArrayEntry) else 0

        if correction == 'gcs':
            distance = self.entries['electrolyte_distance']

            if distance == 0.0:
                raise ValueError(
                    "electrolyte_distance must be greater than zero for gcs correction"
                )

        if correction == 'gcs' or size != 0:
            mode = self._get_value('electrolyte_mode')
            method = self._get_value('electrolyte_deriv_method')

            # non-ionic interfaces
            if mode in {'electronic', 'full', 'system'}:
                if 'mem' in method:
                    raise ValueError(
                        "Only 'fft' or 'chain' are allowed with electronic interfaces"
                    )

            # ionic interface
            elif mode == 'ionic':
                if method == 'chain':
                    raise ValueError(
                        "Only 'highmem', 'lowmem', and 'fft' are allowed with ionic interfaces"
                    )

        # problem/solver validation
        problem = self._get_value('problem')
        solver = self._get_value('solver')
        inner_solver = self._get_value('inner_solver')

        if problem == 'generalized':

            if solver == 'direct' or inner_solver == 'direct':
                raise ValueError(
                    "Cannot use a direct solver for the Generalized Poisson eq."
                )

        elif "pb" in problem:

            if "lin" in problem:
                solvers = {'none', 'cg', 'sd'}

                if solver not in solvers or inner_solver not in solvers:
                    raise ValueError(
                        "Only gradient-based solver for the linearized Poisson-Boltzmann eq."
                    )

                if correction != 'parabolic':
                    raise ValueError(
                        "Linearized-PB problem requires parabolic pbc correction"
                    )

            else:
                solvers = {'direct', 'cg', 'sd'}

                if solver in solvers or inner_solver in solvers:
                    raise ValueError(
                        "No direct or gradient-based solver for the full Poisson-Boltzmann eq."
                    )

        problems = {'pb, modpb, generalized'}

        if inner_solver != 'none' and problem not in problems:
            raise ValueError("Only pb or modpb problems allow inner solver")

    def _get_entry(self, option: str) -> Union[Entry, ArrayEntry]:
        """Return entry object if exists."""
        entry = self.entries.get(option)
        if entry is None: raise ValueError(f"Missing {option} entry")
        return entry

    def _get_value(self, option: str) -> Any:
        """Return entry value if exists."""
        entry = self.entries.get(option)
        if entry is None: raise ValueError(f"Missing {option} entry")
        return entry.value


def main():

    natoms = 5

    my_input = Input(natoms)

    if Path(__file__).parent.joinpath('test.ini').exists():
        my_input.read('test.ini')

    params = my_input.to_dict()
    del my_input

    for k, v in params.items():

        if k in {'externals', 'regions'}:
            print(f"\n{k.upper()}")
            for i, l in enumerate(v):
                print(f"\nGROUP {i + 1}")
                for d in l:
                    print(d)

        else:
            print(f"{k} = {v}")


if __name__ == '__main__':
    main()
