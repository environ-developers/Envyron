from typing import Any, Dict, List, Tuple
from configparser import ConfigParser
from pathlib import Path
from json import load


class Entry:
    """Representation of an input entry."""

    BOOLEAN_STATES = {'true': True, 'false': False}

    def __init__(
        self,
        section: str,
        name: str,
        type: str,
        condition: str = True,
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
        """Convert str value to expected data type."""
        if not isinstance(value, str) or self.type == 'str': return value
        if self.type == 'int': return int(value)
        if self.type == 'float': return float(value)
        if self.type == 'bool': return self._boolean(value)
        raise TypeError(f"Unexpected {self.type} type")

    def _validate(self, user_input: Any) -> bool:
        """Check if value is within criteria."""
        if not self.valid(user_input):
            raise ValueError(f"{user_input} is invalid for {self.name}")

    def _boolean(self, value: str) -> bool:
        """Convert value to boolean."""
        if value.lower() in self.BOOLEAN_STATES:
            return self.BOOLEAN_STATES[value.lower()]
        raise ValueError(f"{value} not a boolean")

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
        condition: str = True,
        description: str = "",
    ) -> None:
        """ArrayEntry constructor."""
        super().__init__(section, name, type, condition, description)
        self.size = size

    def _convert(self, value: Any) -> Tuple[Any]:
        """Convert str value to expected data type."""

        if isinstance(value, list):
            values = value
        elif isinstance(value, str):
            values = value.split()
        else:
            values = tuple((value, ))

        if self.type == 'str':
            values = tuple(self.value)
        elif self.type == 'int':
            values = tuple([int(v) for v in values])
        elif self.type == 'float':
            values = tuple([float(v) for v in values])
        elif self.type == 'bool':
            values = tuple([self._boolean(v) for v in values])
        else:
            raise TypeError(f"Unexpected {self.type} type")

        if self.size is not None:
            n = len(values)
            if n == 1:
                values = tuple([values[0]] * self.size)  # extrapolate to size
            else:
                if n != self.size: raise ValueError("Not enough values")

        return values

    def _validate(self, user_input: Any) -> bool:
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

        self.externals = None
        self.regions = None

        self._read_defaults()

    def read(self, filename: str = 'environ.ini') -> None:
        """Overwrite defaults with user input."""
        if not Path(filename).exists():
            raise FileNotFoundError(f"Missing {filename} in working directory")

        self.parser.read(filename)

        self._process_user_input()

    def validate(self) -> None:
        """Check for unreasonable input values."""

        # rhomax/rhomin validation
        rhomax: Entry = self.entries.get('rhomax')
        rhomin: Entry = self.entries.get('rhomin')

        if rhomax.value < rhomin.value:
            raise ValueError("rhomax < rhomin")

        # electrolyte rhomax/rhomin validation
        rhomax: Entry = self.entries.get('electrolyte_rhomax')
        rhomin: Entry = self.entries.get('electrolyte_rhomin')

        if rhomax.value < rhomin.value:
            raise ValueError("electrolyte_rhomax < electrolyte_rhomin")

        # pbc_dim validation
        pbc_dim: Entry = self.entries.get('pbc_dim')

        if pbc_dim.value == 1:
            raise ValueError("1D periodic boundary correction not implemented")

    def get_parameters(self) -> Any:
        """Return input parameters as a class."""
        return type('InputParams', (), self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        """"""
        param_dict = {opt.name: opt.value for opt in self.entries.values()}

        if self.externals: param_dict['externals'] = self.externals
        if self.regions: param_dict['regions'] = self.regions

        return param_dict

    def _read_defaults(self) -> None:
        """Read and process default values."""
        here = Path(__file__).parent  # io directory

        with open(here.joinpath('params.json')) as f:
            self.params: Dict[str, Dict[str, dict]] = load(f)

        with open(here.joinpath('defaults.json')) as f:
            self.defaults: Dict[str, Dict[str, dict]] = load(f)

        self.entries: Dict[str, Entry] = {}

        for section in self.params:

            # skip template sections
            if section in {'Externals', 'Regions'}: continue

            # instantiate entries
            for param in self.params[section]:
                attrs: dict = self.params[section][param]

                # check for array input
                if 'size' in attrs:
                    entry = ArrayEntry(section, param, **attrs)
                else:
                    entry = Entry(section, param, **attrs)

                entry.value = self.defaults[section][param]

                self.entries[param] = entry

    def _process_user_input(self):
        """Convert user input file to expected data types."""

        # check for simultaneous cionmax/rion setting
        cionmax = self.parser.get('Environ', 'cionmax', fallback=None)
        rion = self.parser.get('Environ', 'rion', fallback=None)

        if all(p is not None for p in (cionmax, rion)):
            raise ValueError("Cannot set both cionmax and rion")

        for section in self.parser.sections():

            if section == 'Externals':
                self._process_externals()
            elif section == 'Regions':
                self._process_regions()
            else:
                if section not in self.params:
                    raise ValueError(f"Unexpected {section} section")

                for opt, val in self.parser.items(section):

                    # verify that option belongs to this section
                    if opt not in self.params[section]:
                        raise ValueError(
                            f"Unexpected {opt} option for {section} section")

                    # get entry object
                    if self.entries[opt].__class__ is ArrayEntry:
                        param: ArrayEntry = self.entries[opt]
                        self._allocate_array_sizes(param, val)
                    else:
                        param: Entry = self.entries[opt]

                    param.value = val

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

    def _process_externals(self) -> None:
        """Process Externals input section."""
        self.externals: List[List[dict]] = []

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
        self.regions: List[List[dict]] = []

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


if __name__ == '__main__':

    natoms = 5

    my_input = Input(natoms)

    if Path(__file__).parent.joinpath('test.ini').exists():
        my_input.read('test.ini')

    my_input.validate()
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
