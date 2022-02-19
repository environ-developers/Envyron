from typing import Any, Dict, Tuple, ValuesView
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


class Input:
    """Container for Environ input."""
    def __init__(self, natoms: int = 0) -> None:
        self.parser = ConfigParser()
        self.natoms = natoms

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

    def to_dict(self) -> Dict[str, Any]:
        """Return input as {option: value} dictionary."""
        return {opt.name: opt.value for opt in self.entries.values()}

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


if __name__ == '__main__':

    natoms = 5

    my_input = Input(natoms)
    my_input.read()
    my_input.validate()
    params = my_input.to_dict()
    del my_input

    for k, v in params.items():
        print(f"{k} = {v}")
