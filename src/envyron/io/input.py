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
        value: Any,
        condition: str = True,
        description: str = "",
    ) -> None:
        """Entry constructor."""
        self.section = section
        self.name = name
        self.type = type
        self.value = value
        self.condition = condition
        self.description = description

    def convert_value(self) -> None:
        """Convert str value to expected data type."""
        if self.type == 'str':
            value = self.value
        elif self.type == 'int':
            value = int(self.value)
        elif self.type == 'float':
            value = float(self.value)
        elif self.type == 'bool':
            value = self.boolean(self.value)
        else:
            raise TypeError(f"Unexpected {self.type} type")

        self.validate(value)
        self.value = value

    def validate(self, user_input: Any) -> bool:
        """Check if value is within criteria."""

        # check if array
        if isinstance(user_input, tuple):
            for v in user_input:
                self.validate(v)

        # validate input
        else:
            valid = eval(f"lambda x: {self.condition}")

            if not valid(user_input):
                raise ValueError(f"{user_input} is invalid for {self.name}")
            else:
                return True

    def boolean(self, value: str) -> bool:
        """Convert value to boolean."""

        # ensure that value is a pre-defined boolean state
        if value.lower() not in self.BOOLEAN_STATES:
            raise ValueError(f"{value} not a boolean")

        return self.BOOLEAN_STATES[value.lower()]

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
        value: Any,
        condition: str = True,
        description: str = "",
    ) -> None:
        """ArrayEntry constructor."""
        super().__init__(section, name, type, value, condition, description)
        self.size = size

    def convert_value(self) -> Any:
        """Convert str value to expected data type."""
        string: str = self.value
        values = string.split()

        if self.type == 'str':
            values = self.value
        elif self.type == 'int':
            values = tuple([int(v) for v in values])
        elif self.type == 'float':
            values = tuple([float(v) for v in values])
        elif self.type == 'bool':
            values = tuple([self.boolean(v) for v in values])
        else:
            raise TypeError(f"Unexpected {self.type} type")

        n = len(values)

        if n == 1:
            values = tuple([values[0]] * self.size)  # extrapolate to size
        else:
            if n != self.size: raise ValueError("Not enough values")

        self.validate(values)
        self.value = values


class Input:
    """Container for Environ input."""
    def __init__(self, natoms: int = 0) -> None:
        self.parser = ConfigParser()
        self.natoms = natoms

        here = Path(__file__).parent  # io directory

        with open(here.joinpath('params.json')) as f:
            self.defaults: dict = load(f)

        self.params: Dict[str, Entry] = {}

        for section in self.defaults:

            # skip template sections
            if section in {'Externals', 'Regions'}: continue

            # instantiate entries
            for param in self.defaults[section]:
                attrs: dict = self.defaults[section][param]

                # check for array input
                if 'size' in attrs:
                    entry = ArrayEntry(section, param, **attrs)
                else:
                    entry = Entry(section, param, **attrs)

                self.params[param] = entry

    def read(self, filename: str = 'environ.ini') -> None:
        """Overwrite defaults with user input."""
        if not Path(filename).exists():
            raise FileNotFoundError(f"Missing {filename} in working directory")

        self.parser.read(filename)

        self._process_user_input()

    def final_validation(self) -> None:
        """Check final input state."""

        # rhomax/rhomin validation
        rhomax: Entry = self.params.get('rhomax')
        rhomin: Entry = self.params.get('rhomin')

        if rhomax.value < rhomin.value:
            raise ValueError("rhomax < rhomin")

        rhomax: Entry = self.params.get('electrolyte_rhomax')
        rhomin: Entry = self.params.get('electrolyte_rhomin')

        if rhomax.value < rhomin.value:
            raise ValueError("electrolyte_rhomax < electrolyte_rhomin")

        # pbc_dim validation
        pbc_dim: Entry = self.params.get('pbc_dim')
        if pbc_dim.value == 1:
            raise ValueError("1D PBC corrections not yet implemented")

    def to_dict(self) -> Dict[str, Any]:
        """"""
        return {opt.name: opt.value for opt in self.params.values()}

    def _process_user_input(self):
        """Convert user input file to expected data types."""

        # check for simultaneous cionmax/rion setting
        cionmax = self.parser.get('Environ', 'cionmax', fallback=None)
        rion = self.parser.get('Environ', 'rion', fallback=None)

        if all(p is not None for p in (cionmax, rion)):
            raise ValueError("Cannot set both cionmax and rion")

        for section in self.parser.sections():

            if section not in self.defaults:
                raise ValueError(f"Unexpected {section} section")

            for opt, val in self.parser.items(section):

                if opt not in self.defaults[section]:
                    raise ValueError(
                        f"Unexpected {opt} option in {section} section")

                if self.params[opt].__class__ is ArrayEntry:
                    param: ArrayEntry = self.params[opt]
                    self._allocate_array_sizes(param, val)
                else:
                    param: Entry = self.params[opt]

                param.value = val
                param.convert_value()

    def _allocate_array_sizes(self, param: ArrayEntry, value: str) -> None:
        """Assign array sizes for allocatable arrays."""

        # assign array sizes
        if param.size is None:

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
    my_input.final_validation()
    params = my_input.to_dict()
    del my_input

    for k, v in params.items():
        print(f"{k} = {v}")
