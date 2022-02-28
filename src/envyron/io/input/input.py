from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
import ruamel.yaml as yaml

from .base import InputModel


class Input:
    """
    Class for reading, casting, and validating Environ's input parameters.
    """

    def __init__(self, natoms: int, filename: Optional[str] = None) -> None:

        self.param_dict = {
            'control': {},
            'environment': {},
            'ions': {},
            'system': {},
            'electrolyte': {},
            'semiconductor': {},
            'solvent': {},
            'electrostatics': {},
            'pbc': {},
        }

        if filename is not None:
            self.read(filename)

        self.params = InputModel(natoms, **self.param_dict)

        del self.param_dict

    def read(self, filename: str) -> None:
        """Read and validate input file."""

        # read input file
        self.file = Path(filename).absolute()
        with open(self.file) as f:
            input_dict = yaml.safe_load(f)

        self.param_dict.update(input_dict)


def main():

    parser = ArgumentParser()

    parser.add_argument(
        '-n',
        metavar='natoms',
        help='Number of atoms',
        type=int,
        default=1,
    )

    parser.add_argument(
        '-f',
        metavar='filename',
        help='Input file name',
        type=str,
        default=None,
    )

    natoms, filename = vars(parser.parse_args()).values()

    my_input = Input(natoms, filename=filename).params

    for section in my_input:
        name, fields = section

        if fields:

            print(f"\n{name}\n")

            for field in fields:
                name, value = field

                if name == 'functions':

                    for i, group in enumerate(value, 1):
                        print(f"\ngroup {i}")

                        for function in group:
                            print(function)

                else:
                    print(f"{name} = {value}")

    print()


if __name__ == '__main__':
    main()
