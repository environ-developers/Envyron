from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional
import ruamel.yaml as yaml

from .base import InputModel


class Input:
    """
    Class for reading, casting, and validating Environ's input parameters.
    """

    def __init__(self, natoms: int, filename: Optional[str] = None) -> None:

        self.param_dict: Dict[str, Dict[str, Any]] = {
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

        self.natoms = natoms

        if filename is not None:
            self.read(filename)

        self.params = InputModel(**self.param_dict)

        self.params.adjust_ionic_arrays(self.natoms)

        if filename is not None:
            self.params.apply_smart_defaults()
            self.params.sanity_check()

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
        dest='natoms',
        help='Number of atoms',
        type=int,
        default=1,
    )

    parser.add_argument(
        '-f',
        metavar='filename',
        dest='filename',
        help='Input file name',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Turn on debugging output',
    )

    args = vars(parser.parse_args())

    if not args['debug']:
        import sys
        sys.tracebacklimit = 0

    my_input = Input(args['natoms'], args['filename']).params

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
