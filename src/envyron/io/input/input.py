from argparse import ArgumentParser
from pathlib import Path
import ruamel.yaml as yaml

from .base import InputModel


class Input:
    """
    Class for reading, casting, and validating Environ's input parameters.
    """

    def __init__(self, natoms: int, filename: str = 'environ.yml') -> None:

        # read input file
        self.file = Path(filename).absolute()        
        with open(self.file) as f:
            param_dict: dict = yaml.safe_load(f)

        # validate and set input parameters
        self.params = InputModel(natoms, **param_dict)


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
        default='test.yml',
    )

    natoms, filename = vars(parser.parse_args()).values()

    my_input = Input(natoms=natoms, filename=filename).params

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
