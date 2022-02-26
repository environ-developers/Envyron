from argparse import ArgumentParser
from pathlib import Path
import ruamel.yaml as yaml

from .validation import EnvironInputModel


class Input:
    """
    Class for reading, casting, and validating Environ's input parameters.
    """

    def __init__(self, natoms: int, filename: str = 'environ.ini') -> None:

        # set the number of atoms in the validation model
        EnvironInputModel.set_number_of_atoms(natoms)

        # get input parameters from file
        self.file = Path(filename).absolute()

        with open(self.file) as f:
            param_dict: dict = yaml.safe_load(f)

        param_dict.update({'natoms': natoms})

        self.params = EnvironInputModel(**param_dict)


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
        default='test.ini',
    )

    natoms, filename = vars(parser.parse_args()).values()

    my_input = Input(natoms=natoms, filename=filename).params

    for k, v in my_input:
        if any(k == s for s in ('externals', 'regions')): continue
        print(f"{k} = {v}")

    externals = my_input.externals

    print(f"\nexternals | units = {externals.units}")

    for group in range(len(externals.functions)):
        print(f"\ngroup = {group + 1}\n")

        for function in externals.functions[group]:
            print(function)

    regions = my_input.regions

    print(f"\nregions | units = {regions.units}")

    for group in range(len(regions.functions)):
        print(f"\ngroup = {group + 1}\n")

        for function in regions.functions[group]:
            print(function)


if __name__ == '__main__':
    main()
