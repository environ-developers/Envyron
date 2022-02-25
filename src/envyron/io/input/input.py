from argparse import ArgumentParser
from typing import Any, Dict, List
from pathlib import Path
from configparser import ConfigParser

from .validation import (
    EnvironInputModel,
    ExternalModel,
    ExternalsContainer,
    RegionModel,
    RegionsContainer,
)


class Input:
    """
    Class for reading, casting, and validating Environ's input parameters.
    """

    def __init__(self, natoms: int = 1, filename: str = 'environ.ini') -> None:

        # set the number of atoms in the validation model
        EnvironInputModel.set_number_of_atoms(natoms)

        # get input parameters from file
        self.file = Path(filename).absolute()
        param_dict = self._get_input_param_dict()

        self.params = EnvironInputModel(**param_dict)

    def _get_input_param_dict(self) -> Dict[str, Any]:
        """
        Return input file as parameter dictionary.
        """
        if not self.file.exists():
            raise FileNotFoundError(
                f"Missing {self.file.name} in working directory")

        # parse config file
        parser = ConfigParser()
        parser.read(self.file)

        # build parameter dictionary
        param_dict = dict.fromkeys(('externals', 'regions'))
        for section in parser.sections():
            section_params = dict(parser.items(section))

            if section == 'Externals':
                externals = self._process_externals(section_params)
                param_dict['externals'] = externals

            elif section == 'Regions':
                regions = self._process_regions(section_params)
                param_dict['regions'] = regions

            else:
                param_dict.update(section_params)

        return param_dict

    def _process_externals(self, data: Dict[str, str]) -> ExternalsContainer:
        """
        Convert externals raw data into grouped lists of ExternalModel objects.
        """
        functions: List[List[ExternalModel]] = []

        group = 0

        for function in sorted(data['functions'].split('\n'), key=by_group):

            g, c, x, y, z, s, d, a = function.split()

            try:
                if int(g) != group:
                    functions.append([])
                    group += 1
            except:
                raise ValueError("externals group must be an integer")

            func_dict = {
                'charge': c,
                'position': [x, y, z],
                'spread': s,
                'dim': d,
                'axis': a,
            }

            functions[group - 1].append(ExternalModel(**func_dict))

        processed_data = {'units': data['units'], 'functions': functions}

        externals = ExternalsContainer(**processed_data)

        return externals

    def _process_regions(self, data: Dict[str, str]) -> RegionsContainer:
        """
        Convert regions raw data into grouped lists of RegionModel objects.
        """

        functions: List[List[RegionModel]] = []

        group = 0

        for function in sorted(data['functions'].split('\n'), key=by_group):

            g, eps, opt, x, y, z, w, s, d, a = function.split()

            try:
                if int(g) != group:
                    functions.append([])
                    group += 1
            except:
                raise ValueError("regions group must be an integer")

            func_dict = {
                'static': eps,
                'optical': opt,
                'position': [x, y, z],
                'spread': s,
                'width': w,
                'dim': d,
                'axis': a,
            }

            functions[group - 1].append(RegionModel(**func_dict))

        processed_data = {'units': data['units'], 'functions': functions}

        regions = RegionsContainer(**processed_data)

        return regions


def by_group(x):
    """
    Sort card functions by group.
    """
    return x[0]


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
