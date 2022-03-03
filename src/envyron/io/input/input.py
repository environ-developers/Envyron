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

        self._adjust_to_natoms()

        if filename is not None:
            self._adjust_input()
            self._final_validation()

        del self.param_dict

    def read(self, filename: str) -> None:
        """Read and validate input file."""

        # read input file
        self.file = Path(filename).absolute()
        with open(self.file) as f:
            input_dict = yaml.safe_load(f)

        self.param_dict.update(input_dict)

    def _adjust_to_natoms(self) -> None:
        """Scale ion input arrays to size of number of atoms."""

        for array in (
                self.params.ions.atomicspread,
                self.params.ions.corespread,
                self.params.ions.solvationrad,
        ):

            if len(array) == 1 and self.natoms != 1: array *= self.natoms

            if len(array) != self.natoms:
                raise ValueError("array size not equal to number of atoms")

    def _adjust_input(self) -> None:
        """Adjust input/default parameters based on user input."""
        self._adjust_environment()
        self._adjust_derivatives_method()
        self._adjust_electrostatics()

    def _adjust_environment(self) -> None:
        """Adjust environment properties according to environment type."""

        # set up vacuum environment
        if self.params.environment.type == 'vacuum':
            self.params.environment.static_permittivity = 1.0
            self.params.environment.optical_permittivity = 1.0
            self.params.environment.surface_tension = 0.0
            self.params.environment.pressure = 0.0

        # set up water environment
        elif 'water' in self.params.environment.type:
            self.params.environment.static_permittivity = 78.3
            self.params.environment.optical_permittivity = 1.776

            # non-ionic interfaces
            if self.params.solvent.mode in {'electronic', 'full'}:
                if self.params.environment.type == 'water':
                    self.params.environment.surface_tension = 50.0
                    self.params.environment.pressure = -0.35
                    self.params.solvent.rhomax = 5e-3
                    self.params.solvent.rhomin = 1e-4

                elif self.params.environment.type == 'water-cation':
                    self.params.environment.surface_tension = 50.0
                    self.params.environment.pressure = -0.35
                    self.params.solvent.rhomax = 5e-3
                    self.params.solvent.rhomin = 1e-4

                elif self.params.environment.type == 'water-anion':
                    self.params.environment.surface_tension = 50.0
                    self.params.environment.pressure = -0.35
                    self.params.solvent.rhomax = 5e-3
                    self.params.solvent.rhomin = 1e-4

            # ionic interface
            if self.params.solvent.mode == 'ionic':
                self.params.environment.surface_tension = 50.0
                self.params.environment.pressure = -0.35
                self.params.solvent.softness = 0.5
                self.params.solvent.radius_mode = 'uff'

                if self.params.environment.type == 'water':
                    self.params.solvent.alpha = 1.12

                elif self.params.environment.type == 'water-cation':
                    self.params.solvent.alpha = 1.1

                elif self.params.environment.type == 'water-anion':
                    self.params.solvent.alpha = 0.98

    def _adjust_derivatives_method(self) -> None:
        """Adjust derivatives method according to solvent mode."""

        if self.params.solvent.deriv_method == 'default':

            # non-ionic interfaces
            if self.params.solvent.mode in {'electronic', 'full', 'system'}:
                self.params.solvent.deriv_method = 'chain'

            # ionic interface
            elif self.params.solvent.mode == 'ionic':
                self.params.solvent.deriv_method = 'lowmem'

    def _adjust_electrostatics(self) -> None:
        """Adjust electrostatics according to solvent properties."""
        self._check_electrolyte_input()
        self._check_dielectric_input()

    def _check_electrolyte_input(self) -> None:
        """Adjust electrostatics according to electrolyte input."""

        if self.params.pbc.correction == 'gcs':

            if self.params.electrolyte.mode != 'system':
                self.params.electrolyte.mode = 'system'

            if self.params.electrolyte.formula is not None:

                # Linearized Poisson-Boltzmann problem
                if self.params.electrolyte.linearized:

                    if self.params.electrolyte.cionmax > 0.0 or \
                        self.params.electrolyte.rion > 0.0:
                        self.params.electrostatics.problem = 'linmodpb'

                    elif self.params.electrostatics.problem == 'none':
                        self.params.electrostatics.problem = 'linpb'

                    if self.params.electrostatics.solver == 'none':
                        self.params.electrostatics.solver = 'cg'

                else:  # Poisson-Boltzmann problem

                    if self.params.electrolyte.cionmax > 0.0 or \
                        self.params.electrolyte.rion > 0.0:
                        self.params.electrostatics.problem = 'modpb'

                    elif self.params.electrostatics.problem == 'none':
                        self.params.electrostatics.problem = 'pb'

                    if self.params.electrostatics.solver == 'none':
                        self.params.electrostatics.solver = 'newton'

        if self.params.pbc.correction == 'gcs' or \
            self.params.electrolyte.formula is not None:

            if self.params.electrolyte.deriv_method == 'default':

                # non-ionic interfaces
                if self.params.electrolyte.mode in {
                        'electronic',
                        'full',
                        'system',
                }:
                    self.params.electrolyte.deriv_method = 'chain'

                # ionic interface
                elif self.params.electrolyte.mode == 'ionic':
                    self.params.electrolyte.deriv_method = 'lowmem'

    def _check_dielectric_input(self) -> None:
        """Adjust electrostatics according to dielectric input."""

        if self.params.environment.static_permittivity > 1.0 or \
            self.params.regions is not None:

            if self.params.electrostatics.problem == 'none':
                self.params.electrostatics.problem = 'generalized'

            if self.params.pbc.correction != 'gcs':

                if self.params.electrostatics.solver == 'none':
                    self.params.electrostatics.solver = 'cg'

            elif self.params.electrostatics.solver != 'fixed-point':
                self.params.electrostatics.solver = 'fixed-point'

        else:

            if self.params.electrostatics.problem == 'none':
                self.params.electrostatics.problem = 'poisson'

            if self.params.electrostatics.solver == 'none':
                self.params.electrostatics.solver = 'direct'

        if self.params.electrostatics.solver == 'fixed-point' and \
            self.params.electrostatics.auxiliary == 'none':
            self.params.electrostatics.auxiliary = 'full'

    def _final_validation(self) -> None:
        """Check for bad input values."""
        self._validate_boundary()
        self._validate_derivatives_method()
        self._validate_electrostatics()

    def _validate_boundary(self) -> None:
        """Check for bad solvent input."""

        # solvent distance
        if self.params.solvent.mode == 'system' and \
            self.params.solvent.distance == 0.0:
            raise ValueError(
                "solvent distance must be greater than zero for system interfaces"
            )

    def _validate_derivatives_method(self) -> None:
        """Check for bad derivatives method."""

        # non-ionic interfaces
        if self.params.solvent.mode in {'electronic', 'full', 'system'}:
            if 'mem' in self.params.solvent.deriv_method:
                raise ValueError(
                    "only 'fft' or 'chain' allowed with electronic interfaces")

        # ionic interface
        elif self.params.solvent.mode == 'ionic':
            if self.params.solvent.deriv_method == 'chain':
                raise ValueError(
                    "only 'highmem', 'lowmem', and 'fft' allowed with ionic interfaces"
                )

    def _validate_electrostatics(self) -> None:
        """Check for bad electrostatics input."""

        # rhomax/rhomin validation
        if self.params.solvent.rhomax < self.params.solvent.rhomin:
            raise ValueError("rhomax < rhomin")

        # electrolyte rhomax/rhomin validation
        if self.params.electrolyte.rhomax < self.params.electrolyte.rhomin:
            raise ValueError("electrolyte rhomax < electrolyte rhomin")

        # pbc dim validation
        if self.params.pbc.dim == 1:
            raise ValueError("1D periodic boundary correction not implemented")

        # electrolyte validation

        if self.params.pbc.correction == 'gcs':

            if self.params.electrolyte.distance == 0.0:
                raise ValueError(
                    "electrolyte distance must be greater than zero for GCS correction"
                )

        if self.params.pbc.correction == 'gcs' or \
            self.params.electrolyte.formula is not None:

            # non-ionic interfaces
            if self.params.electrolyte.mode in {
                    'electronic', 'full', 'system'
            }:
                if 'mem' in self.params.electrolyte.deriv_method:
                    raise ValueError(
                        "only 'fft' or 'chain' allowed with electronic interfaces"
                    )

            # ionic interface
            elif self.params.electrolyte.mode == 'ionic':
                if self.params.electrolyte.deriv_method == 'chain':
                    raise ValueError(
                        "only 'highmem', 'lowmem', and 'fft' allowed with ionic interfaces"
                    )

        # problem/solver validation

        if self.params.electrostatics.problem == 'generalized':

            if self.params.electrostatics.solver == 'direct' or \
                self.params.electrostatics.inner_solver == 'direct':
                raise ValueError(
                    "cannot use direct solver for the Generalized Poisson eq.")

        elif "pb" in self.params.electrostatics.problem:

            if "lin" in self.params.electrostatics.problem:
                solvers = {'none', 'cg', 'sd'}

                if self.params.electrostatics.solver not in solvers or \
                    self.params.electrostatics.inner_solver not in solvers:
                    raise ValueError(
                        "only gradient-based solvers allowed for the linearized Poisson-Boltzmann eq."
                    )

                if self.params.pbc.correction != 'parabolic':
                    raise ValueError(
                        "linearized-PB problem requires parabolic PBC correction"
                    )

            else:
                solvers = {'direct', 'cg', 'sd'}

                if self.params.electrostatics.solver in solvers or \
                    self.params.electrostatics.inner_solver in solvers:
                    raise ValueError(
                        "no direct or gradient-based solver allowed for the full Poisson-Boltzmann eq."
                    )

        problems = {'pb, modpb, generalized'}

        if self.params.electrostatics.inner_solver != 'none' and \
            self.params.electrostatics.problem not in problems:
            raise ValueError("only pb or modpb problems allow inner solver")


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
