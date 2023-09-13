import numpy as np
from ase import Atoms
from ase.units import Bohr
from dataclasses import dataclass, field


# From Stephen Weitzner cube_vizkit
@dataclass
class EnvironCube:
    """
    A data class for storing and manipulating cube files. Contains cell basis,
    atomic basis, and scalar field data.

    atoms: ase.Atoms
        Atoms object constructed from cell and atomic positions in cube file.

    bonds: tuple, shape(M,2)
        Pairs of indices that are first nearest neighbors in self.atoms

    grid: np.ndarray, shape [3, Nx, Ny, Nz]
        Rank 4 array that contains the cartesian coordinates of the numerical
        grid in units of Bohr.

    data3D: np.ndarray, shape [Nx, Ny, Nz]
        Rank 3 array that contains scalar field data on the corresponding grid.

        If charge data, in units of 'e'.
        If potential data, in units of 'Ry/e'.

    cell: np.ndarray, shape [3, 3]
        Each column contains a basis vector of the supercell in units of Bohr.

    origin: np.ndarray, shape[1, 3]
        origin of the supercell / atoms.

    TODO
    ----
    + Add a method for adding, subtracting, scaling cube files
      -> e.g., Allow for density difference plots
    + Add method for getting polyhedra
    + Test the interpolator method and save the output to a new cube object
    + Add method to write cube files (or other file formats?)
    """

    fname: str = ''
    scaling_factor: float = 1.1
    units: str = 'Bohr'
    atoms: list[tuple] = field(default_factory=list, repr=False)
    bonds: list[tuple] = field(default_factory=list, repr=False)
    cell: list[tuple] = field(default_factory=list, repr=False)
    data3D: list[tuple] = field(default_factory=list, repr=False)
    grid: list[tuple] = field(default_factory=list, repr=False)
    origin: list[tuple] = field(default_factory=list, repr=False)
    prefix: str = field(default_factory=str, repr=False)

    def load_cube(self, fname='', units='Bohr'):
        """
        load_cube(cube_file)

        Extracts numerical data from Gaussian *.cube files.
        Atomic units are assumed

        Parameters
        ----------
        units: string, optional (default='Bohr')

        Returns
        -------
        None

        References
        ----------
        [1] http://www.gaussian.com/g_tech/g_ur/u_cubegen.htm

        To Do
        -----
        -> Nothing for the moment.
        """

        if fname:
            self.fname = fname
        assert self.fname, "No filename provided."

        self.prefix = self.fname.split('.')[0]
        self.units = units

        with open(self.fname, 'r') as f:
            contents = f.readlines()

        # -- Parse the header of the cube file
        del contents[0:2]  # remove first 2 comment lines
        tmp = contents[0].split()
        num_atoms, origin = int(tmp[0]), np.array(list(map(float, tmp[1:])))
        self.origin = origin
        header = contents[1:num_atoms + 4]
        N1 = int(header[0].split()[0])
        N2 = int(header[1].split()[0])
        N3 = int(header[2].split()[0])
        R1 = list(map(float, header[0].split()[1:4]))
        R2 = list(map(float, header[1].split()[1:4]))
        R3 = list(map(float, header[2].split()[1:4]))

        # -- Get supercell dimensions
        basis = np.array([R1, R2, R3], dtype='d').T  # store vectors as columns
        scalars = np.array([N1, N2, N3], dtype='d')
        self.cell = basis * scalars  # broadcasting

        # -- Create an ASE Atoms object
        tmp = np.array([line.split() for line in header[3:]], dtype='d')
        numbers = tmp[:, 0].astype(int)
        charges = tmp[:, 1]
        positions = tmp[:, 2:]
        if self.units == 'Bohr' :
            positions = positions * Bohr
        self.atoms = Atoms(numbers=numbers,
                           positions=positions,
                           charges=charges,
                           cell=self.cell.T)

        # -- Construct the grid
        mesh = np.mgrid[0:N3, 0:N2, 0:N1]
        self.grid = np.einsum('ij,jklm->imlk', basis, mesh) + \
            origin[:, None, None, None]

        # -- Isolate scalar field data
        del contents[0:num_atoms + 4]
        data1D = np.array(
            [float(val) for line in contents for val in line.split()])
        self.data3D = data1D.reshape((N3, N2, N1), order='F')
