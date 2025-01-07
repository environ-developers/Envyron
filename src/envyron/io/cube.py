# Refactored from Stephen Weitzner cube_vizkit
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#
from ase import Atoms
from ase.units import Bohr
from dataclasses import dataclass, field
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

    def __init__(self, fname='', units='Bohr'):
        self.read(fname, units)

    def read(self, fname='', units='Bohr'):
        """
        load(cube_file)

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
        self.basis = np.array([R1, R2, R3], dtype='d').T  # store vectors as columns
        self.scalars = np.array([N1, N2, N3], dtype='d')
        self.cell = self.basis * self.scalars  # broadcasting

        # -- Create an ASE Atoms object
        tmp = np.array([line.split() for line in header[3:]], dtype='d')
        numbers = tmp[:, 0].astype(int)
        charges = tmp[:, 1]
        positions = tmp[:, 2:]
        if self.units == 'Bohr':
            positions = positions * Bohr
        self.atoms = Atoms(numbers=numbers,
                           positions=positions,
                           charges=charges,
                           cell=self.cell.T)

        # -- Construct the grid
        mesh = np.mgrid[0:N1, 0:N2, 0:N3]
        self.grid = np.einsum('ij,jklm->iklm', self.basis, mesh) + \
            origin[:, None, None, None]

        # -- Isolate scalar field data
        del contents[0:num_atoms + 4]
        self.data1D = np.array(
            [float(val) for line in contents for val in line.split()])
        self.data3D = self.data1D.reshape((N3, N2, N1), order='F').T

    def toline(self,center,axis,planaraverage=False):
        icenter = np.array([ np.rint(center[i]/self.basis[i,i]) for i in range(3)],dtype='int')
        icenter = icenter - (self.scalars * np.trunc(icenter//self.scalars)).astype('int')
        if axis == 0 :
            axis = self.grid[0,:,icenter[1],icenter[2]]
            if planaraverage :
                value = np.mean(self.data3D,(1,2))
            else:
                value = self.data3D[:,icenter[1],icenter[2]]
        elif axis == 1 :
            axis = self.grid[1,icenter[0],:,icenter[2]]
            if planaraverage :
                value = np.mean(self.data3D,(0,2))
            else :
                value = self.data3D[icenter[0],:,icenter[2]]
        elif axis == 2 :
            axis = self.grid[2,icenter[0],icenter[1],:]
            if planaraverage :
                value = np.mean(self.data3D,(0,1))
            else :
                value = self.data3D[icenter[0],icenter[1],:]
        else:
            raise ValueError('Axis out of range')
        return axis, value

    def tocontour(self,center,axis):
        icenter = np.array([ np.rint(center[i]/self.basis[i,i]) for i in range(3)],dtype='int')
        icenter = icenter - (self.scalars * np.trunc(icenter//self.scalars)).astype('int')
        if axis == 0 :
            ax1 = self.grid[1,icenter[0],:,:]
            ax2 = self.grid[2,icenter[0],:,:]
            value = self.data3D[icenter[0],:,:]
        elif axis == 1 :
            ax1 = self.grid[0,:,icenter[1],:]
            ax2 = self.grid[2,:,icenter[1],:]
            value = self.data3D[:,icenter[1],:]
        elif axis == 2 :
            ax1 = self.grid[0,:,:,icenter[2]]
            ax2 = self.grid[1,:,:,icenter[2]]
            value = self.data3D[:,:,icenter[2]]
        else:
            raise ValueError('Axis out of range')
        return ax1, ax2, value
    
    def plotprojections(self,center:np.ndarray[np.float64],colormap='plasma',centermap=False):
        cmap=mpl.colormaps[colormap]
        axis1_yz, axis2_yz, values_yz = self.tocontour(center,0)
        axis1_xz, axis2_xz, values_xz = self.tocontour(center,1)
        axis1_xy, axis2_xy, values_xy = self.tocontour(center,2)
        width_x = self.cell[0,0] # NEED TO FIX FOR NON ORTHOROMBIC CELLS
        width_y = self.cell[1,1] # NEED TO FIX FOR NON ORTHOROMBIC CELLS
        width_z = self.cell[2,2] # NEED TO FIX FOR NON ORTHOROMBIC CELLS
        vmin = np.min(self.data3D)
        vmax = np.max(self.data3D)
        # The following is an option for centering the colorbar on zero
        if centermap :
            vmax = -np.max([abs(np.min(self.data3D)),abs(np.max(self.data3D))]) * 0.6
            vmin = -vmax
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 2,  width_ratios=(width_x, width_z), height_ratios=(width_z, width_y),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

        ax1 = fig.add_subplot(gs[0, 0])
        cont1 = ax1.contourf(axis1_xz,axis2_xz,values_xz,levels=100,cmap=cmap,vmin=vmin,vmax=vmax)
        ax1.scatter(center[0],center[2])
        ax1.tick_params(labelbottom=False)
        ax1.set_ylabel('Z (a.u.)')

        ax3 = fig.add_subplot(gs[1, 0],sharex=ax1)
        cont3 = ax3.contourf(axis1_xy,axis2_xy,values_xy,levels=100,cmap=cmap,vmin=vmin,vmax=vmax)
        ax3.scatter(center[0],center[1])
        ax3.set_xlabel('X (a.u.)')
        ax3.set_ylabel('Y (a.u.)')

        ax4 = fig.add_subplot(gs[1, 1],sharey=ax3)
        cont4 = ax4.contourf(axis2_yz,axis1_yz,values_yz,levels=100,cmap=cmap,vmin=vmin,vmax=vmax)
        ax4.scatter(center[2],center[1])
        ax4.tick_params(labelleft=False)
        ax4.set_xlabel('Z (a.u.)')

        # Colorbar 
        ax2 = fig.add_subplot(gs[0, 1])
        ax2_pos = ax2.get_position().bounds
        ax2.set_position([ax2_pos[0]+ax2_pos[2]*0.35,ax2_pos[1]+ax2_pos[3]*0.05,ax2_pos[2]*0.1,ax2_pos[3]*0.9])
        fig.colorbar(cont4, cax=ax2)
        plt.show()