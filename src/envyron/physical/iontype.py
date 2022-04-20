from typing import Union


class EnvironIonType:
    """
    docstring
    """

    elements = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
        "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
        "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
        "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
        "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
        "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
        "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
        "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U"
    ]

    pauling = [
        1.2, 0.0, 0.0, 0.0, 0.0, 1.5, 1.5, 1.4, 1.35, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.9, 1.85, 1.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.15, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]

    weights = [
        1.00794, 4.002602, 6.941, 9.012182, 10.811, 12.0107, 14.0067, 15.9994,
        18.9984032, 20.1797, 22.98977, 24.305, 26.981538, 28.0855, 30.973761,
        32.065, 35.453, 39.948, 39.0983, 40.078, 44.95591, 47.867, 50.9415,
        51.9961, 54.938049, 58.6934, 55.845, 58.9332, 63.546, 65.409, 69.723,
        72.64, 74.9216, 78.96, 79.904, 83.798, 85.4678, 87.62, 88.90585,
        91.224, 92.90638, 95.94, 98.0, 101.07, 102.9055, 106.42, 107.8682,
        112.411, 114.818, 118.71, 121.76, 127.6, 126.90447, 131.293, 132.90545,
        137.327, 138.9055, 140.116, 140.90766, 144.24, 145.0, 150.36, 151.964,
        157.25, 158.925354, 162.5, 164.930328, 167.259, 168.934218, 173.045,
        174.9668, 178.49, 180.9479, 183.84, 186.207, 190.23, 192.217, 195.078,
        196.96655, 200.59, 204.3833, 207.2, 208.98038, 209.0, 210.0, 222.0,
        223.0, 226.0, 227.03, 232.04, 231.04, 238.02891
    ]

    bondi = [
        1.2, 1.4, 1.82, 1.85, 1.8, 1.7, 1.55, 1.52, 1.47, 1.54, 2.27, 1.73,
        2.3, 2.1, 1.8, 1.8, 1.75, 1.88, 2.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.63, 1.4, 1.39, 1.87, 2.19, 1.85, 1.9, 0.0, 2.02, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.63, 1.72, 1.58, 1.93, 2.17, 0.0,
        2.06, 1.98, 2.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.75,
        1.66, 1.55, 1.96, 2.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.86
    ]

    uff = [
        2.886, 2.362, 2.451, 2.745, 4.083, 3.851, 3.66, 3.5, 3.364, 3.243,
        2.983, 3.021, 4.499, 4.295, 4.147, 4.035, 3.947, 3.868, 3.812, 3.399,
        3.295, 3.175, 3.144, 3.023, 2.961, 2.912, 2.872, 2.834, 3.495, 2.763,
        4.383, 4.28, 4.23, 4.205, 4.189, 4.141, 4.114, 3.641, 3.345, 3.124,
        3.165, 3.052, 2.998, 2.963, 2.929, 2.899, 3.148, 2.848, 4.463, 4.392,
        4.42, 4.47, 4.5, 4.404, 4.517, 3.703, 3.522, 3.556, 3.606, 3.575,
        3.547, 3.52, 3.493, 3.368, 3.451, 3.428, 3.409, 3.391, 3.374, 3.355,
        3.64, 3.141, 3.17, 3.069, 2.954, 3.12, 2.84, 2.754, 3.293, 2.705,
        4.337, 4.297, 4.379, 4.709, 4.75, 4.765, 4.9, 3.677, 3.478, 3.396,
        3.424, 3.395
    ]

    muff = [
        2.886, 2.362, 2.451, 2.745, 4.083, 3.851, 3.1, 3.5, 3.364, 3.243,
        2.983, 3.021, 4.499, 4.295, 4.147, 4.035, 3.947, 3.868, 3.812, 3.399,
        3.295, 3.175, 3.144, 3.023, 2.961, 2.912, 2.872, 2.834, 3.495, 2.763,
        4.383, 4.28, 4.23, 4.205, 4.189, 4.141, 4.114, 3.641, 3.345, 3.124,
        3.165, 3.052, 2.998, 2.963, 2.929, 2.899, 3.148, 2.848, 4.463, 4.392,
        4.42, 4.47, 4.5, 4.404, 4.517, 3.703, 3.522, 3.556, 3.606, 3.575,
        3.547, 3.52, 3.493, 3.368, 3.451, 3.428, 3.409, 3.391, 3.374, 3.355,
        3.64, 3.141, 3.17, 3.069, 2.954, 3.12, 2.84, 2.754, 3.293, 2.705,
        4.337, 4.297, 4.379, 4.709, 4.75, 4.765, 4.9, 3.677, 3.478, 3.396,
        3.424, 3.395
    ]

    def __init__(
        self,
        index: int,
        ion_id: Union[str, int, float],
        zv: float,
        radius_mode: str,
        atomicspread: float,
        corespread: float,
        solvationrad: float,
    ) -> None:
        self.index = index
        self.zv = -zv

        self._set_ion_id(ion_id)

        self._set_ion_defaults(radius_mode.lower())

        if atomicspread > 0: self.atomicspread = atomicspread

        if self.label == 'H':
            self.corespread = 1e-10
        else:
            self.corespread = corespread

        if solvationrad > 0: self.solvationrad = solvationrad

    def _set_ion_id(self, ion_id: Union[str, int, float]) -> None:
        """docstring"""

        if isinstance(ion_id, str):
            self.label = ion_id.capitalize()
            self.number = self._get_atomic_number_by_label(ion_id)
        elif isinstance(ion_id, int):
            self.label = self.elements[ion_id - 1]
            self.number = ion_id
        elif isinstance(ion_id, float):
            self.number = self._get_atomic_number_by_weight(ion_id)
            self.label = self.elements[self.number - 1]
        else:
            raise TypeError("ion id must be given as number, label, or weight")

        self.weight = self.weights[self.number - 1]

    def _get_atomic_number_by_label(self, atom_id: str) -> int:
        """docstring"""
        for i, element in enumerate(self.elements):
            if element == atom_id:
                return i + 1
        raise ValueError(f"{element} does not match any element")

    def _get_atomic_number_by_weight(self, atom_id: float) -> int:
        """docstring"""
        for i, weight in enumerate(self.weights):
            if abs(weight - atom_id) < 1e-2:
                return i + 1
        raise ValueError(f"{weight} does not match any atomic weight")

    def _set_ion_defaults(self, radius_mode: str) -> None:
        """docstring"""
        self.atomicspread = 0.5
        self.corespread = 0.5

        if radius_mode == 'pauling':
            self.solvationrad = self.pauling[self.number]
        elif radius_mode == 'bondi':
            self.solvationrad = self.bondi[self.number]
        elif radius_mode == 'uff':
            self.solvationrad = self.uff[self.number]
        elif radius_mode == 'muff':
            self.solvationrad = self.muff[self.number]
        else:
            raise ValueError(f"{radius_mode} is not a supported radius mode")