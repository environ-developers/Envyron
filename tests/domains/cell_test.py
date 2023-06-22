from pathlib import Path
from pytest import fixture, mark, approx, raises

from envyron.domains.cell import EnvironGrid

import numpy as np


# TODO: add test to check label
# TODO: add test to check units
# TODO: add test to check ecut and guess function
class TestMinimalCell:
    """docstring"""

    def test_cell_attributes(self, minimal_cell):
        """docstring"""
        assert minimal_cell.label == ''
        assert minimal_cell.volume == 1.
        assert minimal_cell.nnr == 8
        assert minimal_cell.dV == 1 / 8

    def test_get_reciprocal(self, minimal_cell):
        """docstring"""
        reciprocal_cell = minimal_cell.get_reciprocal()
        assert (reciprocal_cell.cell == np.eye(3) * 2 * np.pi).all()
        assert (reciprocal_cell.nnrG == 8)

    def test_cell_corners(self, minimal_cell):
        """docstring"""
        assert (minimal_cell.corners[0] == np.array([0, 0, 0])).all()
        assert (minimal_cell.corners[1] == np.array([0, 0, -1])).all()
        assert (minimal_cell.corners[2] == np.array([0, -1, 0])).all()
        assert (minimal_cell.corners[3] == np.array([0, -1, -1])).all()
        assert (minimal_cell.corners[4] == np.array([-1, 0, 0])).all()
        assert (minimal_cell.corners[5] == np.array([-1, 0, -1])).all()
        assert (minimal_cell.corners[6] == np.array([-1, -1, 0])).all()
        assert (minimal_cell.corners[7] == np.array([-1, -1, -1])).all()


@mark.parametrize('unitary_cell, n', [(3, 3), (10, 10), (100, 100)],
                  indirect=['unitary_cell'])
class TestUnitCell:
    """docstring"""

    def test_unitary_cell_attributes(self, unitary_cell, n):
        """docstring"""
        assert unitary_cell.label == ''
        assert unitary_cell.volume == 1.
        assert unitary_cell.nnr == n**3
        assert unitary_cell.dV == approx(1. / n**3)

    def test_get_reciprocal(self, unitary_cell, n):
        """docstring"""
        reciprocal_cell = unitary_cell.get_reciprocal()
        assert (reciprocal_cell.cell == np.eye(3) * 2 * np.pi).all()
        assert (reciprocal_cell.nnrG == n**3)


@mark.parametrize('cubic_cell, n, L', [((2, 20), 2, 20), ((3, 20), 3, 20),
                                       ((10, 50), 10, 50)],
                  indirect=['cubic_cell'])
class TestCubicCell:
    """docstring"""

    def test_cubic_cell_attributes(self, cubic_cell, n, L):
        """docstring"""
        assert cubic_cell.label == ''
        assert cubic_cell.volume == approx(L**3)
        assert cubic_cell.nnr == n**3
        assert cubic_cell.dV == approx(L**3 / n**3)

    def test_get_reciprocal(self, cubic_cell, n, L):
        """docstring"""
        reciprocal_cell = cubic_cell.get_reciprocal()
        assert (reciprocal_cell.cell == np.eye(3) * 2 * np.pi / L).all()
        assert (reciprocal_cell.nnrG == n**3)

    def test_cell_corners(self, cubic_cell, n, L):
        """docstring"""
        assert (cubic_cell.corners[0] == np.array([0, 0, 0])).all()
        assert (cubic_cell.corners[1] == np.array([0, 0, -L])).all()
        assert (cubic_cell.corners[2] == np.array([0, -L, 0])).all()
        assert (cubic_cell.corners[3] == np.array([0, -L, -L])).all()
        assert (cubic_cell.corners[4] == np.array([-L, 0, 0])).all()
        assert (cubic_cell.corners[5] == np.array([-L, 0, -L])).all()
        assert (cubic_cell.corners[6] == np.array([-L, -L, 0])).all()
        assert (cubic_cell.corners[7] == np.array([-L, -L, -L])).all()


@mark.parametrize('hexagonal_cell, n, a, c_over_a', [((2, 1, 1), 2, 1, 1),
                                                     ((2, 20, 3), 2, 20, 3),
                                                     ((10, 20, 3), 10, 20, 3)],
                  indirect=['hexagonal_cell'])
class TestHexagonalCell:
    """docstring"""

    def test_hexagonal_cell_attributes(self, hexagonal_cell, n, a, c_over_a):
        """docstring"""
        assert hexagonal_cell.label == ''
        assert hexagonal_cell.volume == approx(a**3 * np.sqrt(3) * 0.5 *
                                               c_over_a)
        assert hexagonal_cell.nnr == approx(n**3 * c_over_a)
        assert hexagonal_cell.dV == approx(a**3 * np.sqrt(3) * 0.5 / n**3)

    def test_get_reciprocal(self, hexagonal_cell, n, a, c_over_a):
        """docstring"""
        expected_cell = np.eye(3) * 2 * np.pi / a
        expected_cell[0, 1] = -2 * np.pi / a / np.sqrt(3.)
        expected_cell[1, 1] *= 2. / np.sqrt(3.)
        expected_cell[2, 2] *= 1 / c_over_a
        reciprocal_cell = hexagonal_cell.get_reciprocal()
        assert np.allclose(expected_cell, reciprocal_cell.cell.real)
        assert (reciprocal_cell.nnrG == approx(n**3 * c_over_a))


@mark.parametrize(
    'shift', [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 3, 4)]
)  # the minimum distance should not depend on an arbitrary shift by a lattice vector
class TestMinDistance():

    @mark.parametrize('dim', [(0), (1), (2)])
    @mark.parametrize('axis', [(0), (1), (2)])
    def test_minimal_cell(self, minimal_cell, shift, dim, axis):
        origin = np.array([0., 0.2, 0.9]) + np.dot(minimal_cell.cell.T,
                                                   np.array(shift))
        expected_r = np.array([
            [0.0, -0.2, 0.1],
            [0.5, -0.2, 0.1],
            [0.0, 0.3, 0.1],
            [0.5, 0.3, 0.1],
            [0.0, -0.2, -0.4],
            [0.5, -0.2, -0.4],
            [0.0, 0.3, -0.4],
            [0.5, 0.3, -0.4],
        ])
        if dim == 1:
            expected_r[:, axis] = 0.
        elif dim == 2:
            expected_r[:, np.arange(3) != axis] = 0.
        expected_r2 = np.sum(expected_r**2, axis=1)
        obtained_r = minimal_cell.get_min_distance(origin, dim,
                                                   axis)[0].T.reshape(
                                                       minimal_cell.nnr, 3)
        obtained_r2 = minimal_cell.get_min_distance(
            origin, dim, axis)[1].T.reshape(minimal_cell.nnr)
        assert np.allclose(expected_r, obtained_r)
        assert np.allclose(expected_r2, obtained_r2)

    @mark.parametrize('hexagonal_cell', [(2, 1, 1)],
                      indirect=['hexagonal_cell'])
    @mark.parametrize('dim', [(0), (1), (2)])
    @mark.parametrize('axis', [(0), (1), (2)])
    def test_hexagonal_cell(self, hexagonal_cell, shift, dim, axis):
        origin = np.array([1.1, 0.1, 0.]) + np.dot(hexagonal_cell.cell.T,
                                                   np.array(shift))
        r = np.array([
            [-0.1, -0.1, 0.],
            [0.4, -0.1, 0.],
            [0.15, 0.3330127, 0.],
            [-0.35, 0.3330127, 0.],
            [-0.1, -0.1, 0.5],
            [0.4, -0.1, 0.5],
            [0.15, 0.3330127, 0.5],
            [-0.35, 0.3330127, 0.5],
        ])

        if axis == 0:
            if dim == 1:
                n = np.array([1, 0, 0])
            elif dim == 2:
                n = np.array([np.sqrt(3) * 0.5, -0.5, 0])
                # for this case two of the closest point change
                r[3, :] = [0.65, 0.3330127, 0.]
                r[7, :] = [0.65, 0.3330127, 0.5]
        elif axis == 1:
            if dim == 1:
                n = np.array([0.5, np.sqrt(3) * 0.5, 0])
                # for this case two of the closest point change
                r[3, :] = [0.65, 0.3330127, 0.]
                r[7, :] = [0.65, 0.3330127, 0.5]
            elif dim == 2:
                n = np.array([0, 1, 0])
        elif axis == 2:
            n = np.array([0, 0, 1])

        if dim == 0:
            expected_r = r
        elif dim == 1:
            expected_r = r - np.einsum('j,i->ji', np.einsum('ji,i->j', r, n),
                                       n)
        elif dim == 2:
            expected_r = np.einsum('j,i->ji', np.einsum('ji,i->j', r, n), n)

        expected_r2 = np.sum(expected_r**2, axis=1)
        obtained_r = hexagonal_cell.get_min_distance(origin, dim,
                                                     axis)[0].T.reshape(
                                                         hexagonal_cell.nnr, 3)
        obtained_r2 = hexagonal_cell.get_min_distance(
            origin, dim, axis)[1].T.reshape(hexagonal_cell.nnr)

        print(expected_r)
        print(obtained_r)
        assert np.allclose(expected_r, obtained_r)
        assert np.allclose(expected_r2, obtained_r2)
