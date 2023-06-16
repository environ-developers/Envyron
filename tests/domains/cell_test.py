from pathlib import Path
from pytest import mark, approx, raises

from envyron.domains.cell import EnvironGrid

import numpy as np

class TestMinimalCell:
    """docstring"""
    def test_cell_attributes(self,minimal_cell):
        assert minimal_cell.label == ''
        assert minimal_cell.volume == 1.
        assert minimal_cell.nnr == 8
        assert minimal_cell.dV == 1/8

@mark.parametrize('unitary_cell, n', [(2, 2), (3, 3), (50, 50)], indirect=['unitary_cell'])
class TestUnitCell:
    """docstring"""
    def test_unitary_cell_attributes(self,unitary_cell,n):
        assert unitary_cell.label == ''
        assert unitary_cell.volume == 1.
        assert unitary_cell.nnr == n**3
        assert unitary_cell.dV == approx(1./n**3)

@mark.parametrize('cubic_cell, n, L', [((2, 20), 2, 20), ((3, 20), 3, 20), ((10, 50), 10, 50)], indirect=['cubic_cell'])
class TestCubicCell:
    """docstring"""
    def test_cubic_cell_attributes(self,cubic_cell,n,L):
        assert cubic_cell.label == ''
        assert cubic_cell.volume == approx(L**3)
        assert cubic_cell.nnr == n**3
        assert cubic_cell.dV == approx(L**3/n**3)

    """ add test to check units errors """
    """ add test to check ecut and guess function """
    """ add test to check reciprocal grid """

"""
    def test_cell_corners(self):
        assert (self.grid.corners[0] == np.array([0., 0., 0.])).all()
        assert (self.grid.corners[1] == -self.at[:, 2]).all()
        assert (self.grid.corners[2] == -self.at[:, 1]).all()
        assert (self.grid.corners[3] == -self.at[:, 2] - self.at[:, 1]).all()
        assert (self.grid.corners[4] == -self.at[:, 0]).all()
        assert (self.grid.corners[5] == -self.at[:, 0] - self.at[:, 2]).all()
        assert (self.grid.corners[6] == -self.at[:, 0] - self.at[:, 1]).all()
        assert (self.grid.corners[7] == -self.at[:, 0] - self.at[:, 1] -
                self.at[:, 2]).all()

    def test_get_min_distance(self):
        origin = np.array([0., 2., 1.])
        results_r = np.array([
            [0., -2., -1.],
            [10., -2., -1.],
            [0., 8., -1.],
            [10., 8., -1.],
            [0., -2., 9.],
            [10., -2., 9.],
            [0., 8., 9.],
            [10., 8., 9.],
        ])
        assert (self.grid.get_min_distance(origin)[0].T.reshape(
            self.grid.nnr, 3) == results_r).all()
        results_r2 = np.array([
            5.,
            105.,
            65.,
            165.,
            85.,
            185.,
            145.,
            245.,
        ])
        assert (self.grid.get_min_distance(origin)[1].T.reshape(
            self.grid.nnr) == results_r2).all()
"""