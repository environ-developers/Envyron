from pathlib import Path
from pytest import mark, raises

from envyron.domains.cell import EnvironGrid

import numpy as np

class TestCell:
    """docstring"""

    L = 20.
    at = np.array([
        [L, 0., 0.],
        [0., L, 0.],
        [0., 0., L],
    ])
    nx = 2
    nr = np.array([nx, nx, nx])
    grid = EnvironGrid(at, nr, label='system')

    def test_cell_attributes(self):
        assert self.grid.label == 'system'
        assert self.grid.volume == self.L**3
        assert self.grid.nnr == self.nx**3

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
