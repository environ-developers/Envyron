from pathlib import Path
from pytest import mark, raises, approx

from envyron.representations.density import EnvironDensity

import numpy as np


@mark.parametrize('N', [(1.), (2.), (10.)])
class TestUniformDensity:
    """docstring"""

    def test_minimal_charge(self, minimal_cell, N, uniform_density):
        """"""
        density = uniform_density(minimal_cell, N)
        assert density.N == N
        assert density.integral() == N
        assert density.norm() == N

    def test_minimal_gradient(self, minimal_cell, N, uniform_density):
        """"""
        density = uniform_density(minimal_cell, N)
        assert np.sum(density.gradient()) == 0.

    @mark.parametrize('unitary_cell', [(3), (10), (100)],
                      indirect=['unitary_cell'])
    def test_unitary_charge(self, unitary_cell, N, uniform_density):
        """"""
        density = uniform_density(unitary_cell, N)
        assert density.N == N
        assert density.integral() == N
        assert density.norm() == N

    @mark.parametrize('unitary_cell', [(3), (10), (20)],
                      indirect=['unitary_cell'])
    def test_unitary_gradient(self, unitary_cell, N, uniform_density):
        """"""
        density = uniform_density(unitary_cell, N)
        assert np.sum(density.gradient()) == approx(0.)

    @mark.parametrize('cubic_cell', [(2, 20), (10, 20), (20, 30)],
                      indirect=['cubic_cell'])
    def test_cubic_charge(self, cubic_cell, N, uniform_density):
        """"""
        density = uniform_density(cubic_cell, N)
        assert density.N == N * cubic_cell.volume
        assert density.integral() == N * cubic_cell.volume
        assert density.norm() == approx(np.sqrt(N**2 * cubic_cell.volume))

    @mark.parametrize('cubic_cell', [(2, 20), (10, 20), (20, 30)],
                      indirect=['cubic_cell'])
    def test_cubic_gradient(self, cubic_cell, N, uniform_density):
        """"""
        density = uniform_density(cubic_cell, N)
        assert np.sum(density.gradient()) == approx(0.)

    @mark.parametrize('hexagonal_cell', [(2, 1, 1), (2, 20, 3), (10, 20, 3)],
                      indirect=['hexagonal_cell'])
    def test_hexagonal_charge(self, hexagonal_cell, N, uniform_density):
        """"""
        density = uniform_density(hexagonal_cell, N)
        assert density.N == N * hexagonal_cell.volume
        assert density.integral() == N * hexagonal_cell.volume
        assert density.norm() == approx(np.sqrt(N**2 * hexagonal_cell.volume))

    @mark.parametrize('hexagonal_cell', [(2, 1, 1), (2, 20, 3), (10, 20, 3)],
                      indirect=['hexagonal_cell'])
    def test_hexagonal_gradient(self, hexagonal_cell, N, uniform_density):
        """"""
        density = uniform_density(hexagonal_cell, N)
        assert np.sum(density.gradient()) == approx(0.)