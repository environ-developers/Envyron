import pytest
import numpy as np
from dftpy.field import DirectField
from envyron.domains.cell import EnvironGrid
from envyron.representations.field import EnvironField


@pytest.fixture
def environ_grid():
    return EnvironGrid

def test_environ_field_creation(environ_grid):
    environ_field = EnvironField(environ_grid)

    assert isinstance(environ_field, EnvironField)

    assert environ_field.grid == environ_grid

    assert environ_field.data is None

    assert environ_field.label == ''


def test_standard_view(environ_grid):
    environ_field = EnvironField(environ_grid)

    #  No data
    assert environ_field.standard_view() is None


    # 3D data
    environ_field.data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    environ_field.rank = 3
    environ_grid.nnr = 2
    result = environ_field.standard_view()
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2, 2)
    assert np.array_equal(result, np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))




if __name__ == '__main__':
    pytest.main()
