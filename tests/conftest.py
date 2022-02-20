from os import chdir
from pathlib import Path
from pytest import fixture, FixtureRequest


@fixture
def datadir(request: FixtureRequest) -> None:
    """Change over to data directory."""

    # absolute path of running test module
    test_dir = Path(request.fspath.dirname).resolve()

    marker = request.node.get_closest_marker('datadir')

    if marker is None:
        raise ValueError("Missing datadir marker.")
    elif len(marker.args) == 0:
        raise ValueError("Missing name of data directory.")
    else:
        # path to data directory
        data_dir = test_dir.joinpath(marker.args[0])

    if data_dir.exists():
        if not data_dir.is_dir():
            raise ValueError(f"{data_dir} not a directory.")
    else:
        raise ValueError(f"{data_dir} not found.")

    chdir(data_dir)
