from .validation import InputModel as InputModel
from typing import Any

class Input:
    file: Any
    params: Any
    def __init__(self, natoms: int, filename: str = ...) -> None: ...

def main() -> None: ...
