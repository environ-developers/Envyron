from .base import InputModel as InputModel
from typing import Any, Optional

class Input:
    param_dict: Any
    params: Any
    def __init__(self, natoms: int, filename: Optional[str] = ...) -> None: ...
    file: Any
    def read(self, filename: str) -> None: ...

def main() -> None: ...
