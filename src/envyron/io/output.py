class Output:
    """
    Environ output.
    """
    def __init__(self,
                 is_ionode: bool = False,
                 ionode: int = 0,
                 can_write: bool = False,
                 comm: int = 0,
                 verbosity: int = 0) -> None:

        self.is_ionode = is_ionode
        self.ionode = ionode
        self.can_write = can_write
        self.comm = comm
        self.verbosity = verbosity
