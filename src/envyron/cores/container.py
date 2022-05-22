from typing import Optional

from .core import NumericalCore


class CoreContainer:
    """docstring"""

    def __init__(
        self,
        label: str,
        has_internal_correction: bool = False,
        derivatives_core: Optional[NumericalCore] = None,
        electrostatics_core: Optional[NumericalCore] = None,
        corrections_core: Optional[NumericalCore] = None,
    ) -> None:
        self.label = label
        self.has_internal_correction = has_internal_correction

        self._derivatives = None
        self.has_derivatives = False

        self._electrostatics = None
        self.has_electrostatics = False
        
        self._corrections = None
        self.has_corrections = False

        if derivatives_core is not None:
            self.derivatives = derivatives_core

        if electrostatics_core is not None:
            self.electrostatics = electrostatics_core

        if corrections_core is not None:
            self.corrections = corrections_core

    @property
    def derivatives(self) -> NumericalCore:
        """docstring"""
        return self._derivatives

    @derivatives.setter
    def derivatives(self, core: NumericalCore) -> None:
        """docstring"""
        _typecheck(core)
        self._derivatives = core
        self.has_derivatives = True

    @property
    def electrostatics(self) -> NumericalCore:
        """docstring"""
        return self._electrostatics

    @electrostatics.setter
    def electrostatics(self, core: NumericalCore) -> None:
        """docstring"""
        _typecheck(core)
        self._electrostatics = core
        self.has_electrostatics = True

    @property
    def corrections(self) -> NumericalCore:
        """docstring"""
        return self._corrections

    @corrections.setter
    def corrections(self, core: NumericalCore) -> None:
        """docstring"""
        _typecheck(core)
        self._corrections = core
        self.has_corrections = True


def _typecheck(core: NumericalCore) -> None:
    """docstring"""
    if not isinstance(core, NumericalCore):
        raise ValueError(f"{type(core)} is not a `NumericalCore` object")
