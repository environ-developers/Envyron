from typing import Any, List, Tuple, Union


class Entry:
    """Representation of an input entry."""

    def __init__(
        self,
        section: str,
        name: str,
        dtype: str,
        condition: str = 'True',
        description: str = "",
    ) -> None:
        self.section = section
        self.name = name
        self.dtype = dtype
        self.description = description
        self.__value = None

        self._set_validator(condition)

    @property
    def value(self) -> Any:
        return self.__value

    @value.setter
    def value(self, value: Any) -> None:
        value = self._convert(value)
        self._validate(value)
        self.__value = value

    def _set_validator(self, condition: str) -> None:
        """Set the condition for the value of the Entry."""
        self.valid = eval(f"lambda x: {condition}")

    def _convert(self, value: Any) -> Any:
        """Convert value to expected data type."""
        try:
            if self.dtype == 'str': return value
            if self.dtype == 'int': return int(value)
            if self.dtype == 'float': return float(value)
            if self.dtype == 'bool': return self._boolean(value)
            raise TypeError(f"Unexpected {self.dtype} type")
        except:
            raise TypeError(f"{value} is not of type {self.dtype}")

    def _validate(self, value: Any) -> bool:
        """Check if value is within criteria."""
        if not self.valid(value):
            raise ValueError(f"{value} is invalid for {self.name}")
        return True

    def _boolean(self, value: Union[str, bool]) -> bool:
        """Convert value to boolean."""
        if isinstance(value, bool): return value
        boolean_states = {'true': True, 'false': False}
        boolean = boolean_states.get(value.lower())
        if boolean is None: raise TypeError(f"{value} not a boolean")
        return boolean

    def __str__(self) -> str:
        return f"{self.name} -> {self.description}"


class ArrayEntry(Entry):
    """Representation of an input array entry."""

    def __init__(
        self,
        section: str,
        name: str,
        dtype: str,
        size: int,
        condition: str = 'True',
        description: str = "",
    ) -> None:
        super().__init__(section, name, dtype, condition, description)
        self.size = size

    def _convert(self, values: Any) -> Tuple:
        """Convert value to expected data type."""

        # cast value as array
        if isinstance(values, (list, tuple)):
            pre_conversion = values
        elif isinstance(values, str):
            pre_conversion = values.split()
        elif isinstance(values, (int, float, bool)):
            pre_conversion = [values]
        else:
            raise TypeError("Unexpected type")

        # convert array elements
        converted = []
        for val in pre_conversion:
            converted.append(super()._convert(val))

        n = len(converted)
        if n == 1:
            converted = [converted[0]] * self.size  # extrapolate to size
        else:
            if n != self.size: raise ValueError("Not enough values")

        return tuple(converted)

    def _validate(self, values: Any) -> bool:
        """Check if each value in values is within criteria."""
        for value in values:
            super()._validate(value)
        return True


class Card:
    """Parent class for card input."""

    units = ''

    @classmethod
    def set_units(cls, units: str) -> None:
        """Set Card units."""
        cls.units = units

    def __repr__(self) -> str:
        return f"{self.__dict__}"

    def __str__(self) -> str:
        return f"{self.__dict__}"


class ExternalCard(Card):
    """Input representation of an external charge input."""

    def __init__(
        self,
        charge: float,
        pos: List[str],
        spread: float,
        dim: int,
        axis: int,
    ) -> None:
        self.charge = charge
        self.pos = pos
        self.spread = spread
        self.dim = dim
        self.axis = axis


class RegionCard(Card):
    """Input representation of a dielectric region."""

    def __init__(
        self,
        static: float,
        optical: float,
        pos: List[str],
        width: float,
        spread: float,
        dim: int,
        axis: int,
    ) -> None:
        self.static = static
        self.optical = optical
        self.pos = pos
        self.width = width
        self.spread = spread
        self.dim = dim
        self.axis = axis
