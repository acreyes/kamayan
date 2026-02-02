"""Module to manage input parameters in kamayan."""

from collections.abc import ItemsView
from typing import TypeVar, Protocol

import kamayan.pyKamayan as pk


data_value = int | float | str | bool
DataType = TypeVar("DataType", int, float, str, bool)


class InputBlock(Protocol):
    """Protocol for objects that can be translated to a parthenon input block."""

    def items(self) -> ItemsView[str, data_value]:
        """Should return dict like ItemsView of parm value pairs to be set."""
        ...


class KamayanParams:
    """Class to manage input parameters throuh UnitData owned by a unit collection."""

    def __init__(self, rps: pk.RuntimeParameters):
        """Initialize with the UnitData dict."""
        self._rps = rps

    def __setitem__(self, block: str, value: InputBlock):
        """Set parameters in an input block by dict."""
        for key, val in value.items():
            self._rps.set(block, key, val)
