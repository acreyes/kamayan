"""Module to manage input parameters in kamayan."""

from collections.abc import ItemsView
from typing import TypeVar, Protocol

import kamayan.pyKamayan as pk


data_value = int | float | str | bool
DataType = TypeVar("DataType", int, float, str, bool)


class InputBlock(Protocol):
    """Protocol for objects that can be translated to a parthenon input block."""

    def items(self) -> ItemsView[str, int | float | str | bool]:
        """Should return dict like ItemsView of parm value pairs to be set."""
        ...


class KamayanParams:
    """Class to manage input parameters throuh UnitDataCollections."""

    def __init__(self, ud: dict[str, pk.UnitData]):
        """Initialize with the UnitData dict."""
        self.ud_dict = ud

    def __getitem__(self, key: str) -> pk.UnitData:
        """Fetch the UnitData for an input block."""
        return self.ud_dict[key]

    def __setitem__(self, key: str, value: InputBlock):
        """Set parameters in an input block by dict."""
        if key in self.ud_dict.keys():
            ud = self.ud_dict[key]
            for k, val in value.items():
                ud[k] = val
        else:
            ud = pk.UnitData(key)
            for k, val in value.items():
                ud.AddParm(k, val, "")
            self.ud_dict.update({key: ud})
