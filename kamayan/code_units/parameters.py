"""Module to manage input parameters in kamayan."""

from collections.abc import ItemsView
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, Protocol
import weakref

import kamayan.pyKamayan as pk


data_value = int | float | str | bool
DataType = TypeVar("DataType", int, float, str, bool)


class InputBlock(Protocol):
    """Protocol for objects that can be translated to a parthenon input block."""

    def items(self) -> ItemsView[str, int | float | str | bool]:
        """Should return dict like ItemsView of parm value pairs to be set."""
        ...


param_t = TypeVar("param_t", str, int, float, bool)
T = TypeVar("T", contravariant=True)


class ParmSetter(Protocol[T]):
    """Interface for mapping input parameter types to param_t."""

    def __call__(self, parm_in: T) -> param_t:
        """Input a generic type and return a param_t."""
        ...


def _default_mapper(parm_in: param_t) -> param_t:
    return parm_in


class KamayanParam(Generic[param_t, T]):
    """Wrapper to set runtime parameters in a UnitData.

    * Parameters are owned by KamayanUnits.
    * Each unit has a parameters dict that maps input parameter blocks
      to the corresponding UnitData
    * UnitData maps runtime parameters to the public facing params
      owned by the unit
    """

    def __init__(
        self,
        unit: str,
        key: str,
        default: param_t,
        mapper: ParmSetter = _default_mapper,
    ):
        """Initialize the parameter."""
        self._unit_str = unit
        self._block, self._key = self._get_parm_name(key)
        self._default = default
        self._unit_data: Optional[pk.UnitData] = None
        self._mapper = mapper

    def _get_parm_name(self, parm: str) -> tuple[str, str]:
        """Extract the block and key from a full parthenon input parameter."""
        separator = "/"
        splits = parm.split(separator)
        key = splits.pop()
        block = separator.join(splits)
        return block, key

    def set_unit_data(self, unit_collection: pk.UnitCollection):
        """Set the unit data that owns this parameter."""
        assert not self._unit_data
        self._unit_data = unit_collection.Get(self._unit_str).Data(self._block)

    def set(self, value: T):
        """Sets our parameter in the unit_data."""
        assert self._unit_data
        self._unit_data.UpdateParm(self._mapper(value))

    def get(self) -> param_t:
        """Gets the parameter from the unit_data."""
        assert self._unit_data
        return self._unit_data.Get(type(self._default), self._key)


class KamayanParams:
    """Class to manage input parameters throuh UnitData owned by a unit collection."""

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
                if not ud.Contains(k):
                    ud.AddParm(k, val, "")
                ud[k] = val
        else:
            ud = pk.UnitData(key)
            for k, val in value.items():
                ud.AddParm(k, val, "")
            self.ud_dict.update({key: ud})
