"""Module to manage input parameters in kamayan."""

import inspect
from collections.abc import ItemsView
from typing import TypeVar, Protocol, Any

import kamayan.pyKamayan as pk


data_value = int | float | str | bool
DataType = TypeVar("DataType", int, float, str, bool)


class InputBlock(Protocol):
    """Protocol for objects that can be translated to a parthenon input block."""

    def items(self) -> ItemsView[str, data_value]:
        """Should return dict like ItemsView of parm value pairs to be set."""
        ...


class KamayanParams:
    """Class to manage input parameters through UnitData owned by a unit collection."""

    def __init__(self, units: pk.UnitCollection):
        """Initialize with the UnitCollection."""
        self._units = units
        # Store: {block_name: {"params": {key: val}, "source": "SourceNodeName"}}
        self._new_blocks: dict[str, dict[str, Any]] = {}

    # def __getitem__(self, key: str) -> pk.UnitData:
    #     """Fetch the UnitData for an input block."""
    #     return self.ud_dict[key]

    def get_data(self, unit: str, block: str):
        """Get the unit data from unit that owns block."""
        return self._units.Get(unit).Data(block)

    def __setitem__(self, block: str, value: InputBlock):
        """Set parameters in an input block by dict."""
        # Auto-detect source from call stack
        source = "unknown"
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            if "self" in caller_frame.f_locals:
                caller_self = caller_frame.f_locals["self"]
                source = caller_self.__class__.__name__

        # First, try to find and update existing UnitData
        for unit_name, unit in self._units:
            if unit.HasData(block):
                ud = unit.Data(block)
                for key, val in value.items():
                    if ud.Contains(key):
                        ud.UpdateParm(key, val)
                    else:
                        # Parameter doesn't exist in UnitData, store for direct write
                        if block not in self._new_blocks:
                            self._new_blocks[block] = {"params": {}, "source": source}
                        self._new_blocks[block]["params"][key] = val
                return

        # Block doesn't exist in any UnitData, store for direct write to input file
        if block not in self._new_blocks:
            self._new_blocks[block] = {"params": {}, "source": source}
        for key, val in value.items():
            self._new_blocks[block]["params"][key] = val

    def get_new_blocks(self) -> dict[str, dict[str, Any]]:
        """Get blocks that were added directly (not via UnitData)."""
        return self._new_blocks
