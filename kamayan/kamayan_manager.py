from collections.abc import ItemsView
import functools
from pathlib import Path
from typing import Protocol, TypeVar, Type
import sys

# parthenon will gracefully handle mpi already being initialized
from mpi4py import MPI

import kamayan.pyKamayan as pk

COMM = MPI.COMM_WORLD

DataType = TypeVar("DataType", int, float, str, bool)


def get_parm(data_type: Type[DataType], ud: pk.UnitData, key: str) -> DataType:
    """Helper to declare expected type for a parameter."""
    val = ud.Get(key)
    if isinstance(val, data_type):
        return val

    raise TypeError(f"Expected {str(data_type)}, got {str(type(val))}")


def _input_parameters(udc: dict[str, pk.UnitData]) -> list[str]:
    input_blocks = []
    for _, ud in udc.items():
        block = []
        block.append(f"<{ud.Block}>")
        for _, up in ud:
            block.append(f"{up.key} = {str(up.value)}")
        input_blocks.append("\n".join(block))

    return input_blocks


class InputBlock(Protocol):
    """Protocol for objects that can be translated to a parthenon input block."""

    def items(self) -> ItemsView[str, int | float | str | bool]:
        """Should return tuple of parm value pairs to be set."""
        ...


class KamayanParams:
    """Class to manage input parameters throuh UnitDataCollections."""

    def __init__(self, ud: dict[str, pk.UnitData]):
        """Initialize with the UnitDataCollection."""
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


# should take in all the units, be able to process the runtime parameters
# output an input file that tracks the state of all input paramters
# runs and executes the simulation.
class KamayanManager:
    """Manages the an instance of the kamayan simulation."""

    def __init__(self, name: str, units: pk.UnitCollection) -> None:
        """Initialize the manager from a unit collection."""
        self.name = name
        self.rank = COMM.Get_rank()
        self.units = units
        self.input_file = Path(f".{name}.in")
        for name, unit in units:
            if unit.get_SetupParams() is None:
                continue
            unit.get_SetupParams()(unit.unit_data_collection)

    def write_input(self, file: None | Path = None):
        """Write out all the params owned by the unit data collection.

        Format as a parthenon input file.
        """
        if self.rank != 0:
            return
        if file is None:
            file = self.input_file

        with open(file, "w") as fid:
            input_blocks = [
                f"<parthenon/job>\nproblem_id={self.name}"
            ] + _input_parameters(self.units.GetUnitData())
            fid.write("\n".join(input_blocks))

    @functools.cached_property
    def params(self) -> KamayanParams:
        """Get the UnitData for a given input block."""
        return KamayanParams(self.units.GetUnitData())

    def execute(self):
        """Initialize the kamayan environment and execute the simulation."""
        self.write_input()
        # initialize the environment from the previously generated input file
        pman = pk.InitEnv([sys.argv[0], "-i", ".sedov.in"] + sys.argv[1:])
        # get a driver and execute the code
        driver = pk.InitPackages(pman, self.units)
        driver_status = driver.Execute()
        if driver_status != pk.DriverStatus.complete:
            raise RuntimeError("Simulation has not succesfully completed.")

        pman.ParthenonFinalize()
