"""Manager for a kamayan simulation."""

from collections.abc import Callable
import functools
from pathlib import Path
import sys

# parthenon will gracefully handle mpi already being initialized
from mpi4py import MPI

import kamayan.pyKamayan as pk
from kamayan.pyKamayan import Grid

from kamayan.code_units.Grid import KamayanGrid
from kamayan.code_units.nodes import Node
from kamayan.code_units.parameters import KamayanParams

COMM = MPI.COMM_WORLD


def _input_parameters(udc: dict[str, pk.UnitData]) -> list[str]:
    input_blocks = []
    for _, ud in udc.items():
        block = []
        block.append(f"<{ud.Block}>")
        for _, up in ud:
            block.append(f"{up.key} = {str(up.value)}")
        input_blocks.append("\n".join(block))

    return input_blocks


SetupInterface = Callable[[pk.UnitDataCollection], None]
InitializeInterface = Callable[[pk.UnitDataCollection], None]
ProblemGeneratorInterface = Callable[[Grid.MeshBlock], None]


def process_units(
    name: str,
    setup_params: SetupInterface | None = None,
    initialize: InitializeInterface | None = None,
    pgen: ProblemGeneratorInterface | None = None,
) -> pk.UnitCollection:
    """Build a default UnitCollection with a user provided KamayanUnit.

    Args:
        name: Name for user generated KamayanUnit
        setup_params: hook into parameter setup
        initialize: hook into package initialization
        pgen: problem generator
    """
    simulation = pk.KamayanUnit(name)
    if setup_params:
        simulation.set_SetupParams(setup_params)
    if initialize:
        simulation.set_InitializeData(initialize)
    if pgen:
        simulation.set_ProblemGeneratorMeshBlock(pgen)

    units = pk.ProcessUnits()
    units.Add(simulation)
    return units


# should take in all the units, be able to process the runtime parameters
# output an input file that tracks the state of all input paramters
# runs and executes the simulation.
class KamayanManager:
    """Manages the an instance of the kamayan simulation."""

    def __init__(self, name: str, units: pk.UnitCollection) -> None:
        """Initialize the manager from a unit collection."""
        # we own the root node rather than inherit from it
        # inheriting seeems to cause a bunch of reference leaks on the nanobind side
        self.root_node = Node()

        self.name = name
        self.rank = COMM.Get_rank()
        self.units = units
        self.input_file = Path(f".{name}.in")
        for name, unit in units:
            if unit.get_SetupParams() is None:
                continue
            unit.get_SetupParams()(unit.unit_data_collection)

        self.grid: KamayanGrid | None = None

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
            ] + _input_parameters(self.params.ud_dict)
            fid.write("\n".join(input_blocks))

    @functools.cached_property
    def params(self) -> KamayanParams:
        """Get the UnitData for a given input block."""
        return KamayanParams(self.units.GetUnitData())

    def execute(self):
        """Initialize the kamayan environment and execute the simulation."""
        for node in self.root_node.get_children():
            node.set_params(self.params)

        self.write_input()
        # initialize the environment from the previously generated input file
        pman = pk.InitEnv([sys.argv[0], "-i", ".sedov.in"] + sys.argv[1:])
        # get a driver and execute the code
        driver = pk.InitPackages(pman, self.units)
        driver_status = driver.Execute()
        if driver_status != pk.DriverStatus.complete:
            raise RuntimeError("Simulation has not succesfully completed.")

        pman.ParthenonFinalize()
