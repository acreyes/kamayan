"""Manager for a kamayan simulation."""

from collections.abc import Callable
import functools
from pathlib import Path
import sys
from typing import Type

# parthenon will gracefully handle mpi already being initialized
from mpi4py import MPI

from code_units import nodes
import kamayan.pyKamayan as pk
from kamayan.pyKamayan import Grid

from kamayan.code_units import driver
from kamayan.code_units.Grid import KamayanGrid
from kamayan.code_units.physics import KamayanPhysics
from kamayan.code_units.outputs import KamayanOutputs
from kamayan.code_units.nodes import Node, AutoProperty
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


SetupInterface = Callable[[pk.KamayanUnit], None]
InitializeInterface = Callable[[pk.KamayanUnit], None]
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


def _auto_property(code_unit: Type[nodes.N], attr: str) -> AutoProperty:
    """Wrap the getter/setter for a code_unit.

    Arguments:
        code_unit: type for the property
        attr: attribute name

    Returns:
        AutoProperty for setting the code unit
    """

    def set_node(self: "KamayanManager", value: Node):
        """When we add a node, add it to the root node."""
        self.root_node.add_child(value)

    return AutoProperty(set_node=set_node)


# should take in all the units, be able to process the runtime parameters
# output an input file that tracks the state of all input paramters
# runs and executes the simulation.
class KamayanManager:
    """Manages the an instance of the kamayan simulation."""

    driver = _auto_property(driver.Driver, "driver")
    physics = _auto_property(KamayanPhysics, "physics")
    grid = _auto_property(KamayanGrid, "grid")
    outputs = _auto_property(KamayanOutputs, "outputs")

    def __init__(self, name: str, units: pk.UnitCollection) -> None:
        """Initialize the manager from a unit collection."""
        # we own the root node rather than inherit from it
        # inheriting seeems to cause a bunch of reference leaks on the nanobind side
        self.root_node = Node()

        self.name = name
        self.rank = COMM.Get_rank()
        self.units = units
        self.input_file = Path(f".{name}.in")
        
        # Call SetupParams to populate UnitData with default parameters
        for unit_name, unit in units:
            if unit.get_SetupParams() is not None:
                unit.get_SetupParams()(unit)

        self._grid: KamayanGrid | None = None
        self.physics = KamayanPhysics()
        self.outputs = KamayanOutputs()

    def write_input(self, file: None | Path = None):
        """Write out all the params owned by the unit data collection.

        Format as a parthenon input file.
        """
        if self.rank != 0:
            return
        if file is None:
            file = self.input_file

        # Gather all UnitData from all units
        all_unit_data: dict[str, pk.UnitData] = {}
        for name, unit in self.units:
            for block_name, ud in unit.AllData().items():
                all_unit_data[block_name] = ud

        # Add blocks that were set directly (not via UnitData)
        new_blocks_str = []
        for block, block_data in self.params.get_new_blocks().items():
            source = block_data.get("source", "unknown")
            params_dict = block_data.get("params", {})
            
            block_lines = [f"# Set by: {source}", f"<{block}>"]
            for key, val in params_dict.items():
                block_lines.append(f"{key} = {val}")
            new_blocks_str.append("\n".join(block_lines))

        with open(file, "w") as fid:
            input_blocks = (
                [f"<parthenon/job>\nproblem_id={self.name}"]
                + _input_parameters(all_unit_data)
                + new_blocks_str
            )
            fid.write("\n".join(input_blocks))

    @functools.cached_property
    def params(self) -> KamayanParams:
        """Get parameters interface for setting overrides."""
        return KamayanParams(self.units)

    def execute(self, *args: str):
        """Initialize the kamayan environment and execute the simulation.

        Args:
            *args: Additional arguments to forward to Parthenon (e.g., parthenon/time/nlim=100)
        """
        for node in self.root_node.get_children():
            node.set_params(self.params)

        self.write_input()

        # Build the argument list for Parthenon
        # Format: [prog, -i, input_file] + extra_args
        parthenon_args = [sys.argv[0], "-i", str(self.input_file)]
        if args:
            parthenon_args.extend(args)
        else:
            # Include any remaining sys.argv arguments not already consumed
            parthenon_args.extend(sys.argv[1:])

        # Temporarily replace sys.argv for Parthenon initialization
        original_argv = sys.argv
        try:
            sys.argv = parthenon_args
            # initialize the environment from the previously generated input file
            pman = pk.InitEnv(sys.argv)
            # get a driver and execute the code
            driver = pk.InitPackages(pman, self.units)
            driver_status = driver.Execute()
            if driver_status != pk.DriverStatus.complete:
                raise RuntimeError("Simulation has not successfully completed.")

            # Use kamayan.Finalize() instead of pman.ParthenonFinalize() to properly
            # clean up lambda callbacks that capture units and break reference cycles
            pk.Finalize(pman)
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
