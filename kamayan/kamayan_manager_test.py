"""Pytest unit tests for KamayanManager integration."""

from pathlib import Path

from kamayan.kamayan_manager import KamayanManager, process_units
from kamayan.code_units.Grid import AdaptiveGrid
from kamayan.code_units.driver import Driver
from kamayan.code_units.eos import GammaEos
from kamayan.code_units.Hydro import Hydro


class TestProcessUnits:
    """Test suite for process_units function."""

    def test_process_units_with_name_only(self):
        """Test creating units with just a name."""
        units = process_units("test_simulation")
        assert units is not None

    def test_process_units_with_setup_callback(self):
        """Test process_units with setup callback."""

        def setup(unit):
            data = unit.AddData("test_block")
            data.AddParm("param1", 1.0, "Test parameter")

        units = process_units("test_simulation", setup_params=setup)
        # Smoke test - creation doesn't crash

    def test_process_units_with_initialize_callback(self):
        """Test process_units with initialize callback."""

        def setup(unit):
            data = unit.AddData("test_block")
            data.AddParm("param1", 1.0, "Test parameter")

        def initialize(unit):
            data = unit.Data("test_block")
            # Just access the data

        units = process_units(
            "test_simulation", setup_params=setup, initialize=initialize
        )

    def test_process_units_with_pgen_callback(self):
        """Test process_units with pgen callback."""

        def pgen(mb):
            # Problem generator would access mesh block
            pass

        units = process_units("test_simulation", pgen=pgen)

    def test_process_units_with_all_callbacks(self):
        """Test process_units with all callbacks."""

        def setup(unit):
            data = unit.AddData("test_block")
            data.AddParm("value", 42, "Value")

        def initialize(unit):
            data = unit.Data("test_block")

        def pgen(mb):
            pass

        units = process_units(
            "test_simulation",
            setup_params=setup,
            initialize=initialize,
            pgen=pgen,
        )


class TestKamayanManager:
    """Test KamayanManager initialization and basic properties."""

    def test_kamayan_manager_has_units(self):
        """Test that KamayanManager stores unit collection."""
        units = process_units("test")
        km = KamayanManager("test", units)

        assert km.units is units

    def test_kamayan_manager_has_root_node(self):
        """Test that KamayanManager creates a root node."""
        units = process_units("test")
        km = KamayanManager("test", units)

        assert km.root_node is not None

    def test_kamayan_manager_has_params_property(self):
        """Test that KamayanManager has params property."""
        units = process_units("test")
        km = KamayanManager("test", units)

        assert km.params is not None

    def test_kamayan_manager_input_file_path(self):
        """Test that KamayanManager creates correct input file path."""
        units = process_units("my_simulation")
        km = KamayanManager("my_simulation", units)

        assert km.input_file == Path(".my_simulation.in")

    def test_setup_params_called_on_init(self):
        """Test that SetupParams is called during KamayanManager init."""
        setup_called = []

        def setup(unit):
            setup_called.append(True)
            data = unit.AddData("test_block")
            data.AddParm("value", 1.0, "Value")

        units = process_units("test", setup_params=setup)
        km = KamayanManager("test", units)

        # SetupParams should have been called
        assert len(setup_called) == 1


class TestKamayanManagerConfiguration:
    """Test configuration of KamayanManager components."""

    def test_set_grid_adaptive(self):
        """Test setting an adaptive grid.

        Uses AdaptiveGrid with explicit block specification to avoid
        MPI decomposition issues in serial tests.
        """
        units = process_units("test")
        km = KamayanManager("test", units)

        grid = AdaptiveGrid(
            xbnd1=(-0.5, 0.5),
            xbnd2=(-0.5, 0.5),
            nxb1=32,
            nxb2=32,
            num_levels=3,
            nblocks1=4,
            nblocks2=4,
        )
        km.grid = grid

        assert km.grid is grid

    def test_set_driver(self):
        """Test setting driver configuration."""
        units = process_units("test")
        km = KamayanManager("test", units)

        drv = Driver(integrator="rk2", tlim=1.0)
        km.driver = drv

        assert km.driver is drv
        assert km.driver.tlim == 1.0

    def test_set_eos(self):
        """Test setting EOS configuration."""
        units = process_units("test")
        km = KamayanManager("test", units)

        eos = GammaEos(gamma=5.0 / 3.0, mode_init="dens_pres")
        km.physics.eos = eos

        assert km.physics.eos is eos
        assert abs(km.physics.eos.gamma - 5.0 / 3.0) < 1e-10

    def test_set_hydro(self):
        """Test setting Hydro configuration."""
        units = process_units("test")
        km = KamayanManager("test", units)

        hydro = Hydro(reconstruction="plm", riemann="hllc")
        km.physics.hydro = hydro

        assert km.physics.hydro is hydro
        assert km.physics.hydro.reconstruction == "plm"
        assert km.physics.hydro.riemann == "hllc"

    def test_set_multiple_components(self):
        """Test setting multiple configuration components."""
        units = process_units("test")
        km = KamayanManager("test", units)

        # Configure everything (skip grid due to MPI decomposition in serial)
        km.driver = Driver(integrator="rk2", tlim=0.5)
        km.physics.eos = GammaEos(gamma=1.4)
        km.physics.hydro = Hydro(reconstruction="wenoz", riemann="hllc")

        # Verify all set correctly
        assert km.driver.tlim == 0.5
        assert abs(km.physics.eos.gamma - 1.4) < 1e-10
        assert km.physics.hydro.reconstruction == "wenoz"


class TestKamayanManagerParameters:
    """Test parameter management in KamayanManager."""

    def test_set_custom_parameters(self):
        """Test setting custom parameters via params dict."""

        def setup(unit):
            data = unit.AddData("custom_block")
            data.AddParm("param1", 1.0, "Parameter 1")
            data.AddParm("param2", 2.0, "Parameter 2")

        units = process_units("test", setup_params=setup)
        km = KamayanManager("test", units)

        # Set parameters
        km.params["custom_block"] = {"param1": 10.0, "param2": 20.0}

        # Verify parameters were set
        data = km.params.get_data("test", "custom_block")
        assert abs(data.Get(float, "param1") - 10.0) < 1e-10
        assert abs(data.Get(float, "param2") - 20.0) < 1e-10

    def test_set_new_parameter_block(self):
        """Test setting a completely new parameter block."""
        units = process_units("test")
        km = KamayanManager("test", units)

        # Add new block
        km.params["new_block"] = {"value1": 100, "value2": "test"}

        # Verify stored in new_blocks
        new_blocks = km.params.get_new_blocks()
        assert "new_block" in new_blocks
        assert new_blocks["new_block"]["params"]["value1"] == 100
        assert new_blocks["new_block"]["params"]["value2"] == "test"


class TestKamayanManagerWriteInput:
    """Test input file generation."""

    def test_write_input_creates_file(self):
        """Test that write_input creates an input file."""
        units = process_units("test_write")
        km = KamayanManager("test_write", units)

        km.driver = Driver(integrator="rk2", tlim=1.0)

        # Write to a test file
        test_file = Path("test_input.in")
        try:
            km.write_input(test_file)

            # Verify file was created
            assert test_file.exists()

            # Read and verify basic content
            content = test_file.read_text()
            assert "<parthenon/job>" in content
            assert "problem_id=test_write" in content

        finally:
            # Clean up
            if test_file.exists():
                test_file.unlink()

    def test_write_input_includes_custom_parameters(self):
        """Test that write_input includes custom parameters."""

        def setup(unit):
            data = unit.AddData("test_block")
            data.AddParm("test_param", 42.0, "Test parameter")

        units = process_units("test_params", setup_params=setup)
        km = KamayanManager("test_params", units)

        km.params["test_block"] = {"test_param": 84.0}

        test_file = Path("test_params.in")
        try:
            km.write_input(test_file)

            content = test_file.read_text()
            assert "<test_block>" in content
            assert "test_param" in content

        finally:
            if test_file.exists():
                test_file.unlink()

    def test_write_input_includes_new_blocks(self):
        """Test that write_input includes new parameter blocks."""
        units = process_units("test_new")
        km = KamayanManager("test_new", units)

        km.params["completely_new_block"] = {"value": 123}

        test_file = Path("test_new.in")
        try:
            km.write_input(test_file)

            content = test_file.read_text()
            assert "<completely_new_block>" in content
            assert "value = 123" in content

        finally:
            if test_file.exists():
                test_file.unlink()


class TestKamayanManagerIntegration:
    """Integration tests for complete setup workflow."""

    def test_complete_simulation_setup(self):
        """Test a complete simulation setup similar to sedov."""

        def setup(unit):
            data = unit.AddData("simulation")
            data.AddParm("density", 1.0, "ambient density")
            data.AddParm("pressure", 1.0e-5, "ambient pressure")
            data.AddParm("energy", 1.0, "explosion energy")

        def initialize(unit):
            data = unit.Data("simulation")
            dens = data.Get(float, "density")
            # Would do calculations here
            assert dens == 1.0

        units = process_units(
            "complete_test", setup_params=setup, initialize=initialize
        )
        km = KamayanManager("complete_test", units)

        # Full configuration
        km.grid = AdaptiveGrid(
            xbnd1=(-0.5, 0.5),
            xbnd2=(-0.5, 0.5),
            nxb1=32,
            nxb2=32,
            nblocks1=2,
            nblocks2=2,
            num_levels=2,
        )
        km.driver = Driver(integrator="rk2", tlim=0.05)
        km.physics.eos = GammaEos(gamma=5.0 / 3.0, mode_init="dens_pres")
        km.physics.hydro = Hydro(reconstruction="wenoz", riemann="hllc")
        km.params["simulation"] = {
            "density": 1.0,
            "pressure": 1.0e-5,
            "energy": 1.0,
        }

        # Write input file
        test_file = Path("complete_test.in")
        try:
            km.write_input(test_file)

            # Verify file exists and has content
            assert test_file.exists()
            content = test_file.read_text()
            assert "problem_id=complete_test" in content
            assert "<simulation>" in content

        finally:
            if test_file.exists():
                test_file.unlink()

    def test_node_tree_structure(self):
        """Test that configuration builds proper node tree."""
        units = process_units("tree_test")
        km = KamayanManager("tree_test", units)

        km.driver = Driver(integrator="rk2", tlim=1.0)
        km.physics.eos = GammaEos(gamma=1.4)
        km.physics.hydro = Hydro(reconstruction="plm", riemann="hllc")

        # Verify children were added to root node
        children = km.root_node.get_children()
        assert len(children) > 0

        # Driver and physics should be in tree
        child_types = [type(child).__name__ for child in children]
        assert "Driver" in child_types
        assert "KamayanPhysics" in child_types
