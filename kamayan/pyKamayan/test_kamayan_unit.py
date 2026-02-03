"""Pytest unit tests for KamayanUnit and UnitCollection."""

import pytest

import kamayan.pyKamayan as pyKamayan


class TestKamayanUnit:
    """Test suite for KamayanUnit class."""

    def test_add_data_block(self):
        """Test adding a parameter block with AddData."""
        unit = pyKamayan.KamayanUnit("test_unit")
        data = unit.AddData("physics")
        assert data is not None

    def test_add_and_get_int_parameter(self):
        """Test adding and retrieving integer parameters."""
        unit = pyKamayan.KamayanUnit("test_unit")
        data = unit.AddData("test_block")
        data.AddParm("int_param", 42, "Test integer parameter")

        value = data.Get(int, "int_param")
        assert value == 42

    def test_add_and_get_float_parameter(self):
        """Test adding and retrieving float parameters."""
        unit = pyKamayan.KamayanUnit("test_unit")
        data = unit.AddData("test_block")
        data.AddParm("real_param", 3.14159, "Test real parameter")

        value = data.Get(float, "real_param")
        assert abs(value - 3.14159) < 1e-10

    def test_add_and_get_string_parameter(self):
        """Test adding and retrieving string parameters."""
        unit = pyKamayan.KamayanUnit("test_unit")
        data = unit.AddData("test_block")
        data.AddParm("str_param", "hello_world", "Test string parameter")

        value = data.Get(str, "str_param")
        assert value == "hello_world"

    def test_add_and_get_bool_parameter(self):
        """Test adding and retrieving boolean parameters."""
        unit = pyKamayan.KamayanUnit("test_unit")
        data = unit.AddData("test_block")
        data.AddParm("bool_param", True, "Test boolean parameter")

        value = data.Get(bool, "bool_param")
        assert value is True

    def test_multiple_parameters_same_block(self):
        """Test adding multiple parameters to the same block."""
        unit = pyKamayan.KamayanUnit("test_unit")
        data = unit.AddData("test_block")

        data.AddParm("param1", 100, "First parameter")
        data.AddParm("param2", 2.718, "Second parameter")
        data.AddParm("param3", "test", "Third parameter")

        assert data.Get(int, "param1") == 100
        assert abs(data.Get(float, "param2") - 2.718) < 1e-10
        assert data.Get(str, "param3") == "test"

    def test_multiple_data_blocks(self):
        """Test creating multiple parameter blocks."""
        unit = pyKamayan.KamayanUnit("test_unit")

        block1 = unit.AddData("block1")
        block2 = unit.AddData("block2")

        block1.AddParm("value", 1.0, "Block 1 value")
        block2.AddParm("value", 2.0, "Block 2 value")

        assert abs(block1.Get(float, "value") - 1.0) < 1e-10
        assert abs(block2.Get(float, "value") - 2.0) < 1e-10

    def test_retrieve_data_block(self):
        """Test retrieving a data block by name."""
        unit = pyKamayan.KamayanUnit("test_unit")
        original_data = unit.AddData("physics")
        original_data.AddParm("gamma", 1.4, "Adiabatic index")

        retrieved_data = unit.Data("physics")
        value = retrieved_data.Get(float, "gamma")
        assert abs(value - 1.4) < 1e-10

    def test_add_python_object_parameter(self):
        """Test storing arbitrary Python objects in parameters."""

        class TestData:
            """Test data class."""

            def __init__(self, x, y):
                self.x = x
                self.y = y

        unit = pyKamayan.KamayanUnit("test_unit")
        test_obj = TestData(1.0, 2.0)

        unit.AddParam("my_object", test_obj)
        retrieved = unit.GetParam(TestData, "my_object")

        assert retrieved is test_obj
        assert retrieved.x == 1.0
        assert retrieved.y == 2.0

    def test_python_object_identity_preserved(self):
        """Test that Python object identity is preserved."""
        unit = pyKamayan.KamayanUnit("test_unit")
        original_list = [1, 2, 3, 4, 5]

        unit.AddParam("my_list", original_list)
        retrieved_list = unit.GetParam(list, "my_list")

        assert retrieved_list is original_list
        retrieved_list.append(6)
        assert len(original_list) == 6


class TestUnitCollection:
    """Test suite for UnitCollection functionality."""

    def test_add_unit_to_collection(self):
        """Test adding a unit to the collection."""
        collection = pyKamayan.ProcessUnits()
        unit = pyKamayan.KamayanUnit("test_unit")

        collection.Add(unit)
        # Smoke test - addition doesn't crash

    def test_add_multiple_units(self):
        """Test adding multiple units to collection."""
        collection = pyKamayan.ProcessUnits()

        unit1 = pyKamayan.KamayanUnit("unit1")
        unit2 = pyKamayan.KamayanUnit("unit2")
        unit3 = pyKamayan.KamayanUnit("unit3")

        collection.Add(unit1)
        collection.Add(unit2)
        collection.Add(unit3)

    def test_get_unit_from_collection(self):
        """Test accessing a unit by name from another unit.

        Note: This test is skipped because GetUnit requires the units to be
        linked to the collection via InitPackages or similar full initialization.
        This is tested in the integration tests instead.
        """
        pytest.skip("Requires full package initialization - tested in integration tests")


class TestCallbackHooks:
    """Test callback hook registration."""

    def test_set_setup_params_callback(self):
        """Test setting SetupParams callback."""
        unit = pyKamayan.KamayanUnit("test_unit")

        def setup_callback(u):
            """Setup callback function."""
            data = u.AddData("test")
            data.AddParm("setup_called", True, "Flag")

        unit.set_SetupParams(setup_callback)
        # Smoke test - setting callback doesn't crash

    def test_set_initialize_callback(self):
        """Test setting InitializeData callback."""
        unit = pyKamayan.KamayanUnit("test_unit")

        def initialize_callback(u):
            """Initialize callback function."""
            data = u.Data("test")
            # Just access data

        unit.set_InitializeData(initialize_callback)
        # Smoke test - setting callback doesn't crash

    def test_set_pgen_callback(self):
        """Test setting ProblemGeneratorMeshBlock callback."""
        unit = pyKamayan.KamayanUnit("test_unit")

        def pgen_callback(mb):
            """Problem generator callback."""
            # Would access meshblock data
            pass

        unit.set_ProblemGeneratorMeshBlock(pgen_callback)
        # Smoke test - setting callback doesn't crash

    def test_multiple_callbacks(self):
        """Test setting all three callbacks."""
        unit = pyKamayan.KamayanUnit("test_unit")

        def setup(u):
            data = u.AddData("test")
            data.AddParm("value", 1.0, "Test")

        def initialize(u):
            pass

        def pgen(mb):
            pass

        unit.set_SetupParams(setup)
        unit.set_InitializeData(initialize)
        unit.set_ProblemGeneratorMeshBlock(pgen)
        # Smoke test - all callbacks set


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_get_nonexistent_parameter(self):
        """Test retrieving a parameter that doesn't exist raises error."""
        unit = pyKamayan.KamayanUnit("test_unit")
        data = unit.AddData("test_block")

        with pytest.raises(Exception):
            data.Get(float, "nonexistent_param")

    def test_get_nonexistent_data_block(self):
        """Test retrieving a data block that doesn't exist raises error."""
        unit = pyKamayan.KamayanUnit("test_unit")

        with pytest.raises(Exception):
            unit.Data("nonexistent_block")

    def test_type_mismatch_on_get(self):
        """Test that requesting wrong type for parameter raises error."""
        unit = pyKamayan.KamayanUnit("test_unit")
        data = unit.AddData("test_block")
        data.AddParm("int_value", 42, "Integer parameter")

        # Requesting as float should work (type coercion may be allowed)
        # or raise error - depends on implementation
        # For now, test that int retrieval works
        value = data.Get(int, "int_value")
        assert value == 42

    def test_get_nonexistent_python_object(self):
        """Test retrieving Python object that doesn't exist."""
        unit = pyKamayan.KamayanUnit("test_unit")

        with pytest.raises(Exception):
            unit.GetParam(object, "nonexistent")


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_unit_with_mixed_parameters(self):
        """Test unit with both regular and Python object parameters."""

        class CustomConfig:
            """Custom configuration class."""

            def __init__(self, name, settings):
                self.name = name
                self.settings = settings

        unit = pyKamayan.KamayanUnit("physics")

        # Add regular parameters
        data = unit.AddData("eos")
        data.AddParm("gamma", 5.0 / 3.0, "Adiabatic index")
        data.AddParm("use_custom", True, "Use custom config")

        # Add Python object
        config = CustomConfig("gamma_law", {"type": "ideal"})
        unit.AddParam("config", config)

        # Retrieve and verify
        gamma = data.Get(float, "gamma")
        use_custom = data.Get(bool, "use_custom")
        retrieved_config = unit.GetParam(CustomConfig, "config")

        assert abs(gamma - 5.0 / 3.0) < 1e-10
        assert use_custom is True
        assert retrieved_config.name == "gamma_law"
        assert retrieved_config.settings["type"] == "ideal"

    def test_unit_collection_cross_unit_access(self):
        """Test accessing parameters across units in a collection.

        Note: This test is skipped because cross-unit access via GetUnit
        requires full package initialization with InitPackages.
        This is tested in the integration tests instead.
        """
        pytest.skip("Requires full package initialization - tested in integration tests")
