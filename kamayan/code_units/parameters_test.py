"""Pytest unit tests for KamayanParams."""

import pytest

import kamayan.pyKamayan as pyKamayan
from kamayan.code_units.parameters import KamayanParams


class TestKamayanParams:
    """Test suite for KamayanParams functionality."""

    def test_create_with_unit_collection(self):
        """Test creating KamayanParams with a UnitCollection."""
        collection = pyKamayan.ProcessUnits()
        params = KamayanParams(collection)
        assert params is not None

    def test_get_data_from_unit(self):
        """Test retrieving UnitData from a unit in the collection."""
        collection = pyKamayan.ProcessUnits()
        unit = pyKamayan.KamayanUnit("test_unit")
        data = unit.AddData("test_block")
        data.AddParm("value", 42, "Test value")
        collection.Add(unit)

        params = KamayanParams(collection)
        retrieved_data = params.get_data("test_unit", "test_block")

        value = retrieved_data.Get(int, "value")
        assert value == 42

    def test_set_parameter_in_existing_block(self):
        """Test setting parameters via dict when block exists."""
        collection = pyKamayan.ProcessUnits()
        unit = pyKamayan.KamayanUnit("test_unit")
        data = unit.AddData("test_block")
        data.AddParm("param1", 1.0, "Parameter 1")
        data.AddParm("param2", 2.0, "Parameter 2")
        collection.Add(unit)

        params = KamayanParams(collection)

        # Update existing parameters
        params["test_block"] = {"param1": 10.0, "param2": 20.0}

        # Verify parameters were updated
        retrieved_data = params.get_data("test_unit", "test_block")
        assert abs(retrieved_data.Get(float, "param1") - 10.0) < 1e-10
        assert abs(retrieved_data.Get(float, "param2") - 20.0) < 1e-10

    def test_set_new_parameter_in_existing_block(self):
        """Test adding new parameters to existing block via dict.

        When a parameter doesn't exist in UnitData but the block does,
        it gets stored in _new_blocks.
        """
        collection = pyKamayan.ProcessUnits()
        unit = pyKamayan.KamayanUnit("test_unit")
        data = unit.AddData("test_block")
        data.AddParm("existing_param", 1.0, "Existing parameter")
        collection.Add(unit)

        params = KamayanParams(collection)

        # Add new parameter to existing block
        params["test_block"] = {"existing_param": 10.0, "new_param": 99.0}

        # Verify existing param was updated
        retrieved_data = params.get_data("test_unit", "test_block")
        assert abs(retrieved_data.Get(float, "existing_param") - 10.0) < 1e-10

        # Verify new param is in new_blocks
        new_blocks = params.get_new_blocks()
        assert "test_block" in new_blocks
        assert new_blocks["test_block"]["params"]["new_param"] == 99.0

    def test_set_parameters_in_new_block(self):
        """Test setting parameters when block doesn't exist in any UnitData."""
        collection = pyKamayan.ProcessUnits()
        params = KamayanParams(collection)

        # Set parameters in a completely new block
        params["new_block"] = {"param1": 1.0, "param2": "test", "param3": True}

        # Verify stored in new_blocks
        new_blocks = params.get_new_blocks()
        assert "new_block" in new_blocks
        assert new_blocks["new_block"]["params"]["param1"] == 1.0
        assert new_blocks["new_block"]["params"]["param2"] == "test"
        assert new_blocks["new_block"]["params"]["param3"] is True

    def test_multiple_new_blocks(self):
        """Test adding multiple new parameter blocks."""
        collection = pyKamayan.ProcessUnits()
        params = KamayanParams(collection)

        params["block1"] = {"value": 1.0}
        params["block2"] = {"value": 2.0}
        params["block3"] = {"value": 3.0}

        new_blocks = params.get_new_blocks()
        assert len(new_blocks) == 3
        assert new_blocks["block1"]["params"]["value"] == 1.0
        assert new_blocks["block2"]["params"]["value"] == 2.0
        assert new_blocks["block3"]["params"]["value"] == 3.0

    def test_update_existing_block_multiple_times(self):
        """Test updating the same block multiple times."""
        collection = pyKamayan.ProcessUnits()
        unit = pyKamayan.KamayanUnit("test_unit")
        data = unit.AddData("test_block")
        data.AddParm("value", 1.0, "Test value")
        collection.Add(unit)

        params = KamayanParams(collection)

        # Update multiple times
        params["test_block"] = {"value": 10.0}
        params["test_block"] = {"value": 20.0}
        params["test_block"] = {"value": 30.0}

        # Verify final value
        retrieved_data = params.get_data("test_unit", "test_block")
        assert abs(retrieved_data.Get(float, "value") - 30.0) < 1e-10


class TestKamayanParamsIntegration:
    """Integration tests for KamayanParams with multiple units."""

    def test_params_with_multiple_units(self):
        """Test parameter management with multiple units."""
        collection = pyKamayan.ProcessUnits()

        # Create first unit
        unit1 = pyKamayan.KamayanUnit("unit1")
        data1 = unit1.AddData("config1")
        data1.AddParm("value", 1.0, "Value 1")
        collection.Add(unit1)

        # Create second unit
        unit2 = pyKamayan.KamayanUnit("unit2")
        data2 = unit2.AddData("config2")
        data2.AddParm("value", 2.0, "Value 2")
        collection.Add(unit2)

        params = KamayanParams(collection)

        # Update both blocks
        params["config1"] = {"value": 10.0}
        params["config2"] = {"value": 20.0}

        # Verify both were updated
        retrieved_data1 = params.get_data("unit1", "config1")
        retrieved_data2 = params.get_data("unit2", "config2")

        assert abs(retrieved_data1.Get(float, "value") - 10.0) < 1e-10
        assert abs(retrieved_data2.Get(float, "value") - 20.0) < 1e-10

    def test_params_mixed_existing_and_new(self):
        """Test mix of updating existing blocks and creating new ones."""
        collection = pyKamayan.ProcessUnits()

        unit = pyKamayan.KamayanUnit("test_unit")
        data = unit.AddData("existing_block")
        data.AddParm("param", 1.0, "Parameter")
        collection.Add(unit)

        params = KamayanParams(collection)

        # Update existing and add new
        params["existing_block"] = {"param": 100.0}
        params["new_block_1"] = {"value": 1}
        params["new_block_2"] = {"value": 2}

        # Verify existing updated
        retrieved_data = params.get_data("test_unit", "existing_block")
        assert abs(retrieved_data.Get(float, "param") - 100.0) < 1e-10

        # Verify new blocks stored
        new_blocks = params.get_new_blocks()
        assert len(new_blocks) == 2
        assert new_blocks["new_block_1"]["params"]["value"] == 1
        assert new_blocks["new_block_2"]["params"]["value"] == 2

    def test_params_with_process_units_defaults(self):
        """Test KamayanParams with default units from ProcessUnits."""
        collection = pyKamayan.ProcessUnits()
        params = KamayanParams(collection)

        # ProcessUnits creates default units: driver, eos, grid, physics, hydro
        # We can access their data if they have any default blocks

        # Add a custom unit
        custom_unit = pyKamayan.KamayanUnit("custom")
        custom_data = custom_unit.AddData("custom_block")
        custom_data.AddParm("custom_param", 42, "Custom parameter")
        collection.Add(custom_unit)

        # Update custom block
        params["custom_block"] = {"custom_param": 84}

        # Verify
        retrieved_data = params.get_data("custom", "custom_block")
        assert retrieved_data.Get(int, "custom_param") == 84


class TestErrorHandling:
    """Test error handling in KamayanParams."""

    def test_get_data_nonexistent_unit(self):
        """Test getting data from a unit that doesn't exist."""
        collection = pyKamayan.ProcessUnits()
        params = KamayanParams(collection)

        with pytest.raises(Exception):
            params.get_data("nonexistent_unit", "some_block")

    def test_get_data_nonexistent_block(self):
        """Test getting data block that doesn't exist in unit."""
        collection = pyKamayan.ProcessUnits()
        unit = pyKamayan.KamayanUnit("test_unit")
        collection.Add(unit)

        params = KamayanParams(collection)

        with pytest.raises(Exception):
            params.get_data("test_unit", "nonexistent_block")
