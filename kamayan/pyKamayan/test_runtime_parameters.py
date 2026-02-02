"""Pytest unit tests for RuntimeParameters."""

import pytest

import kamayan.pyKamayan as pyKamayan


class TestFactoryFunctionIntegration:
    """Test suite for factory function integration with existing functionality."""

    @pytest.fixture
    def params_with_input(self):
        """Create RuntimeParameters from ParameterInput for integration testing."""
        parameter_input = pyKamayan.ParameterInput()
        return pyKamayan.make_runtime_parameters(parameter_input)

    def test_all_parameter_types(self, params_with_input: pyKamayan.RuntimeParameters):
        """Test that all parameter types work with factory function."""
        # Add each parameter type
        params_with_input.add("test_block", "int_param", 42, "Test integer")
        params_with_input.add("test_block", "real_param", 3.14159, "Test real")
        params_with_input.add("test_block", "str_param", "hello", "Test string")
        params_with_input.add("test_block", "bool_param", True, "Test boolean")

        # Test getting each type
        int_val = params_with_input.get_int("test_block", "int_param")
        real_val = params_with_input.get_real("test_block", "real_param")
        str_val = params_with_input.get_str("test_block", "str_param")
        bool_val = params_with_input.get_bool("test_block", "bool_param")

        assert int_val == 42
        assert abs(real_val - 3.14159) < 1e-10
        assert str_val == "hello"
        assert bool_val is True

    def test_assignment_operator_integration(
        self, params_with_input: pyKamayan.RuntimeParameters
    ):
        """Test that assignment operator works with factory-created objects."""
        # Add parameter
        params_with_input.add("test_block", "int_param", 42, "Test parameter")

        # Use set method (which uses our assignment operator)
        params_with_input.set("test_block", "int_param", 100)

        # Verify updated value
        updated_val = params_with_input.get_int("test_block", "int_param")
        assert updated_val == 100

    def test_parameter_persistence(
        self, params_with_input: pyKamayan.RuntimeParameters
    ):
        """Test that parameters persist correctly."""
        # Add parameter
        params_with_input.add("test_block", "persistent_param", 42, "Test persistence")

        # Initial get
        initial_val = params_with_input.get_int("test_block", "persistent_param")
        assert initial_val == 42

        # Multiple sets
        params_with_input.set("test_block", "persistent_param", 100)
        params_with_input.set("test_block", "persistent_param", 200)

        # Final get
        final_val = params_with_input.get_int("test_block", "persistent_param")
        assert final_val == 200


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_runtime_parameters_functionality(self):
        """Test empty RuntimeParameters created via factory function."""
        params = pyKamayan.make_runtime_parameters()

        # Should work normally
        params.add("test_block", "int_param", 42, "Test parameter")
        value = params.get_int("test_block", "int_param")
        assert value == 42

        # Setting should work
        params.set("test_block", "int_param", 100)
        updated_value = params.get_int("test_block", "int_param")
        assert updated_value == 100

    def test_parameter_input_empty_object(self):
        """Test factory function with empty ParameterInput."""
        empty_input = pyKamayan.ParameterInput()
        params = pyKamayan.make_runtime_parameters(empty_input)

        # Should work normally
        params.add("test_block", "int_param", 42, "Test parameter")
        value = params.get_int("test_block", "int_param")
        assert value == 42

    def test_parameter_input_reflection_manual(self):
        """Test ParameterInput reflection manually without accessing pinput."""
        # Create ParameterInput and RuntimeParameters
        pinput = pyKamayan.ParameterInput()
        params = pyKamayan.make_runtime_parameters(pinput)

        # Add parameter to RuntimeParameters
        params.add("test_block", "int_param", 42, "Test parameter")

        # Test that we can access the parameter through ParameterInput directly
        # This tests the reflection capability
        assert "test_block/int_param" in pinput
        reflected_value = pinput["test_block/int_param"]
        assert reflected_value == 42.0 or reflected_value == 42

        # Test bidirectional sync
        params.set("test_block", "int_param", 100)
        updated_value = pinput["test_block/int_param"]
        assert updated_value == 100.0 or updated_value == 100
