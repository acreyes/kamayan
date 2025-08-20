"""Test the regression infrastructure."""

import pytest
from pathlib import Path

from tests.regression_testing import ManagerType, Case, RegressionParameters

import kamayan.code_units.parameters as parms
import kamayan.pyKamayan as pk


class _KmanMock(ManagerType):
    def execute(
        self, inputs: None | dict[str, parms.data_value] = None
    ) -> pk.DriverStatus:
        return pk.DriverStatus.complete


base_dir = Path("test_output")
test_cases = [
    RegressionParameters(name="test_one", kman=_KmanMock()),
    RegressionParameters(name="test_two", kman=_KmanMock()),
    RegressionParameters(name="test_three", kman=_KmanMock()),
]


@pytest.fixture(params=test_cases, ids=lambda c: c.name)
def parameters(request):
    """Generate the parameters fixture for TestCase."""
    return request.param


class TestRegression(Case):
    """Test the test case machinery."""

    def test_analyse(self, parameters: RegressionParameters):
        """Pass if we made the output directory for each test."""
        assert parameters.output_dir
        assert parameters.output_dir.exists()
