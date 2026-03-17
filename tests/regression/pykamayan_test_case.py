"""PyKamayan test case base classes."""


class TestCaseAbs:
    """Base class interface (for compatibility)."""

    def Prepare(self, parameters):
        """Prepare method (for compatibility)."""
        raise NotImplementedError("Every TestCase must implement Prepare method")
        return parameters

    def Analyse(self, parameters):
        """Analyse method (for compatibility)."""
        raise NotImplementedError("Every TestCase must implement Analyse method")

    def ErrorOnNonZeroReturnCode(self, parameters, returncode):
        """Check if return code should trigger error."""
        return returncode == 1


class PyKamayanTestCaseBase(TestCaseAbs):
    """Base class for pyKamayan tests.

    This class provides the 3-argument Prepare signature that pyKamayan tests need.

    Example:
        class MyTestCase(PyKamayanTestCaseBase):
            def Prepare(self, parameters, step):
                parameters.driver_cmd_line_args = [...]
                return parameters

            def Analyse(self, parameters):
                # comparison logic
                return True
    """

    def Prepare(self, parameters, step):
        """Configure each run.

        Args:
            parameters: TestParameters object
            step: Current test step number (1-indexed)

        Returns:
            Modified parameters object
        """
        raise NotImplementedError("Every TestCase must implement Prepare method")

    def Analyse(self, parameters) -> bool:
        """Analyze test results.

        Args:
            parameters: TestParameters object

        Returns:
            True if test passes, False otherwise
        """
        raise NotImplementedError("Every TestCase must implement Analyse method")
