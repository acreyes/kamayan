"""Regression testing infrastructure.

Needs to support a few types of workflows:
    * Run arbitrary instantiations of the same problem n-number of times
        * compare against baseline files
        * convergence to some solution
        * arbitrary success criteria

pytest + mpi???
    * https://pytest-mpi.readthedocs.io/en/latest/usage.html
    * see also :https://github.com/firedrakeproject/mpi-pytest
        * seems to be more recently maintained

TestCase:
    * consume a KamayanManager
        * determine output directory / where does this test live relative to where it is run
        * serialize the input parameters?
    * run simulation
    * determine success
        * success if runs without error, or with expected error
        * can provide a default comparison utility for baselines

TestCollection:
    * sequence of test-cases to run
        * some unique way of identifying these test cases
            * user provided dict[str | int, TestCase]?
    * run tests in sequence
    * determine success
        * can provide a convergence study utility
            * determines convergence rate
        * some generic mechanism to compare all results
"""

from dataclasses import dataclass
from pathlib import Path
import pytest
from typing import Protocol

import kamayan.code_units.parameters as parms
import kamayan.pyKamayan as pk


class ManagerType:
    """Looks like a KamayanManager."""

    def execute(
        self, inputs: None | dict[str, parms.data_value] = None
    ) -> pk.DriverStatus:
        """Run the simulation."""
        ...


@dataclass
class RegressionParameters:
    """Hold all the parameters needed to run a test case."""

    name: str
    kman: ManagerType
    inputs: None | dict[str, parms.data_value] = None
    output_dir: None | Path = None
    success: bool = False


class Case:
    """Run a single test."""

    def test_run(self, parameters: RegressionParameters):
        """Run our test."""
        if not parameters.output_dir:
            for i in range(100):
                output_dir = Path(f"test_output/{parameters.name}_{i:02d}")
                if output_dir.exists():
                    continue
                parameters.output_dir = output_dir
                break
        assert parameters.output_dir, "Need to make an empty output directory."
        parameters.output_dir.mkdir(parents=True)

        status = parameters.kman.execute(inputs=parameters.inputs)
        assert status == pk.DriverStatus.complete

    def test_analyse(self, parameters: RegressionParameters):
        """Analyse the output."""
        raise NotImplementedError("test analysis not implemented.")
