"""Sedov pyKamayan regression test."""

from dataclasses import dataclass
from pathlib import Path

import sys
import utils.test_case

from kamayan.testing import baselines
from parthenon_tools import phdf_diff

sys.dont_write_bytecode = True


@dataclass
class SedovConfig:
    """Configuration for running sedov in different modes."""

    riemann: str = "hllc"
    resolution: int = 64
    nxb: int = 32
    numlevel: int = 1


configs = [
    SedovConfig(riemann="hllc"),
    SedovConfig(riemann="hllc"),
    SedovConfig(riemann="hllc"),
    SedovConfig(riemann="hllc"),
]


class TestCase(utils.test_case.TestCaseAbs):
    """Test class for sedov pyKamayan."""

    def Prepare(self, parameters, step):
        """Configure each run."""
        config = configs[step - 1]
        parameters.driver_cmd_line_args = [
            f"parthenon/mesh/nx1={config.resolution}",
            f"parthenon/mesh/nx2={config.resolution}",
            f"parthenon/meshblock/nx1={config.nxb}",
            f"parthenon/meshblock/nx2={config.nxb}",
            f"parthenon/mesh/numlevel={config.numlevel}",
            f"hydro/riemann={config.riemann}",
            "hydro/ReconstructionStrategy=scratchpad",
            "parthenon/output0/file_type=hdf5",
            "parthenon/output0/dt=1.0",
            "parthenon/output0/variables=dens,pres",
        ]
        return parameters

    def Analyse(self, parameters) -> bool:
        """Determine success of test cases."""
        baseline_dir = baselines.get_baseline_dir()
        output_dir = Path(parameters.output_path)

        output_file = output_dir / "sedov.out0.final.phdf"
        baseline_file = baseline_dir / "sedov.out0.final.phdf"
        delta = phdf_diff.compare(
            [str(output_file), str(baseline_file)],
            check_metadata=False,
            tol=baselines.EPSILON,
            relative=True,
        )

        return delta == 0
