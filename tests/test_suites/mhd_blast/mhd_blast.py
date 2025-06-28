# Modules
from dataclasses import dataclass
from pathlib import Path

import sys
import utils.test_case

from kamayan.testing import baselines
from parthenon_tools import phdf_diff

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

resolution = 64


@dataclass
class BlastConfig:
    riemann: str = "hll"
    recon: str = "wenoz"
    slope_limiter: str = "mc"
    max_error: float = 1.0e-12


configs = [
    BlastConfig(riemann="hll"),
    BlastConfig(riemann="hllc"),
]


class TestCase(utils.test_case.TestCaseAbs):
    def _test_namer(self, config: BlastConfig) -> str:
        return f"mhd_blast_{config.riemann}"

    def Prepare(self, parameters, step):
        config = configs[step - 1]
        integrator = "rk2"
        parameters.driver_cmd_line_args = [
            f"parthenon/job/problem_id={self._test_namer(config)}",
            f"parthenon/mesh/nx1={resolution}",
            f"parthenon/mesh/nx2={resolution}",
            f"parthenon/meshblock/nx1={resolution / 2}",
            f"parthenon/meshblock/nx2={resolution / 2}",
            "parthenon/mesh/nghost=4",
            f"parthenon/time/integrator={integrator}",
            f"hydro/reconstruction={config.recon}",
            f"hydro/riemann={config.riemann}",
            "parthenon/output0/file_type=hdf5",
            "parthenon/output0/dt=1.0",
            "parthenon/output0/variables=dens,pres,magc_0,magc_1",
            "physics/MHD=ct",
        ]
        return parameters

    def Analyse(self, parameters) -> bool:
        baseline_dir = baselines.get_baseline_dir()
        output_dir = Path(parameters.output_path)

        passing = True
        for config in configs:
            name = self._test_namer(config) + ".out0.final.phdf"
            output_file = output_dir / name
            baseline_file = baseline_dir / name
            delta = phdf_diff.compare(
                [str(output_file), str(baseline_file)],
                check_metadata=False,
                tol=baselines.EPSILON,
                relative=True,
            )
            passing = passing and delta == 0

        return passing
