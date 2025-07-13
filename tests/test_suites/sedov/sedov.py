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
class SedovConfig:
    riemann: str = "hll"
    recon: str = "wenoz"
    slope_limiter: str = "mc"
    max_error: float = 1.0e-12
    resolution: int = 64
    nxb: int = 32
    numlevel: int = 1


configs = [
    SedovConfig(riemann="hll"),
    SedovConfig(riemann="hllc"),
    SedovConfig(resolution=32, nxb=8, numlevel=3),
]


class TestCase(utils.test_case.TestCaseAbs):
    def _test_namer(self, config: SedovConfig) -> str:
        return (
            f"sedov_{config.riemann}_N{config.resolution}_"
            f"n{config.nxb}_l{config.numlevel}"
        )

    def Prepare(self, parameters, step):
        config = configs[step - 1]
        integrator = "rk2"
        refinement = "none"
        if config.numlevel > 1:
            refinement = "adaptive"
        parameters.driver_cmd_line_args = [
            f"parthenon/job/problem_id={self._test_namer(config)}",
            f"parthenon/mesh/refinement={refinement}",
            f"parthenon/mesh/nx1={config.resolution}",
            f"parthenon/mesh/nx2={config.resolution}",
            f"parthenon/meshblock/nx1={config.nxb}",
            f"parthenon/meshblock/nx2={config.nxb}",
            f"parthenon/mesh/numlevel={config.numlevel}",
            "parthenon/mesh/nghost=4",
            f"parthenon/time/integrator={integrator}",
            f"hydro/riemann={config.riemann}",
            "parthenon/output0/file_type=hdf5",
            "parthenon/output0/dt=1.0",
            "parthenon/output0/variables=dens,pres",
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
