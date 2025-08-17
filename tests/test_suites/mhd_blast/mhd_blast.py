"""MHD Blast regression test."""

# Modules
from dataclasses import dataclass
from pathlib import Path

import sys
import utils.test_case

import numpy as np

from kamayan.testing import baselines
from parthenon_tools import phdf_diff, compare_analytic

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

resolution = 64


@dataclass
class BlastConfig:
    """Configuration for running blast in different modes."""

    riemann: str = "hll"
    recon: str = "wenoz"
    slope_limiter: str = "mc"
    max_error: float = 1.0e-12
    resolution: int = 64
    nxb: int = 32
    numlevel: int = 1


configs = [
    BlastConfig(riemann="hll"),
    BlastConfig(riemann="hllc"),
    BlastConfig(resolution=32, nxb=8, numlevel=3, max_error=1.0e-10),
]


def analytic_divb(Z, Y, X, t):
    """Returns 0.0 for B-field divergence everywhere."""
    return np.zeros_like(Z)


class TestCase(utils.test_case.TestCaseAbs):
    """Test class for sedov."""

    def _test_namer(self, config: BlastConfig) -> str:
        return (
            f"mhd_blast_{config.riemann}_N{config.resolution}_"
            f"n{config.nxb}_l{config.numlevel}"
        )

    def Prepare(self, parameters, step):
        """Configure each run."""
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
            f"hydro/reconstruction={config.recon}",
            f"hydro/riemann={config.riemann}",
            "parthenon/output0/file_type=hdf5",
            "parthenon/output0/dt=1.0",
            "parthenon/output0/variables=dens,pres,magc_0,magc_1",
            # there should be a better way to compare separate vars...
            "parthenon/output1/file_type=hdf5",
            "parthenon/output1/dt=1.0",
            "parthenon/output1/variables=divb",
            "physics/MHD=ct",
        ]
        return parameters

    def Analyse(self, parameters) -> bool:
        """Determine success of test cases."""
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
                tol=config.max_error,
                relative=True,
            )
            # error wrt to gold files
            passing = passing and delta == 0

            name = self._test_namer(config) + ".out1.final.phdf"
            output_file = output_dir / name
            # l2 norm of divb
            passing = passing and compare_analytic.compare_analytic(
                str(output_file), {"divb": analytic_divb}, tol=1.0e-10
            )

            # linf norm of divb
            def linf_norm(gold, test):
                return compare_analytic.norm_err_func(gold, test, norm_ord=2)

            passing = passing and compare_analytic.compare_analytic(
                str(output_file),
                {"divb": analytic_divb},
                err_func=linf_norm,
                tol=1.0e-10,
            )

        return passing
