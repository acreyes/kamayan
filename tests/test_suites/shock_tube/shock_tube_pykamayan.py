"""Shock tube pyKamayan regression test."""

from dataclasses import dataclass
from pathlib import Path

import sys

sys.path.insert(
    0, "/Users/adamreyes/Documents/research/repos/kamayan/trees/sparse/tests/regression"
)
from pykamayan_test_case import PyKamayanTestCaseBase

from kamayan.testing import baselines
from parthenon_tools import phdf_diff

sys.dont_write_bytecode = True


@dataclass
class ShockTubeConfig:
    """Configuration for running sedov in different modes."""

    problem: str = "sod"
    ndim: int = 1
    aspect12: int = 2
    aspect13: int = 1

    @property
    def name(self) -> str:
        """Name the test."""
        return f"{self.problem}-NDIM{self.ndim}-a12_{self.aspect12}-a13_{self.aspect13}"

    @property
    def output_vars(self) -> str:
        """Output variables for comparison."""
        vars = ["dens", "pres"]
        if self.problem == "briowu":
            vars += ["magc"]
        return ",".join(vars)


configs = [
    ShockTubeConfig(problem="sod"),
    ShockTubeConfig(problem="briowu"),
    ShockTubeConfig(problem="einfeldt"),
    ShockTubeConfig(problem="briowu", ndim=2),
]


class TestCase(PyKamayanTestCaseBase):
    """Test class for shock tube pyKamayan."""

    def Prepare(self, parameters, step):
        """Configure each run."""
        config = configs[step - 1]
        parameters.driver_cmd_line_args = [
            f"--problem={config.problem}",
            f"--ndim={config.ndim}",
            f"--aspect12={config.aspect12}",
            f"--aspect13={config.aspect13}",
            f"parthenon/job/problem_id={config.name}",
            "parthenon/output0/file_type=hdf5",
            "parthenon/output0/dt=0.05",
            f"parthenon/output0/variables={config.output_vars}",
        ]
        return parameters

    def Analyse(self, parameters) -> bool:
        """Determine success of test cases."""
        baseline_dir = baselines.get_baseline_dir()
        output_dir = Path(parameters.output_path)

        passing = True
        for config in configs:
            name = config.name + ".out0.final.phdf"
            output_file = output_dir / name
            name = config.name + ".out0.final.phdf"
            baseline_file = baseline_dir / name
            delta = phdf_diff.compare(
                [str(output_file), str(baseline_file)],
                check_metadata=False,
                tol=baselines.EPSILON,
            )
            passing = passing and delta == 0

        return passing
