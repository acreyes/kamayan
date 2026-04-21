"""Reconstruction regression test."""

# Modules
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

import sys
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

RES = 64
resolution = RES


_GEO = Literal["cartesian", "cylindrical"]


@dataclass
class ReconstructionConfig:
    """Configuration for running reconstruction in different modes."""

    recon: str
    slope_limiter: str = "minmod"
    max_error: float = 1.0e-12
    geometry: _GEO = "cartesian"
    resolution: int = RES
    mhd: str = "off"

    @property
    def _cyl(self):
        """Check if geometry is cylindrical."""
        return self.geometry == "cylindrical"

    @property
    def x1min(self):
        """Left domain boundary."""
        return -0.0 if self._cyl else -0.5

    @property
    def ix1_bc(self):
        """Left domain boundary condition."""
        return "axisymmetric" if self._cyl else "outflow"

    @property
    def x1max(self):
        """Right domain boundary."""
        return 5.0 if self._cyl else 5.0

    @property
    def ox1_bc(self):
        """Right domain boundary condition."""
        return "outflow" if self._cyl else "periodic"

    @property
    def x2min(self):
        """x2 domain minimum boundary."""
        return -5.0 if self._cyl else -5.0

    @property
    def x2max(self):
        """x2 domain maximum boundary."""
        return 5.0 if self._cyl else 5.0

    @property
    def x2_bc(self):
        """x2 domain boundary condition."""
        return "periodic" if self._cyl else "periodic"

    @property
    def nx1(self):
        """Number of zones in first dimension."""
        return self.resolution

    @property
    def nx2(self):
        """Number of zones in second dimension."""
        return 1 if self._cyl else self.resolution

    @property
    def nx3(self):
        """Number of zones in third dimension."""
        return 1

    @property
    def meshblock_nx3(self):
        """Meshblock size in third dimension."""
        return 1

    @property
    def nghost(self):
        """Number of ghost zones."""
        return 4 if self._cyl else 3

    @property
    def meshblock_nx1(self):
        """Meshblock size in first dimension."""
        return 8 if self._cyl else self.resolution // 2

    @property
    def meshblock_nx2(self):
        """Meshblock size in second dimension."""
        return 1 if self._cyl else self.resolution // 2

    @property
    def name(self) -> str:
        """Problem ID string for the simulation."""
        geo_suffix = f"_{self.geometry}_Mhd-{self.mhd}" if self._cyl else ""
        return f"isentropic_vortex_{self.recon}_{self.slope_limiter}{geo_suffix}"

    @property
    def driver_input(self) -> str:
        """Input file path for the driver."""
        return "isen_1d.in" if self._cyl else "isentropic_vortex.in"

    @property
    def column(self) -> int:
        """Column in the history file to use for error comparison."""
        if self._cyl:
            return 7  # azimuthal velocity
        return 4  # density


configs = [
    ReconstructionConfig("fog", max_error=0.3),
    ReconstructionConfig("plm", slope_limiter="minmod", max_error=0.08),
    ReconstructionConfig("plm", slope_limiter="mc", max_error=0.02),
    ReconstructionConfig("plm", slope_limiter="van_leer", max_error=0.03),
    ReconstructionConfig("ppm", slope_limiter="minmod", max_error=0.02),
    ReconstructionConfig("ppm", slope_limiter="mc", max_error=0.015),
    ReconstructionConfig("wenoz", max_error=0.005),
    ReconstructionConfig("wenoz", geometry="cylindrical", max_error=0.3),
    ReconstructionConfig("wenoz", geometry="cylindrical", mhd="ct", max_error=0.35),
]


class TestCase(utils.test_case.TestCaseAbs):
    """Test class for reconstruction."""

    def Prepare(self, parameters, step):
        """Configure each run with the given configuration.

        Args:
            parameters: The test parameters object.
            step: The test step number (1-indexed).

        Returns:
            The configured parameters object.
        """
        config = configs[step - 1]
        integrator = "rk2"
        args = [
            f"parthenon/job/problem_id={config.name}",
            f"parthenon/mesh/nx1={config.nx1}",
            f"parthenon/mesh/nx2={config.nx2}",
            f"parthenon/mesh/nx3={config.nx3}",
            f"parthenon/meshblock/nx1={config.meshblock_nx1}",
            f"parthenon/meshblock/nx2={config.meshblock_nx2}",
            f"parthenon/meshblock/nx3={config.meshblock_nx3}",
            f"parthenon/mesh/nghost={config.nghost}",
            f"parthenon/time/integrator={integrator}",
            f"hydro/reconstruction={config.recon}",
            f"hydro/slope_limiter={config.slope_limiter}",
            "parthenon/output0/file_type=hst",
            "parthenon/output0/dt=1.0",
            f"physics/MHD={config.mhd}",
        ]
        if config._cyl:
            args.extend(
                [
                    "geometry/geometry=cylindrical",
                    "parthenon/mesh/x1min=-0.0",
                    "parthenon/mesh/x1max=5.0",
                    "parthenon/mesh/ix1_bc=axisymmetric",
                    "parthenon/mesh/ox1_bc=outflow",
                ]
            )
        parameters.driver_cmd_line_args = args
        return parameters

    def Analyse(self, parameters):
        """Determine success of test cases.

        Args:
            parameters: The test parameters object.

        Returns:
            True if all tests pass, False otherwise.
        """
        output_dir = Path(parameters.output_path)
        with open("isentropic_vortex_convergence.out", "w") as fid:
            errors = {}
            for config in configs:
                name = config.name
                history_file = output_dir / f"{name}.out0.hst"
                data = np.loadtxt(history_file, usecols=config.column)
                error = float(data[-1])
                errors[name] = (config, error)
                fid.write(f"{name}: {error} | max error: {config.max_error}\n")

        msg = ""
        test_pass = True
        for name, (config, error) in errors.items():
            if error > config.max_error or np.isnan(error):
                test_pass = False
                msg += f"{name} -- error: {error} | max error: {config.max_error}\n"
        assert test_pass, msg
        return True
