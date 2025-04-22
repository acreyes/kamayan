# Modules
import math
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

base_resolution = 16
resolutions = []


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        mx = base_resolution * 2 ** step
        resolutions.append(mx)
        recon = "ppm"
        integrator = "rk2"
        parameters.driver_cmd_line_args = [
                f"parthenon/job/problem_id=isentropic_vortex_{mx}",
                f"parthenon/mesh/nx1={mx}",
                f"parthenon/mesh/nx2={mx}",
                f"parthenon/meshblock/nx1={mx / 2}",
                f"parthenon/meshblock/nx2={mx / 2}",
                f"parthenon/mesh/nghost=3",
                f"parthenon/time/integrator={integrator}",
                f"hydro/reconstruction={recon}",
                f"parthenon/output0/file_type=hst",
                f"parthenon/output0/dt=1.0",
        ]
        return parameters

    def Analyse(self, parameters):
        output_dir = Path(parameters.output_path)
        errors = []
        for mx in resolutions:
            history_file = output_dir / f"isentropic_vortex_{mx}.out0.hst"
            data = np.loadtxt(history_file, usecols=4)
            errors.append(data[-1])

        with open("isentropic_vortex_convergence.out", "w") as fid:
            min_slope = 100.
            for i in range(1, len(errors)):
                slope = - np.log(errors[i]/errors[i-1]) / np.log(resolutions[i]/resolutions[i-1])
                min_slope = min(slope, min_slope)
                fid.write(f"{resolutions[i-1]} {errors[i-1]} {resolutions[i]} {errors[i]} {slope}\n")

        return min_slope > 1.5
