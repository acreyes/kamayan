# Modules
import math
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

base_resolution = 32


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        mx = base_resolution * 2 ** step
        recon = "wenoz"
        integrator = "rk3"
        parameters.driver_cmd_line_args = [
                f"parthenon/mesh/nx1={mx}",
                f"parthenon/mesh/nx2={mx}",
                f"parthenon/meshblock/nx1={base_resolution}",
                f"parthenon/meshblock/nx2={base_resolution}",
                f"parthenon/mesh/nghost=3",
                f"parthenon/time/integrator={integrator}",
                f"hydro/reconstruction={recon}",
        ]
        return parameters

    def Analyse(self, parameters):


        return True
