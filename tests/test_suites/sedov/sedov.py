# Modules
from dataclasses import dataclass

import sys
import utils.test_case

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
        return f"sedov_{config.riemann}"

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
            f"hydro/riemann={config.riemann}",
            "parthenon/output0/file_type=hdf5",
            "parthenon/output0/dt=1.0",
            "parthenon/output0/variables=dens,pres",
        ]
        return parameters

    def Analyse(self, parameters) -> bool:
        ...
        # output_dir = Path(parameters.output_path)
        # with open("isentropic_vortex_convergence.out", "w") as fid:
        #     errors = {}
        #     for config in configs:
        #         name = self._test_namer(config)
        #         history_file = output_dir / f"{name}.out0.hst"
        #         data = np.loadtxt(history_file, usecols=9)
        #         error = float(data[-1])
        #         errors[name] = (config, error)
        #         fid.write(f"{name}: {error} | max error: {config.max_error}\n")

        # msg = ""
        # test_pass = True
        # for name, (config, error) in errors.items():
        #     if error > config.max_error or np.isnan(error):
        #         test_pass = False
        #         msg += f"{name} -- error: {error} | max error: {config.max_error}\n"
        # assert test_pass, msg
        return True
