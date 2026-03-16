#!/usr/bin/env python3
"""PyKamayan regression test runner.

Runs pyKamayan problem scripts via mpirun and compares outputs against baselines.

Usage:
    python run_test.py --script src/problems/sedov.py --num_steps 4 --output_dir ./output
    mpirun -np 4 python run_test.py --script src/problems/sedov.py --num_steps 4
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from shutil import rmtree
from subprocess import PIPE, STDOUT
import subprocess

try:
    import mpi4py

    mpi4py.rc(initialize=False)
except ImportError:
    ()

sys.dont_write_bytecode = True

import utils.test_case

TestCaseAbs = utils.test_case.TestCaseAbs


class Parameters:
    script_path = ""
    test_path = ""
    output_path = ""
    mpi_cmd = ""
    num_ranks = 1
    mpi_ranks_flag = None
    mpi_opts = ""
    driver_cmd_line_args = []
    stdouts = []
    kokkos_args = []
    coverage_status = "only-regression"
    sparse_disabled = False


class PyKamayanTestManager:
    def __init__(self, run_test_path, **kwargs):
        self._run_coverage = kwargs.pop("coverage")
        self.parameters = Parameters()
        self._run_test_py_path = run_test_path
        self._regression_test_suite_path = os.path.join(
            self._run_test_py_path, "..", "test_suites"
        )

        kamayan_path = os.path.join(self._run_test_py_path, "..", "..", "python")
        if os.path.isdir(kamayan_path) and kamayan_path not in sys.path:
            sys.path.insert(0, kamayan_path)

        test_dir = kwargs.pop("test_dir")
        script_path = kwargs.pop("script")
        self.parameters.kokkos_args = " ".join(kwargs.pop("kokkos_args")).split()
        mpi_executable = kwargs.pop("mpirun")

        test_path = self._checkAndGetRegressionTestFolder(test_dir[0])

        self.test = os.path.basename(os.path.normpath(test_path))

        self._checkRegressionTestScript(test_path, self.test)

        script_path_input = script_path[0]
        if os.path.isabs(script_path_input):
            script_path = script_path_input
        else:
            script_path_input = os.path.join(
                self._run_test_py_path, "..", "..", "..", script_path_input
            )
            script_path = os.path.abspath(script_path_input)

        output_dir = kwargs.pop("output_dir")
        if output_dir == "":
            output_dir = os.path.abspath(test_path + "/output")
        else:
            output_dir = os.path.abspath(output_dir)

        self.parameters.script_path = script_path
        self.parameters.output_path = output_dir
        self.parameters.test_path = test_path
        self.parameters.mpi_cmd = mpi_executable
        self.parameters.mpi_ranks_flag = kwargs.pop("mpirun_ranks_flag")
        self.parameters.num_ranks = int(kwargs.pop("mpirun_ranks_num"))
        self.parameters.mpi_opts = kwargs.pop("mpirun_opts")
        self.parameters.sparse_disabled = kwargs.pop("sparse_disabled")

        output_msg = "Using:\n"
        output_msg += "script at:       " + script_path + "\n"
        output_msg += "test folder:     " + test_path + "\n"
        output_msg += "output sent to:  " + output_dir + "\n"
        print(output_msg)
        sys.stdout.flush()

        module_root_path = os.path.join(test_path, "..", "..")
        if module_root_path not in sys.path:
            sys.path.insert(0, module_root_path)

        utils_path = os.path.join(self._run_test_py_path, "utils")
        if utils_path not in sys.path:
            sys.path.insert(0, utils_path)

        test_base_name = os.path.split(test_path)[1]
        self._test_module = (
            "test_suites." + test_base_name + "." + test_base_name + "_pykamayan"
        )

        try:
            module = __import__(
                self._test_module,
                globals(),
                locals(),
                fromlist=["TestCase"],
            )
            my_TestCase = getattr(module, "TestCase")
        except (ImportError, AttributeError):
            fallback_module = "test_suites." + test_base_name + "." + test_base_name
            module = __import__(
                fallback_module,
                globals(),
                locals(),
                fromlist=["TestCase"],
            )
            my_TestCase = getattr(module, "TestCase")

        self.test_case = my_TestCase()

        if not issubclass(my_TestCase, TestCaseAbs):
            raise TestManagerError("TestCase is not a child of TestCaseAbs")

    def _checkAndGetRegressionTestFolder(self, test_dir):
        if not os.path.isdir(test_dir):
            test_suites_path = os.path.join(
                self._run_test_py_path, "..", "test_suites", test_dir
            )
            if not os.path.isdir(test_suites_path):
                error_msg = "Regression test folder is unknown: " + test_dir + "\n"
                error_msg += "looked in:\n"
                error_msg += "  tests/test_suites/" + test_dir + "\n"
                error_msg += "  " + test_dir + "\n"
                error_msg += "Each regression test must have a folder in "
                error_msg += "tests/test_suites.\n"
                error_msg += "Known tests folders are:"
                known_test_folders = os.listdir(self._regression_test_suite_path)
                for folder in known_test_folders:
                    if folder != "__init__.py":
                        error_msg += "\n  " + folder
                raise TestManagerError(error_msg)
            else:
                return os.path.abspath(test_suites_path)
        else:
            return os.path.abspath(test_dir)

    def _checkRegressionTestScript(self, test_dir, test_base_name):
        python_test_script = os.path.join(test_dir, test_base_name + "_pykamayan.py")
        if not os.path.isfile(python_test_script):
            fallback_script = os.path.join(test_dir, test_base_name + ".py")
            if not os.path.isfile(fallback_script):
                error_msg = "Missing regression test file "
                error_msg += python_test_script + " or " + fallback_script
                error_msg += "\nEach test folder must have a python script with the "
                error_msg += "same name as the regression test folder."
                raise TestManagerError(error_msg)

    def _checkScriptPath(self, script_path):
        if not os.path.isfile(script_path):
            raise TestManagerError("Unable to locate script " + script_path)

    def MakeOutputFolder(self):
        if os.path.isdir(self.parameters.output_path):
            try:
                rmtree(self.parameters.output_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        if not os.path.isdir(self.parameters.output_path):
            os.makedirs(self.parameters.output_path)

        os.chdir(self.parameters.output_path)

    def Prepare(self, step):
        print("*****************************************************************")
        print("Preparing Test Case Step %d" % step)
        print("*****************************************************************\n")
        sys.stdout.flush()
        self.parameters = self.test_case.Prepare(self.parameters, step)

    def Run(self):
        run_command = []

        if self.parameters.mpi_cmd:
            run_command.extend(self.parameters.mpi_cmd)

        if self.parameters.mpi_ranks_flag is not None:
            run_command.append(self.parameters.mpi_ranks_flag)
            run_command.append(str(self.parameters.num_ranks))

        for opt in self.parameters.mpi_opts:
            run_command.extend(opt.split())

        uv_path = shutil.which("uv") or "uv"
        run_command.extend([uv_path, "run", "kamayan", self.parameters.script_path])

        for arg in self.parameters.driver_cmd_line_args:
            run_command.append(arg)

        for arg in self.parameters.kokkos_args:
            run_command.append(arg)

        if self._run_coverage and self.parameters.coverage_status != "only-regression":
            print("*****************************************************************")
            print("Running PyKamayan with Coverage")
            print("*****************************************************************\n")
        elif (
            not self._run_coverage
            and self.parameters.coverage_status != "only-coverage"
        ):
            print("*****************************************************************")
            print("Running PyKamayan Simulation")
            print("*****************************************************************\n")
        elif (
            self._run_coverage and self.parameters.coverage_status == "only-regression"
        ):
            print("*****************************************************************")
            print("Test Case Ignored for Calculating Coverage")
            print("*****************************************************************\n")
            return
        else:
            return

        print("Command to execute:")
        print(" ".join(run_command))
        sys.stdout.flush()

        try:
            result = subprocess.run(
                run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                print("STDOUT:", result.stdout.decode()[:1000] if result.stdout else "")
                print("STDERR:", result.stderr.decode()[:1000] if result.stderr else "")
                raise subprocess.CalledProcessError(
                    result.returncode, run_command, result.stdout, result.stderr
                )
            else:
                print(result.stdout.decode()[:500] if result.stdout else "")
                self.parameters.stdouts.append(result.stdout or b"")
        except subprocess.CalledProcessError as err:
            if self.test_case.ErrorOnNonZeroReturnCode(self.parameters, err.returncode):
                print(
                    "\n*****************************************************************"
                )
                print("Subprocess error message")
                print(
                    "*****************************************************************\n"
                )
                print(str(repr(err.output)).replace("\\n", os.linesep))
                print(
                    "\n*****************************************************************"
                )
                print("Error detected while running subprocess command")
                print(
                    "*****************************************************************\n"
                )
                raise TestManagerError(
                    "\nReturn code {0} from command '{1}'".format(
                        err.returncode, " ".join(run_command)
                    )
                )
            else:
                self.parameters.stdouts.append(err.stdout)

        self.parameters.coverage_status = "only-regression"

    def Analyse(self):
        test_pass = False
        if self._run_coverage:
            print("*****************************************************************")
            print("Running with Coverage, Analysis Section Ignored")
            print("*****************************************************************\n")
            return True

        print("Running with coverage")
        print(self._run_coverage)
        print("*****************************************************************")
        print("Analysing PyKamayan Output")
        print("*****************************************************************\n")
        sys.stdout.flush()
        test_pass = self.test_case.Analyse(self.parameters)

        return test_pass


def checkRunScriptLocation(run_test_py_path):
    """Check that run_test is in the correct folder."""
    if not os.path.normpath(run_test_py_path).endswith(
        os.path.normpath("tests/regression")
    ):
        error_msg = (
            "Cannot run run_test.py, it is not in the correct directory, must be "
        )
        error_msg += "kept in tests/regression"
        raise TestError(error_msg)

    test_suites_path = os.path.join(run_test_py_path, "..", "test_suites")
    if not os.path.isdir(test_suites_path):
        raise TestError("Cannot run run_test.py, the test_suites folder is missing.")


def main(**kwargs):
    print("\n")
    print("\n".join(["{}={!r}".format(k, v) for k, v in kwargs.items()]))

    mpirun_opts = kwargs.get("mpirun_opts", [])
    mpirun = kwargs.get("mpirun")
    if mpirun_opts and (mpirun is None or mpirun == ""):
        raise TestError("Cannot provide --mpirun_opts without specifying --mpirun")

    print("*****************************************************************")
    print("Beginning PyKamayan Regression Testing Script")
    print("*****************************************************************\n")

    run_test_py_path = os.path.dirname(os.path.realpath(__file__))
    checkRunScriptLocation(run_test_py_path)

    print("Initializing Test Case")

    test_manager = PyKamayanTestManager(run_test_py_path, **kwargs)

    if "analyze" not in kwargs or kwargs["analyze"] == False:
        print("Make output folder in test if does not exist")
        test_manager.MakeOutputFolder()

        for step in range(1, kwargs["num_steps"] + 1):
            test_manager.Prepare(step)
            test_manager.Run()

    test_result = test_manager.Analyse()

    if test_result == True:
        return 0
    else:
        raise TestError("Test " + test_manager.test + " failed")


class TestError(RuntimeError):
    pass


class TestManagerError(RuntimeError):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "run_test.py - a pyKamayan regression testing script"
    )

    parser.add_argument(
        "--test_dir",
        "-t",
        type=str,
        nargs=1,
        required=True,
        help="Name of the test directory, relative to test_suites/, excluding .py",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="",
        help="Path to simulation outputs. Defaults to individual 'output' folders in test_suites.",
    )

    parser.add_argument(
        "--script",
        "-s",
        type=str,
        nargs=1,
        required=True,
        help="Path to the pyKamayan problem script (e.g., src/problems/sedov.py)",
    )

    parser.add_argument(
        "--kokkos_args",
        "-k_a",
        default=[],
        action="append",
        help="Kokkos arguments to pass to driver",
    )

    parser.add_argument(
        "--num_steps",
        "-n",
        type=int,
        default=1,
        required=False,
        help="Number of steps in test. Default: 1",
    )

    parser.add_argument(
        "--mpirun",
        default="",
        nargs=1,
        help="MPI run wrapper command (e.g., mpirun, srun)",
    )

    parser.add_argument(
        "--mpirun_ranks_flag",
        default=None,
        type=str,
        help="Flag for the number of ranks",
    )

    parser.add_argument(
        "--mpirun_ranks_num", default=1, type=int, help="Number of ranks"
    )

    parser.add_argument(
        "--mpirun_opts",
        default=[],
        action="append",
        help="Add options to mpirun command",
    )

    parser.add_argument(
        "--coverage",
        "-c",
        action="store_true",
        help="Run test cases where coverage has been enabled",
    )

    parser.add_argument(
        "--analyze",
        "-a",
        action="store_true",
        help="Skip to analysis, assumes test data already exists",
    )

    parser.add_argument(
        "--sparse_disabled",
        action="store_true",
        help="Signal that sparse is compile-time disabled",
    )

    args = parser.parse_args()

    try:
        sys.exit(main(**vars(args)))
    except TestError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception:
        raise
