"""Shock tube THINC pyKamayan regression test.

Runs Sod problem twice: once with MC limiter (baseline), once with THINC.
Validates that THINC produces a sharper contact discontinuity.
"""

from pathlib import Path
import glob
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".." / "regression"))
from pykamayan_test_case import PyKamayanTestCaseBase

sys.dont_write_bytecode = True


class TestCase(PyKamayanTestCaseBase):
    """Test that THINC sharpens the Sod contact discontinuity vs MC baseline."""

    def Prepare(self, parameters, step):
        """Configure each run: step 1 = MC baseline, step 2 = THINC."""
        base_args = [
            f"parthenon/job/problem_id=sod_{'thinc' if step == 2 else 'mc'}",
            "--problem=sod",
            "--ndim=1",
            "parthenon/output0/file_type=hdf5",
            "parthenon/output0/dt=0.05",
            "parthenon/output0/variables=dens",
            "hydro/reconstruction=plm",
            "hydro/riemann=hllc",
        ]
        if step == 2:
            base_args += [
                "hydro/slope_limiter=thinc",
                "hydro/thinc_fallback=mc",
                "hydro/beta_thinc=1.6",
            ]
        else:
            base_args += ["hydro/slope_limiter=mc"]
        parameters.driver_cmd_line_args = base_args
        return parameters

    def Analyse(self, parameters) -> bool:
        """Compare contact width between MC and THINC runs."""
        try:
            from parthenon_tools.phdf import phdf
        except ImportError:
            print("ERROR: parthenon_tools not available")
            return False

        output_dir = Path(parameters.output_path)

        # Find final output files
        mc_files = sorted(glob.glob(str(output_dir / "sod_mc.out0.*.phdf")))
        thinc_files = sorted(glob.glob(str(output_dir / "sod_thinc.out0.*.phdf")))

        if not mc_files or not thinc_files:
            print(f"ERROR: Missing output files in {output_dir}")
            print(f"  MC files: {mc_files}")
            print(f"  THINC files: {thinc_files}")
            return False

        mc_data = phdf(mc_files[-1])
        thinc_data = phdf(thinc_files[-1])

        dens_mc = mc_data.Get("dens", flatten=True)
        dens_thinc = thinc_data.Get("dens", flatten=True)

        if dens_mc is None or dens_thinc is None:
            print("ERROR: Could not read density data")
            return False

        passing = True

        # Check physical validity: 0 < dens <= 1.0
        for label, dens in [("MC", dens_mc), ("THINC", dens_thinc)]:
            if dens.min() <= 0:
                print(
                    f"FAIL: {label} density has non-positive values: min={dens.min()}"
                )
                passing = False
            if dens.max() > 1.0 + 1e-10:
                print(f"FAIL: {label} density exceeds initial state: max={dens.max()}")
                passing = False

        # Measure contact width: count cells in transition region
        lo, hi = 0.2, 0.9
        width_mc = ((dens_mc > lo) & (dens_mc < hi)).sum()
        width_thinc = ((dens_thinc > lo) & (dens_thinc < hi)).sum()

        print(f"Contact width (cells in [{lo}, {hi}]):")
        print(f"  MC:    {width_mc}")
        print(f"  THINC: {width_thinc}")

        if width_thinc >= width_mc:
            print(
                f"FAIL: THINC contact ({width_thinc} cells) not sharper "
                f"than MC ({width_mc} cells)"
            )
            passing = False
        else:
            print(
                f"PASS: THINC contact ({width_thinc} cells) sharper "
                f"than MC ({width_mc} cells)"
            )

        return passing
