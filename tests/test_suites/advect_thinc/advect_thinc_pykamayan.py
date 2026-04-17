"""Advecting disc THINC regression test.

Advects a circular density disc across a 2D periodic domain and compares
interface sharpness between MC limiter (baseline) and THINC reconstruction.
After one full crossing (t=1.0), THINC should preserve a sharper interface.
"""

from pathlib import Path
import glob
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".." / "regression"))
from pykamayan_test_case import PyKamayanTestCaseBase

sys.dont_write_bytecode = True


class TestCase(PyKamayanTestCaseBase):
    """Test that THINC keeps the advected disc interface sharper than MC."""

    def Prepare(self, parameters, step):
        """Configure each run: step 1 = MC baseline, step 2 = THINC."""
        label = "thinc" if step == 2 else "mc"
        base_args = [
            f"parthenon/job/problem_id=advect_{label}",
            "--problem=advect_disc",
            "parthenon/output0/file_type=hdf5",
            "parthenon/output0/dt=0.5",
            "parthenon/output0/variables=dens",
            "hydro/reconstruction=plm",
            "hydro/riemann=hllc",
            "hydro/cfl=0.4",
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
        """Compare interface width between MC and THINC runs."""
        try:
            from parthenon_tools.phdf import phdf
        except ImportError:
            print("ERROR: parthenon_tools not available")
            return False

        output_dir = Path(parameters.output_path)

        mc_files = sorted(glob.glob(str(output_dir / "advect_mc.out0.*.phdf")))
        thinc_files = sorted(glob.glob(str(output_dir / "advect_thinc.out0.*.phdf")))

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

        # Physical validity: density should remain positive and bounded
        dens_in, dens_out = 10.0, 1.0
        for label, dens in [("MC", dens_mc), ("THINC", dens_thinc)]:
            if dens.min() <= 0:
                print(
                    f"FAIL: {label} density has non-positive values: min={dens.min()}"
                )
                passing = False
            if dens.max() > dens_in * 1.01:
                print(f"FAIL: {label} density exceeds initial: max={dens.max()}")
                passing = False

        # Interface sharpness metric: measure the "width" of the transition
        # region as the number of cells in the intermediate density range.
        # A sharper interface means fewer cells in the transition.
        lo = dens_out + 0.1 * (dens_in - dens_out)  # 1.9
        hi = dens_in - 0.1 * (dens_in - dens_out)  # 9.1
        width_mc = ((dens_mc > lo) & (dens_mc < hi)).sum()
        width_thinc = ((dens_thinc > lo) & (dens_thinc < hi)).sum()

        print(f"Interface width (transition cells in [{lo:.1f}, {hi:.1f}]):")
        print(f"  MC:    {width_mc}")
        print(f"  THINC: {width_thinc}")

        # Additionally measure L1 error vs initial condition.
        # After one full period the disc should return to its starting position.
        # Compute the fraction of density that has been diffused away from
        # the sharp profile: sum of |dens - dens_initial| / sum(dens_initial).
        # Since we don't have the initial profile in the output, use the
        # transition cell count as the primary metric.

        if width_thinc >= width_mc:
            print(
                f"FAIL: THINC interface ({width_thinc} cells) not sharper "
                f"than MC ({width_mc} cells)"
            )
            passing = False
        else:
            reduction = (1.0 - width_thinc / width_mc) * 100
            print(
                f"PASS: THINC interface ({width_thinc} cells) sharper "
                f"than MC ({width_mc} cells) — {reduction:.0f}% reduction"
            )

        return passing
