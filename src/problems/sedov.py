#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
import sys

import mpi4py
import numpy as np
from numpy.typing import ArrayLike

from kamayan import pyKamayan, RuntimeParameters
from kamayan.pyKamayan import Grid

te = Grid.TopologicalElement
mpi4py.rc.initialize = False  # Disable automatic MPI initialization


@dataclass
class SedovData:
    radius: float
    p_ambient: float
    rho_ambient: float
    p_explosion: float

    def dens(self, r: ArrayLike):
        return self.rho_ambient

    def vel(self, r: ArrayLike):
        return 0.0

    def pres(self, r: ArrayLike):
        return (r <= self.radius) * self.p_explosion + (
            r > self.radius
        ) * self.p_ambient


def pgen(mb: Grid.MeshBlock):
    pkg = mb.get_package("sedov")
    data = pkg.GetParam("data")

    pack = mb.pack(["dens", "pres", "velocity"])
    coords = pack.GetCoordinates(0)
    dens = np.array(pack.GetParArray3D(0, "dens", te.CC).view(), copy=False)
    pres = np.array(pack.GetParArray3D(0, "pres", te.CC).view(), copy=False)
    vel1 = np.array(pack.GetParArray3D(0, "velocity", te.CC).view(), copy=False)
    vel2 = np.array(pack.GetParArray3D(0, "velocity", te.CC, 1).view(), copy=False)
    vel3 = np.array(pack.GetParArray3D(0, "velocity", te.CC, 2).view(), copy=False)

    indices = np.indices(dens.shape)
    xx = coords.Xc1(indices[2, :, :, :])
    yy = coords.Xc2(indices[1, :, :, :])
    rr = np.sqrt(xx**2 + yy**2)

    dens[:, :, :] = data.dens(rr)
    pres[:, :, :] = data.pres(rr)
    vel1[:, :, :] = 0.0
    vel2[:, :, :] = 0.0
    vel3[:, :, :] = 0.0


def setup(udc: pyKamayan.UnitDataCollection):
    sedov = udc.AddData("sedov")
    sedov.AddReal("density", 1.0, "ambient density")
    sedov.AddReal("pressure", 1.0e-5, "ambient pressure")
    sedov.AddReal("energy", 1.0, "explosion energy")


def initialize(udc: pyKamayan.UnitDataCollection):
    pkg = udc.Package()
    pmesh = udc.Data("parthenon/mesh")

    nlevels = pmesh.Get("numlevel")
    nx = pmesh.Get("nx1")
    xmin = pmesh.Get("x1min")
    xmax = pmesh.Get("x1max")
    dx = (xmax - xmin) / (2 ** (nlevels - 1) * nx)
    nu = 2.0

    sedov = udc.Data("sedov")
    radius = 3.5 * dx
    dens = sedov.Get("density")
    p = sedov.Get("pressure")
    E = sedov.Get("energy")

    eos = udc.Data("eos/gamma")
    gamma = eos.Get("gamma")
    pres = 3.0 * (gamma - 1.0) * E / ((nu + 1) * np.pi * radius**nu)

    data = SedovData(rho_ambient=dens, p_ambient=p, p_explosion=pres, radius=radius)
    pkg.AddParam("data", data)


def input_parameters(input_file: Path) -> RuntimeParameters.InputParameters:
    rpb = RuntimeParameters.RuntimeParametersBlock
    pin = RuntimeParameters.InputParameters(input_file)
    pin.add(rpb("parthenon/job", {"problem_id": "sedov"}))

    def mesh(
        dir: int, nx: int, bnd: tuple[float, float], bc: tuple[str, str]
    ) -> dict[str, int | float | str]:
        return {
            f"nx{dir}": nx,
            f"x{dir}min": bnd[0],
            f"x{dir}max": bnd[1],
            f"ix{dir}_bc": bc[0],
            f"ox{dir}_bc": bc[1],
        }

    pin.add(
        rpb(
            "parthenon/mesh",
            mesh(1, 128, (-0.5, 0.5), ("outflow", "outflow"))
            | mesh(2, 128, (-0.5, 0.5), ("outflow", "outflow"))
            | mesh(3, 1, (-0.5, 0.5), ("outflow", "outflow"))
            | {"nghost": 4, "refinement": "adaptive", "numlevel": 3},
        )
    )

    pin.add(rpb("parthenon/meshblock", {"nx1": 32, "nx2": 32, "nx3": 1}))
    pin.add(rpb("kamayan/refinement0", {"field": "pres"}))
    pin.add(
        rpb(
            "parthenon/time",
            {
                "nlim": 10000,
                "tlim": 0.05,
                "integrator": "rk2",
                "ncycle_out_mesh": -10000,
            },
        )
    )
    pin.add(rpb("parthenon/output0", {"file_type": "rst", "dt": 0.01, "dn": -1}))
    pin.add(rpb("eos", {"mode_init": "dens_pres"}))
    pin.add(
        rpb(
            "hydro",
            {
                "cfl": 0.8,
                "reconstruction": "wenoz",
                "riemann": "hllc",
                "slope_limiter": "minmod",
            },
        )
    )
    pin.add(rpb("sedov", {"density": 1.0, "pressure": 1.0e-5, "energy": 1.0}))

    return pin


def main():
    # generate the input file we will use
    input_file = Path(".sedov.in")
    pin = input_parameters(input_file)
    pin.write()

    # initialize the environment from the previously generated input file
    pman = pyKamayan.InitEnv([sys.argv[0], "-i", str(pin.input_file)] + sys.argv[1:])

    # make the simulation unit, handles registering runtime parameters, caching
    # simulation data as a Param, and initial conditions
    simulation = pyKamayan.KamayanUnit("sedov")
    # register callbacks
    simulation.set_SetupParams(setup)
    simulation.set_InitializeData(initialize)
    simulation.set_ProblemGeneratorMeshBlock(pgen)

    # get the default units and register our simulation unit
    units = pyKamayan.ProcessUnits()
    units.Add(simulation)

    # get a driver and execute the code
    driver = pyKamayan.InitPackages(pman, units)
    driver_status = driver.Execute()
    if driver_status != pyKamayan.DriverStatus.complete:
        raise RuntimeError("Simulation has not succesfully completed.")

    pman.ParthenonFinalize()


if __name__ == "__main__":
    main()
