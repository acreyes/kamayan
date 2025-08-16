#!/usr/bin/env python3
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import kamayan.pyKamayan as pyKamayan
import kamayan.pyKamayan.Grid as Grid
from kamayan.pyKamayan.Grid import TopologicalElement as te

import kamayan.kamayan_manager as kman
from kamayan.kamayan_manager import KamayanManager


@dataclass
class SedovData:
    radius: float
    p_ambient: float
    rho_ambient: float
    p_explosion: float

    def dens(self, r: NDArray):
        return self.rho_ambient

    def vel(self, r: NDArray):
        return 0.0

    def pres(self, r: NDArray):
        return (r <= self.radius) * self.p_explosion + (
            r > self.radius
        ) * self.p_ambient


def pgen(mb: Grid.MeshBlock):
    pkg = mb.get_package("sedov")
    data = pkg.GetParam(SedovData, "data")

    pack = mb.pack(["dens", "pres", "velocity"])
    coords = pack.GetCoordinates(0)
    dens = np.array(pack.GetParArray3D(0, "dens", te.CC).view(), copy=False)
    pres = np.array(pack.GetParArray3D(0, "pres", te.CC).view(), copy=False)
    vel1 = np.array(pack.GetParArray3D(0, "velocity", te.CC).view(), copy=False)
    vel2 = np.array(pack.GetParArray3D(0, "velocity", te.CC, 1).view(), copy=False)
    vel3 = np.array(pack.GetParArray3D(0, "velocity", te.CC, 2).view(), copy=False)

    indices = np.indices(dens.shape)
    xx = np.array(coords.Xc1(indices[2, :, :, :]))
    yy = np.array(coords.Xc2(indices[1, :, :, :]))
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

    nlevels = pmesh.Get(int, "numlevel")
    nx = pmesh.Get(int, "nx1")
    xmin = pmesh.Get(float, "x1min")
    xmax = pmesh.Get(float, "x1max")
    dx = (xmax - xmin) / (2 ** (nlevels - 1) * nx)
    nu = 2.0

    sedov = udc.Data("sedov")
    radius = 3.5 * dx
    dens = sedov.Get(float, "density")
    p = sedov.Get(float, "pressure")
    E = sedov.Get(float, "energy")

    eos = udc.Data("eos/gamma")
    gamma = eos.Get(float, "gamma")
    pres = 3.0 * (gamma - 1.0) * E / ((nu + 1) * np.pi * radius**nu)

    data = SedovData(rho_ambient=dens, p_ambient=p, p_explosion=pres, radius=radius)
    pkg.AddParam("data", data)


def set_parameters(params) -> None:
    """Set input parameters by block."""

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

    params["parthenon/mesh"] = (
        mesh(1, 128, (-0.5, 0.5), ("outflow", "outflow"))
        | mesh(2, 128, (-0.5, 0.5), ("outflow", "outflow"))
        | mesh(3, 1, (-0.5, 0.5), ("outflow", "outflow"))
        | {"nghost": 4, "refinement": "adaptive", "numlevel": 3}
    )

    params["parthenon/meshblock"] = {"nx1": 32, "nx2": 32, "nx3": 1}
    params["kamayan/refinement0"] = {"field": "pres"}
    params["parthenon/time"] = {
        "nlim": 10000,
        "tlim": 0.05,
        "integrator": "rk2",
        "ncycle_out_mesh": -10000,
    }

    params["parthenon/output0"] = {"file_type": "rst", "dt": 0.01, "dn": -1}
    params["eos"] = {"mode_init": "dens_pres"}
    params["hydro"] = {
        "cfl": 0.8,
        "reconstruction": "wenoz",
        "riemann": "hllc",
        "slope_limiter": "minmod",
    }
    params["sedov"] = {"density": 1.0, "pressure": 1.0e-5, "energy": 1.0}


def main():
    units = kman.process_units(
        "sedov", setup_params=setup, initialize=initialize, pgen=pgen
    )
    km = KamayanManager("sedov", units)
    set_parameters(km.params)

    km.execute()


if __name__ == "__main__":
    main()
