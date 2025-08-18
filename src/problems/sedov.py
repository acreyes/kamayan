#!/usr/bin/env python3
"""Setup for the Sedov blast wave."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import kamayan.pyKamayan as pyKamayan
import kamayan.pyKamayan.Grid as Grid
from kamayan.pyKamayan.Grid import TopologicalElement as te

import kamayan.kamayan_manager as kman
from kamayan.kamayan_manager import KamayanManager, KamayanParams

from kamayan.code_units import Grid as gr, eos as eos, driver
from kamayan.code_units.Grid import AdaptiveGrid
from kamayan.code_units.Hydro import Hydro


@dataclass
class SedovData:
    """Light data class for sedov IC."""

    radius: float
    p_ambient: float
    rho_ambient: float
    p_explosion: float

    def dens(self, r: NDArray):
        """Rho(r)."""
        return self.rho_ambient

    def vel(self, r: NDArray):
        """Velocity(r)."""
        return 0.0

    def pres(self, r: NDArray):
        """Initial delta function energy deposition."""
        return (r <= self.radius) * self.p_explosion + (
            r > self.radius
        ) * self.p_ambient


def pgen(mb: Grid.MeshBlock):
    """Single mesh block initial conditions."""
    # --8<-- [start:py_get_param]
    pkg = mb.get_package("sedov")
    # Any python object can get pulled out of Params and validated for type
    data = pkg.GetParam(SedovData, "data")
    # --8<-- [end:py_get_param]

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
    """Setup runtime parameters for sedov."""
    # --8<-- [start:py_add_parms]
    # add a new input parameter block
    sedov = udc.AddData("sedov")
    # add Real runtime parameters
    sedov.AddReal("density", 1.0, "ambient density")
    sedov.AddReal("pressure", 1.0e-5, "ambient pressure")
    sedov.AddReal("energy", 1.0, "explosion energy")
    # --8<-- [end:py_add_parms]


def initialize(udc: pyKamayan.UnitDataCollection):
    """Initialize sedov package data/params."""
    pkg = udc.Package()
    pmesh = udc.Data("parthenon/mesh")

    nlevels = pmesh.Get(int, "numlevel")
    nx = pmesh.Get(int, "nx1")
    xmin = pmesh.Get(float, "x1min")
    xmax = pmesh.Get(float, "x1max")
    dx = (xmax - xmin) / (2 ** (nlevels - 1) * nx)
    nu = 2.0

    # --8<-- [start:py_get_parms]
    # fetch out the parameters during initialize
    sedov = udc.Data("sedov")
    radius = 3.5 * dx
    # provide types to validate the expected types and provide static type checking
    dens = sedov.Get(float, "density")
    p = sedov.Get(float, "pressure")
    E = sedov.Get(float, "energy")
    # --8<-- [end:py_get_parms]

    eos = udc.Data("eos/gamma")
    gamma = eos.Get(float, "gamma")
    pres = 3.0 * (gamma - 1.0) * E / ((nu + 1) * np.pi * radius**nu)

    # --8<-- [start:py_set_param]
    # arbitrary python types can be added to our package Params
    data = SedovData(rho_ambient=dens, p_ambient=p, p_explosion=pres, radius=radius)
    pkg.AddParam("data", data)
    # --8<-- [end:py_set_param]


def set_parameters(params: KamayanParams) -> None:
    """Set input parameters by block."""
    params["parthenon/output0"] = {"file_type": "rst", "dt": 0.01, "dn": -1}


def make_kman() -> KamayanManager:
    """Build the KamayanManager for Sedov."""
    units = kman.process_units(
        "sedov", setup_params=setup, initialize=initialize, pgen=pgen
    )
    km = KamayanManager("sedov", units)

    nxb = 32  # zones per block
    nblocks = int(128 / 32)  # number of blocks to get 128 zones at coarsest resolution
    km.grid = AdaptiveGrid(
        xbnd1=(-0.5, 0.5),  # xmin/max
        xbnd2=(-0.5, 0.5),  # ymin/max
        nxb1=nxb,  # zones per block along x
        nxb2=nxb,
        num_levels=3,  # 3 levels of refinement
        nblocks1=nblocks,  # number of root blocks in each direction
        nblocks2=nblocks,
    )
    km.grid.refinement_fields.add("pres")
    km.grid.boundary_conditions = gr.outflow_box()

    km.driver = driver.Driver(integrator="rk2", tlim=0.05)
    km.physics.hydro = Hydro(reconstruction="wenoz", riemann="hllc")
    km.physics.eos = eos.GammaEos(gamma=5.0 / 3.0, mode_init="dens_pres")

    km.params["sedov"] = {"density": 1.0, "pressure": 1.0e-5, "energy": 1.0}
    return km


def main():
    """Construct and run sedov."""
    km = make_kman()
    set_parameters(km.params)

    km.execute()


if __name__ == "__main__":
    main()
