#!/usr/bin/env python3
from dataclasses import dataclass

import mpi4py
import numpy as np

from kamayan import pyKamayan
from kamayan.pyKamayan import Grid

te = Grid.TopologicalElement
mpi4py.rc.initialize = False  # Disable automatic MPI initialization
from mpi4py import MPI


@dataclass
class SedovData:
    radius: float
    p_ambient: float
    rho_ambient: float
    p_explosion: float

    def dens(self, r: float):
        return self.rho_ambient

    def vel(self, r: float):
        return 0.0

    def pres(self, r: float):
        return (r <= self.radius) * self.p_explosion + (
            r > self.radius
        ) * self.p_ambient


def pgen(mb: Grid.MeshBlock):
    pkg = mb.get_package("sedov")
    data = pkg.GetParam("data")

    pack = mb.pack(["dens", "pres", "velocity"])
    coords = pack.GetCoordinates(0)
    dens = np.array(pack.GetParArray3D(0, "dens", te.CC), copy=False)
    pres = np.array(pack.GetParArray3D(0, "pres", te.CC), copy=False)
    vel1 = np.array(pack.GetParArray3D(0, "velocity", te.CC), copy=False)
    vel2 = np.array(pack.GetParArray3D(0, "velocity", te.CC, 1), copy=False)
    vel3 = np.array(pack.GetParArray3D(0, "velocity", te.CC, 2), copy=False)

    indices = np.indices(dens.shape)
    xx = coords.Xc1(indices[0, :, :, :])
    yy = coords.Xc2(indices[1, :, :, :])
    rr = np.sqrt(xx**2 + yy**2)

    dens[:, :, :] = data.dens(rr)
    pres[:, :, :] = data.pres(rr)
    vel1[:, :, :] = 0.0
    vel2[:, :, :] = 0.0
    vel3[:, :, :] = 0.0


def setup(config: pyKamayan.Config, rps: pyKamayan.RuntimeParameters):
    rps.AddReal("sedov", "density", 1.0, "ambient density")
    rps.AddReal("sedov", "pressure", 1.0e-5, "ambient pressure")
    rps.AddReal("sedov", "energy", 1.0, "explosion energy")


def initialize(
    config: pyKamayan.Config, rps: pyKamayan.RuntimeParameters
) -> pyKamayan.StateDescriptor:
    pkg = pyKamayan.StateDescriptor("sedov")

    nlevels = rps.GetInt("parthenon/mesh", "numlevel")
    nx = rps.GetInt("parthenon/mesh", "nx1")
    xmin = rps.GetReal("parthenon/mesh", "x1min")
    xmax = rps.GetReal("parthenon/mesh", "x1max")
    dx = (xmax - xmin) / (2 ** (nlevels - 1) * nx)
    nu = 2.0

    radius = 3.5 * dx
    dens = rps.GetReal("sedov", "density")
    p = rps.GetReal("sedov", "pressure")
    E = rps.GetReal("sedov", "energy")
    gamma = rps.GetReal("eos/gamma", "gamma")
    pres = 3.0 * (gamma - 1.0) * E / ((nu + 1) * np.pi * radius**nu)
    data = SedovData(rho_ambient=dens, p_ambient=p, p_explosion=pres, radius=radius)
    pkg.AddParam("data", data)

    return pkg


def main():
    pman = pyKamayan.InitEnv(["sedov", "-i", "src/problems/sedov.in"])
    units = pyKamayan.ProcessUnits()
    simulation = pyKamayan.KamayanUnit("simulation")

    simulation.set_Setup(setup)
    simulation.set_Initialize(initialize)
    simulation.set_ProblemGeneratorMeshBlock(pgen)
    units.Add(simulation)
    driver = pyKamayan.InitPackages(pman, units)
    driver_status = driver.Execute()

    pman.ParthenonFinalize()


if __name__ == "__main__":
    main()
