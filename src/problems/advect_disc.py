"""Problem setup for advecting a density disc across a 2D periodic domain.

A circular region of high density is embedded in a low-density isobaric
background and advected at constant velocity. With perfect numerics the
disc translates without deformation. Numerical diffusion smears the
interface; THINC reconstruction keeps it sharp.

Inspired by FLASH's AdvectMS2D test.
"""

from dataclasses import dataclass

import numpy as np
import typer

import kamayan.pyKamayan as pyKamayan
import kamayan.pyKamayan.Grid as Grid
from kamayan.pyKamayan.Grid import TopologicalElement as te

import kamayan.kamayan_manager as kman
from kamayan.kamayan_manager import KamayanManager
from kamayan.cli import kamayan_app

from kamayan.code_units import Grid as gr, eos as eos, driver
from kamayan.code_units.Grid import UniformGrid
from kamayan.code_units.Hydro import Hydro


@dataclass
class AdvectDiscData:
    """Initial condition parameters for the advecting disc."""

    dens_in: float  # density inside disc
    dens_out: float  # density outside disc
    pressure: float  # uniform pressure (isobaric)
    velx: float  # x-velocity
    vely: float  # y-velocity
    cx: float  # disc center x
    cy: float  # disc center y
    radius: float  # disc radius


def pgen(mb: Grid.MeshBlock):
    """Single mesh block initial conditions."""
    pkg = mb.get_package("advect_disc")
    data = pkg.GetParam(AdvectDiscData, "data")

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

    rr = np.sqrt((xx - data.cx) ** 2 + (yy - data.cy) ** 2)

    dens[:, :, :] = np.where(rr <= data.radius, data.dens_in, data.dens_out)
    pres[:, :, :] = data.pressure
    vel1[:, :, :] = data.velx
    vel2[:, :, :] = data.vely
    vel3[:, :, :] = 0.0


def setup(unit: pyKamayan.KamayanUnit):
    """Setup runtime parameters for advect_disc."""
    ad = unit.AddData("advect_disc")
    ad.AddParm("dens_in", 10.0, "density inside disc")
    ad.AddParm("dens_out", 1.0, "density outside disc")
    ad.AddParm("pressure", 1.0, "uniform pressure")
    ad.AddParm("velx", 1.0, "x velocity")
    ad.AddParm("vely", 1.0, "y velocity")
    ad.AddParm("radius", 0.15, "disc radius")


def initialize(unit: pyKamayan.KamayanUnit):
    """Initialize advect_disc package data."""
    ad = unit.Data("advect_disc")

    data = AdvectDiscData(
        dens_in=ad.Get(float, "dens_in"),
        dens_out=ad.Get(float, "dens_out"),
        pressure=ad.Get(float, "pressure"),
        velx=ad.Get(float, "velx"),
        vely=ad.Get(float, "vely"),
        cx=0.5,
        cy=0.5,
        radius=ad.Get(float, "radius"),
    )
    unit.AddParam("data", data)


@kamayan_app(description="Advect a density disc in 2D with periodic BCs")
def advect_disc(
    ncells: int = typer.Option(128, help="cells per side"),
) -> KamayanManager:
    """Build KamayanManager for the advecting disc problem."""
    units = kman.process_units(
        "advect_disc", setup_params=setup, initialize=initialize, pgen=pgen
    )
    km = KamayanManager("advect_disc", units)

    km.grid = UniformGrid(
        xbnd1=(0.0, 1.0),
        xbnd2=(0.0, 1.0),
        N1=ncells,
        N2=ncells,
    )
    km.grid.boundary_conditions = gr.periodic_box()

    # One full crossing: domain is [0,1]^2, velocity = (1,1),
    # so the disc returns to its starting position at t=1.0.
    km.driver = driver.Driver(integrator="rk3", tlim=1.0)
    km.physics.hydro = Hydro(
        reconstruction="plm", riemann="hllc", cfl=0.4, slope_limiter="thinc"
    )
    km.physics.eos = eos.GammaEos(gamma=5.0 / 3.0, mode_init="dens_pres")
    km.outputs.add(
        "snapshots", "hdf5", dt=0.1, variables="dens,velocity,pres,thinc_sensor"
    )
    return km
