"""Problem setup for a shock tube with initial left/right states."""

from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np
import typer

from kamayan.cli.app import kamayan_app
from kamayan.code_units import Grid, driver, eos, physics
from kamayan.code_units.Grid import AdaptiveGrid
import kamayan.pyKamayan as pyKamayan
from kamayan.pyKamayan.Grid import TopologicalElement as te
from kamayan.code_units.Hydro import Hydro
import kamayan.kamayan_manager as kman
from kamayan.kamayan_manager import KamayanManager

from kamayan.pyKamayan.Grid import GetConfig
from kamayan.pyKamayan.Options import Mhd

_PROBLEMS = Literal["sod", "briowu", "einfeldt"]
_DIMENSION = Literal[1, 2, 3]

mhd_problems: list[_PROBLEMS] = ["briowu"]


@dataclass
class State:
    """Constant fluid state."""

    dens: float = 1.0
    vel1: float = 0.0
    vel2: float = 0.0
    vel3: float = 0.0
    pres: float = 1.0
    mag1: float = 0.0
    mag2: float = 0.0
    mag3: float = 0.0


def setup(problem: _PROBLEMS) -> Callable[[pyKamayan.KamayanUnit], None]:
    """Build the setup.

    Uses states from main function to define the default input parameters.
    """
    if problem == "sod":
        vL = State(dens=1.0, pres=1.0)
        vR = State(dens=0.125, pres=0.1)
    elif problem == "briowu":
        vL = State(dens=1.0, pres=1.0, mag1=0.75, mag2=1.0)
        vR = State(dens=0.125, pres=0.1, mag1=0.75, mag2=-1.0)
    elif problem == "einfeldt":
        vL = State(vel1=-1.0)
        vR = State(vel1=1.0)
    else:
        raise NotImplementedError(f"problem {problem} not available")

    def _setup(unit: pyKamayan.KamayanUnit):
        shock_tube = unit.AddData("shock_tube")
        shock_tube.AddParm("densL", vL.dens, "left state")
        shock_tube.AddParm("vel1L", vL.vel1, "left state")
        shock_tube.AddParm("vel2L", vL.vel2, "left state")
        shock_tube.AddParm("vel3L", vL.vel3, "left state")
        shock_tube.AddParm("presL", vL.pres, "left state")
        shock_tube.AddParm("mag1L", vL.mag1, "left state")
        shock_tube.AddParm("mag2L", vL.mag2, "left state")
        shock_tube.AddParm("mag3L", vL.mag3, "left state")

        shock_tube.AddParm("densR", vR.dens, "right state")
        shock_tube.AddParm("vel1R", vR.vel1, "right state")
        shock_tube.AddParm("vel2R", vR.vel2, "right state")
        shock_tube.AddParm("vel3R", vR.vel3, "right state")
        shock_tube.AddParm("presR", vR.pres, "right state")
        shock_tube.AddParm("mag1R", vR.mag1, "right state")
        shock_tube.AddParm("mag2R", vR.mag2, "right state")
        shock_tube.AddParm("mag3R", vR.mag3, "right state")

    return _setup


def initialize(
    ndim: _DIMENSION, aspect12: int, aspect13: int
) -> Callable[[pyKamayan.KamayanUnit], None]:
    """Initialize shock tube packaga deta."""

    def _initialize(unit: pyKamayan.KamayanUnit):
        unit.AddParam("ndim", ndim)
        unit.AddParam("apsect12", aspect12)
        unit.AddParam("apsect13", aspect13)

        shock_tube = unit.Data("shock_tube")

        def _get_state(s: Literal["L", "R"]) -> State:
            return State(
                dens=shock_tube.Get(float, f"dens{s}"),
                vel1=shock_tube.Get(float, f"vel1{s}"),
                vel2=shock_tube.Get(float, f"vel3{s}"),
                vel3=shock_tube.Get(float, f"vel3{s}"),
                pres=shock_tube.Get(float, f"pres{s}"),
                mag1=shock_tube.Get(float, f"mag1{s}"),
                mag2=shock_tube.Get(float, f"mag3{s}"),
                mag3=shock_tube.Get(float, f"mag3{s}"),
            )

        unit.AddParam("vL", _get_state("L"))
        unit.AddParam("vR", _get_state("R"))

    return _initialize


def pgen(mb: pyKamayan.Grid.MeshBlock):
    """Single block initial conditions for shock tube."""
    shock_tube = mb.get_package("shock_tube")
    vL = shock_tube.GetParam(State, "vL")
    vR = shock_tube.GetParam(State, "vR")

    # config = GetConfig(mb)
    # mhd = config.GetMhd()
    # ct = mhd == Mhd.ct
    ct = False

    vars = ["dens", "pres", "velocity"]
    vars += ["mag"] if ct else []
    pack = mb.pack(vars)
    coords = pack.GetCoordinates(0)

    dens = np.array(pack.GetParArray3D(0, "dens", te.CC).view(), copy=False)
    pres = np.array(pack.GetParArray3D(0, "pres", te.CC).view(), copy=False)
    vel1 = np.array(pack.GetParArray3D(0, "velocity", te.CC).view(), copy=False)
    vel2 = np.array(pack.GetParArray3D(0, "velocity", te.CC, 1).view(), copy=False)
    vel3 = np.array(pack.GetParArray3D(0, "velocity", te.CC, 2).view(), copy=False)

    indices = np.indices(dens.shape)
    xx = np.array(coords.Xc1(indices[2, :, :, :]))
    yy = np.array(coords.Xc2(indices[1, :, :, :]))

    dens[:, :, :] = np.where(xx <= 0.5, vL.dens, vR.dens)
    vel1[:, :, :] = np.where(xx <= 0.5, vL.vel1, vR.vel1)
    vel2[:, :, :] = np.where(xx <= 0.5, vL.vel2, vR.vel2)
    vel3[:, :, :] = np.where(xx <= 0.5, vL.vel3, vR.vel3)
    pres[:, :, :] = np.where(xx <= 0.5, vL.pres, vR.pres)


@kamayan_app(description="Shock Tube with initial left/right states")
def shock_tube(
    problem: _PROBLEMS = typer.Argument(
        "sod",
        help="""Initial configuration.
                 - sod: hydrodynamic shock tube
                 - briowu: mhd shock tube
                 - einfeldt: strong rarefaction.""",
    ),
    ndim: _DIMENSION = typer.Argument(
        1, help="""Dimension. When >1 uses rotated with periodic boundaries."""
    ),
    aspect12: int = typer.Argument(
        1, help="Aspect ratio to use for rotated shock tube. x2/x1"
    ),
    aspect13: int = typer.Argument(
        1, help="Aspect ratio to use for rotated shock tube. x3/x1"
    ),
) -> KamayanManager:
    """Build KamayanManager for the shock tube problem."""
    mhd: physics.MHD = "ct" if problem in mhd_problems else "off"

    units = kman.process_units(
        "shock_tube",
        setup_params=setup(problem),
        initialize=initialize(ndim, aspect12, aspect13),
        pgen=pgen,
    )
    km = KamayanManager("shock_tube", units)

    nxb = 128
    nblocks = int(128 / nxb)
    km.grid = AdaptiveGrid(
        xbnd1=(0.0, 1.0),
        xbnd2=(0.0, 1.0 * aspect12),
        xbnd3=(0.0, 1.0 * aspect13),
        nxb1=nxb,
        nxb2=nxb if ndim > 1 else 1,
        nxb3=nxb if ndim > 2 else 1,
        nblocks1=nblocks,
        nblocks2=nblocks * aspect12 if ndim > 1 else 1,
        nblocks3=nblocks * aspect13 if ndim > 2 else 1,
    )

    km.grid.refinement_fields.add("dens")
    km.grid.boundary_conditions = (
        Grid.periodic_box() if ndim > 1 else Grid.outflow_box()
    )

    km.driver = driver.Driver(integrator="rk2", tlim=0.1)
    km.outputs.add("restarts", "rst", dt=0.01)
    km.physics.eos = eos.GammaEos(gamma=1.4, mode_init="dens_pres")
    km.physics.hydro = Hydro(reconstruction="wenoz", riemann="hllc")
    km.physics.mhd = mhd
    return km
