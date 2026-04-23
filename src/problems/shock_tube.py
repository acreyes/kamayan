"""Problem setup for a shock tube with initial left/right states."""

from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np
import typer

from kamayan.cli.app import kamayan_app
from kamayan.code_units import Grid, driver, eos, physics
from kamayan.code_units.Grid import AdaptiveGrid
import kamayan.pyKamayan as pyKamayan
from kamayan.pyKamayan.Grid import GenericCoordinate, TopologicalElement as te
from kamayan.code_units.Hydro import Hydro
import kamayan.kamayan_manager as kman
from kamayan.kamayan_manager import KamayanManager

from kamayan.pyKamayan.Grid import GetConfig
from kamayan.pyKamayan.Options import Mhd

_PROBLEMS = Literal["sod", "briowu", "einfeldt"]
_DIMENSION = Literal[1, 2]  # support work out in 3d

mhd_problems: list[_PROBLEMS] = ["briowu"]


def _get_unit_dirs(
    ndim: float, aspect12: int, aspect13: int, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a12 = 1.0 / (float(aspect12) if ndim > 1 else np.inf)
    # a13 = 1.0 / (float(aspect13) if ndim > 2 else np.inf)
    denom = np.sqrt(1 + a12**2) if normalize else 1.0

    n_perp = np.array([1.0, a12, 0.0]) / denom
    n_par1 = np.array([-a12, 1.0, 0.0]) / denom
    n_par2 = [0.0, 0.0, 1.0]
    return np.array(n_perp), np.array(n_par1), np.array(n_par2)


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
        n_perp, n_par1, n_par2 = _get_unit_dirs(
            ndim, aspect12, aspect13, normalize=True
        )

        def _get_state(s: Literal["L", "R"]) -> State:
            # rotate states from the reference of the interface
            vel_perp = shock_tube.Get(float, f"vel1{s}")
            vel_par1 = shock_tube.Get(float, f"vel2{s}")
            vel_par2 = shock_tube.Get(float, f"vel3{s}")
            mag_perp = shock_tube.Get(float, f"mag1{s}")
            mag_par1 = shock_tube.Get(float, f"mag2{s}")
            mag_par2 = shock_tube.Get(float, f"mag3{s}")

            return State(
                dens=shock_tube.Get(float, f"dens{s}"),
                vel1=vel_perp * n_perp[0] + vel_par1 * n_par1[0] + vel_par2 * n_par2[0],
                vel2=vel_perp * n_perp[1] + vel_par1 * n_par1[1] + vel_par2 * n_par2[1],
                vel3=vel_perp * n_perp[2] + vel_par1 * n_par1[2] + vel_par2 * n_par2[2],
                pres=shock_tube.Get(float, f"pres{s}"),
                mag1=mag_perp,
                mag2=mag_par1,
                mag3=mag_par2,
            )

        unit.AddParam("vL", _get_state("L"))
        unit.AddParam("vR", _get_state("R"))

    return _initialize


def _get_coords(
    indices: np.ndarray, coords: GenericCoordinate, el: te = te.CC
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = np.array(coords.Xc1(indices[2, :, :, :]))
    yy = np.array(coords.Xc2(indices[1, :, :, :]))
    zz = np.array(coords.Xc2(indices[0, :, :, :]))
    if el == te.F1:
        xx = np.array(coords.Xf1(indices[2, :, :, :]))
    elif el == te.F2:
        yy = np.array(coords.Xf2(indices[1, :, :, :]))
    elif el == te.F3:
        zz = np.array(coords.Xf3(indices[0, :, :, :]))

    return xx, yy, zz


def _x_perp(
    n_perp: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    wrap: bool = True,
) -> np.ndarray:
    x_perp = n_perp[0] * xx + n_perp[1] * yy + n_perp[2] * zz
    if wrap:
        return x_perp % 1.0
    return x_perp


def _calc_face_mag(
    face: np.ndarray,
    el: te,
    coords: GenericCoordinate,
    n_perp: np.ndarray,
    n_par1: np.ndarray,
    vL: State,
    vR: State,
) -> np.ndarray:
    if el not in [te.F1, te.F2, te.F3]:
        raise ValueError(f"Topological Element {el} not a face")
    # coordinates at the face el
    xx, yy, zz = _get_coords(np.indices(face.shape), coords, el)
    # shift by delta / 2 to the edges
    if el == te.F1:
        yy -= 0.5 * coords.Dx2()
    if el == te.F2:
        xx -= 0.5 * coords.Dx1()

    def Az(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        x_perp = _x_perp(n_perp, x, y, z)
        x_par1 = _x_perp(n_par1, x, y, z, wrap=False)
        ## we need for Az to be continuous at x_per == 0.5
        return np.where(
            x_perp <= 0.5,
            -(x_perp - 0.5) * vL.mag2 + x_par1 * vL.mag1,
            -(x_perp - 0.5) * vR.mag2 + x_par1 * vR.mag1,
        )

    if el == te.F1:
        return (Az(xx, yy + coords.Dx2(), zz) - Az(xx, yy, zz)) / coords.Dx2()
    elif el == te.F2:
        return -(Az(xx + coords.Dx1(), yy, zz) - Az(xx, yy, zz)) / coords.Dx1()

    return face


def pgen(mb: pyKamayan.Grid.MeshBlock):
    """Single block initial conditions for shock tube."""
    shock_tube = mb.get_package("shock_tube")
    vL = shock_tube.GetParam(State, "vL")
    vR = shock_tube.GetParam(State, "vR")

    ndim = shock_tube.GetParam(int, "ndim")
    aspect12 = shock_tube.GetParam(int, "apsect12")
    aspect13 = shock_tube.GetParam(int, "apsect13")

    n_perp, n_par1, n_par2 = _get_unit_dirs(ndim, aspect12, aspect13, normalize=True)

    config = GetConfig(mb)
    mhd = config.GetMhd()
    ct = mhd == Mhd.ct

    vars = ["dens", "pres", "velocity"]
    vars += ["mag", "magc"] if ct else []
    pack = mb.pack(vars)
    coords = pack.GetCoordinates(0)

    dens = np.array(pack.GetParArray3D(0, "dens", te.CC).view(), copy=False)
    pres = np.array(pack.GetParArray3D(0, "pres", te.CC).view(), copy=False)
    vel1 = np.array(pack.GetParArray3D(0, "velocity", te.CC).view(), copy=False)
    vel2 = np.array(pack.GetParArray3D(0, "velocity", te.CC, 1).view(), copy=False)
    vel3 = np.array(pack.GetParArray3D(0, "velocity", te.CC, 2).view(), copy=False)

    xx, yy, zz = _get_coords(np.indices(dens.shape), coords)

    x_perp = _x_perp(n_perp, xx, yy, zz)

    dens[:, :, :] = np.where(x_perp <= 0.5, vL.dens, vR.dens)
    vel1[:, :, :] = np.where(x_perp <= 0.5, vL.vel1, vR.vel1)
    vel2[:, :, :] = np.where(x_perp <= 0.5, vL.vel2, vR.vel2)
    vel3[:, :, :] = np.where(x_perp <= 0.5, vL.vel3, vR.vel3)
    pres[:, :, :] = np.where(x_perp <= 0.5, vL.pres, vR.pres)

    if ct:
        mag1 = np.array(pack.GetParArray3D(0, "magc", te.CC, 0).view(), copy=False)
        mag2 = np.array(pack.GetParArray3D(0, "magc", te.CC, 1).view(), copy=False)
        mag3 = np.array(pack.GetParArray3D(0, "magc", te.CC, 2).view(), copy=False)
        mag1[:, :, :] = np.where(x_perp <= 0.5, vL.mag1, vR.mag1)
        mag2[:, :, :] = np.where(x_perp <= 0.5, vL.mag2, vR.mag2)
        mag3[:, :, :] = np.where(x_perp <= 0.5, vL.mag3, vR.mag3)

        if ndim == 1:
            assert vL.mag1 == vR.mag1, "1D requires constant magnetic field."
        fac1 = np.array(pack.GetParArray3D(0, "mag", te.F1).view(), copy=False)
        fac1[:, :, :] = vL.mag1

        if ndim > 1:
            fac2 = np.array(pack.GetParArray3D(0, "mag", te.F2).view(), copy=False)
            fac1[:, :, :] = _calc_face_mag(fac1, te.F1, coords, n_perp, n_par1, vL, vR)
            fac2[:, :, :] = _calc_face_mag(fac2, te.F2, coords, n_perp, n_par1, vL, vR)


@kamayan_app(description="Shock Tube with initial left/right states")
def shock_tube(
    problem: _PROBLEMS = typer.Option(
        "sod",
        help="""Initial configuration.
                 - sod: hydrodynamic shock tube
                 - briowu: mhd shock tube
                 - einfeldt: strong rarefaction.""",
    ),
    ndim: _DIMENSION = typer.Option(
        1, help="""Dimension. When >1 uses rotated with periodic boundaries."""
    ),
    aspect12: int = typer.Option(
        1, help="Aspect ratio to use for rotated shock tube. x2/x1"
    ),
    aspect13: int = typer.Option(
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

    nxb = 8
    nblocks = int(16 / nxb)
    mult = 2.0 if ndim > 1 else 1.0
    L = np.sqrt(aspect12**2 + 1) / (2 * aspect12) if ndim > 1 else 1.0
    km.grid = AdaptiveGrid(
        xbnd1=(0.0, L * mult),
        xbnd2=(0.0, L * mult * aspect12),
        xbnd3=(0.0, L * mult * aspect13),
        nxb1=nxb,
        nxb2=nxb if ndim > 1 else 1,
        nxb3=nxb if ndim > 2 else 1,
        nblocks1=nblocks,
        nblocks2=nblocks * aspect12 if ndim > 1 else 1,
        nblocks3=nblocks * aspect13 if ndim > 2 else 1,
        num_levels=4,
    )

    km.grid.refinement_fields.add("dens")
    km.grid.boundary_conditions = (
        Grid.periodic_box() if ndim > 1 else Grid.outflow_box()
    )

    km.driver = driver.Driver(integrator="rk2", tlim=0.1)
    km.outputs.add("restarts", "rst", dt=0.01)
    gamma = 2.0 if problem == "briowu" else 1.4
    km.physics.eos = eos.GammaEos(gamma=gamma, mode_init="dens_pres")
    km.physics.hydro = Hydro(reconstruction="wenoz", riemann="hllc")
    km.physics.mhd = mhd
    return km
