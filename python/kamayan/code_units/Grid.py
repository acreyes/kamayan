"""Settings for Grid and mesh."""

from dataclasses import dataclass
import math
from typing import Literal

from kamayan.code_units.nodes import Node, auto_property_node
import kamayan.code_units.parameters as parms
from kamayan.code_units.parameters import KamayanParams


_resolution = None | int
_dx = None | float
refinement_strategy = Literal["none", "static", "adaptive"]


def _get_N(N: _resolution, dx: _dx, L: float) -> int:
    """Resolve the choice of resolution vs dx."""
    if (N and dx) or (not N and not dx):
        raise ValueError("Must specify only one of N or dx")

    if N:
        return N

    if dx:
        return int(L / dx)

    raise ValueError("Must specify one of N or dx")


def _mesh(
    dir: int, nx: int, xbnd: tuple[float, float] | None
) -> dict[str, parms.data_value]:
    if not xbnd:
        xbnd = (0.0, 1.0)
    return {f"nx{dir}": nx, f"x{dir}min": xbnd[0], f"x{dir}max": xbnd[1]}


def _set_mesh(
    params: KamayanParams,
    strategy: refinement_strategy,
    numlevel: int,
    mesh_params: dict[str, parms.data_value],
) -> None:
    params["parthenon/mesh"] = {
        "refinement": strategy,
        "numlevel": numlevel,
    } | mesh_params


def _set_mesh_block(params: KamayanParams, nxb1: int, nxb2: int, nxb3: int) -> None:
    params["parthenon/meshblock"] = {
        "nx1": nxb1,
        "nx2": nxb2,
        "nx3": nxb3,
    }


_boundary_conditions = Literal["outflow", "periodic", "user", "reflect"]


@dataclass
class BoundaryConditions(Node):
    """Class for boundary conditions."""

    ix1: _boundary_conditions = "outflow"
    ix2: _boundary_conditions = "outflow"
    ix3: _boundary_conditions = "outflow"
    ox1: _boundary_conditions = "outflow"
    ox2: _boundary_conditions = "outflow"
    ox3: _boundary_conditions = "outflow"

    def __post_init__(self):
        """Init node."""
        super().__init__()

    def set_params(self, params: KamayanParams):
        """Set inputs."""
        params["parthenon/mesh"] = {
            "ix1_bc": self.ix1,
            "ix2_bc": self.ix2,
            "ix3_bc": self.ix3,
            "ox1_bc": self.ox1,
            "ox2_bc": self.ox2,
            "ox3_bc": self.ox3,
        }


def periodic_box() -> BoundaryConditions:
    """Box with outflow on all sides."""
    return BoundaryConditions(
        ix1="periodic",
        ix2="periodic",
        ix3="periodic",
        ox1="periodic",
        ox2="periodic",
        ox3="periodic",
    )


def outflow_box() -> BoundaryConditions:
    """Box with outflow on all sides."""
    return BoundaryConditions(
        ix1="outflow",
        ix2="outflow",
        ix3="outflow",
        ox1="outflow",
        ox2="outflow",
        ox3="outflow",
    )


@dataclass
class KamayanGrid(Node):
    """General Kamayan grid class."""

    strategy: refinement_strategy = "none"
    nx1: int = 32
    nx2: int = 1
    nx3: int = 1
    nxb1: int = 32
    nxb2: int = 1
    nxb3: int = 1
    xbnd1: tuple[float, float] = (0.0, 1.0)
    xbnd2: tuple[float, float] | None = None
    xbnd3: tuple[float, float] | None = None
    numlevel: int = 1

    boundary_conditions = auto_property_node(BoundaryConditions, "boundary_conditions")

    def __post_init__(self):
        """Initialize the node."""
        super().__init__()

    def set_params(self, params) -> None:
        """Set the input parameters."""
        _set_mesh(
            params,
            self.strategy,
            self.numlevel,
            _mesh(1, self.nx1, self.xbnd1)
            | _mesh(2, self.nx2, self.xbnd2)
            | _mesh(3, self.nx3, self.xbnd3),
        )

        _set_mesh_block(params, self.nxb1, self.nxb2, self.nxb3)


class UniformGrid(KamayanGrid):
    """Class to manage a uniform cartesian mesh."""

    def __init__(
        self,
        xbnd1: tuple[float, float],
        xbnd2: tuple[float, float] | None = None,
        xbnd3: tuple[float, float] | None = None,
        N1: _resolution = None,
        N2: _resolution = None,
        N3: _resolution = None,
        dx1: _dx = None,
        dx2: _dx = None,
        dx3: _dx = None,
    ) -> None:
        """Set parameters for a uniform grid.

        Can specify the either the zone count or zone size for each dimension
        of the domain, but not both.

        Args:
            xbnd1: length limits of the domain
            xbnd2: length limits of the domain
            xbnd3: length limits of the domain
            N1: Zone count along dimensions of the domain
            N2: Zone count along dimensions of the domain
            N3: Zone count along dimensions of the domain
            dx1: zone size along dimensions of the domain.
            dx2: zone size along dimensions of the domain.
            dx3: zone size along dimensions of the domain.

        Raises:
            ValueError: When a dimension is over-constrained
        """
        if xbnd3 and not xbnd2:
            raise ValueError("x3 dimension provided, but no x2 dimensions")

        ndim = 1 + (1 if xbnd2 else 0) + (1 if xbnd3 else 0)
        self.ndim = ndim

        nx1 = _get_N(N1, dx1, xbnd1[1] - xbnd1[0])
        nx2 = 1 if not xbnd2 else _get_N(N2, dx2, xbnd2[1] - xbnd2[0])
        nx3 = 1 if not xbnd3 else _get_N(N3, dx3, xbnd3[1] - xbnd3[0])

        # try to figure out how to decompose the domain
        import mpi4py

        num_procs = mpi4py.MPI.COMM_WORLD.Get_size()
        nxb1 = num_procs
        nxb2 = 1
        nxb3 = 1
        if ndim == 2:
            # np = np1*np2
            # np1 = 2*np2 => np = 2*np2^2 => np2 = sqrt(np/2)
            nxb2 = int(math.sqrt(num_procs / 2))
            nxb1 = nxb2 * 2
        elif ndim == 3:
            # np1 = 2*np2 = 2*np3
            # np = np1*np2*np3 = 2*np2^3 => np2 = cbrt(np/2)
            nxb2 = int(math.cbrt(num_procs / 2))
            nxb3 = nxb2
            nxb1 = 2 * nxb2

        nxb1 = int(nx1 / nxb1)
        nxb2 = int(nx2 / nxb2)
        nxb3 = int(nx3 / nxb3)

        super().__init__(
            strategy="none",
            nx1=nx1,
            nx2=nx2,
            nx3=nx3,
            nxb1=nxb1,
            nxb2=nxb2,
            nxb3=nxb3,
            xbnd1=xbnd1,
            xbnd2=xbnd2,
            xbnd3=xbnd3,
        )


_refinement_method = Literal["loehner", "derivative_order_1", "derivative_order_2"]


@dataclass
class RefinementVariable:
    """Class for refinement fields."""

    field: str
    max_level: int | None = None
    method: _refinement_method = "loehner"
    filter: float = 1.0e-2
    derefine_tol: float = 2.0e-1
    refine_tol: float = 8.0e-1

    def set_params(self, nref_var: int, params: KamayanParams) -> None:
        """Set the parameters for kamayan/refinement#nref_var."""
        max_level = (
            self.max_level
            if self.max_level
            else params.get_data("grid", "parthenon/mesh").Get(int, "numlevel")
        )
        key = f"kamayan/refinement{nref_var}"
        params[key] = {
            "field": self.field,
            "method": self.method,
            "derefine_tol": self.derefine_tol,
            "refine_tol": self.refine_tol,
            "filter": self.filter,
            "max_level": max_level,
        }


class RefinementCollection(Node):
    """Collection of fields to refine on."""

    def __init__(self):
        """Initialize the collection."""
        super().__init__()
        self._fields: dict[str, RefinementVariable] = {}

    def add(
        self,
        field: str,
        max_level: int | None = None,
        method: _refinement_method = "loehner",
        filter: float = 1.0e-2,
        derefine_tol: float = 2.0e-1,
        refine_tol: float = 8.0e-1,
    ):
        """Add a new field to refine on.

        Args:
            field: name of the field to refine on
            max_level: maximum level to refine on this field up to
            method: method to tag zones for refinement
            filter: used by Loehner method to filter high freq. noise
            derefine_tol: error tolerance to trigger derefinement
            refine_tol: float = error tolerance to trigger refinement
        """
        self._fields[field] = RefinementVariable(
            field,
            max_level=max_level,
            method=method,
            filter=filter,
            derefine_tol=derefine_tol,
            refine_tol=refine_tol,
        )

    def __post_init__(self):
        """Initialize the node."""
        super().__init__()

    def clear(self):
        """Clear the collection."""
        self._fields.clear()

    def pop(self, key: str) -> RefinementVariable:
        """Pop key from collection."""
        return self._fields.pop(key)

    def set_params(self, params: KamayanParams) -> None:
        """Set input parameters for our refinement vars."""
        n = 0
        for rv in self._fields.values():
            rv.set_params(n, params)
            n += 1


class AdaptiveGrid(KamayanGrid):
    """Class to manage an adaptive grid."""

    def __init__(
        self,
        xbnd1: tuple[float, float],
        nxb1: int,
        xbnd2: tuple[float, float] | None = None,
        nxb2: int | None = None,
        xbnd3: tuple[float, float] | None = None,
        nxb3: int | None = None,
        nblocks1: int | None = None,
        nblocks2: int | None = None,
        nblocks3: int | None = None,
        dx1: _dx = None,
        dx2: _dx = None,
        dx3: _dx = None,
        num_levels: int = 1,
    ) -> None:
        """Set parameters for a uniform grid.

        Can specify the either the block count at root level or zone size for each dimension
        of the domain at highest refinement level, but not both.

        Args:
            xbnd1: length limits of the domain
            xbnd2: length limits of the domain
            xbnd3: length limits of the domain
            nxb1: zone count along dimensions of a single meshblock
            nxb2: zone count along dimensions of a single meshblock
            nxb3: zone count along dimensions of a single meshblock
            nblocks1: block count along dimensions of the domain at lowest resolution
            nblocks2: block count along dimensions of the domain at lowest resolution
            nblocks3: block count along dimensions of the domain at lowest resolution
            dx1: zone size along dimensions of the domain at highest resoltution.
            dx2: zone size along dimensions of the domain at highest resoltution.
            dx3: zone size along dimensions of the domain at highest resoltution.
            num_levels: number of total amr refinement levels

        Raises:
            ValueError: When a dimension is over-constrained
        """
        if xbnd3 and not xbnd2:
            raise ValueError("x3 dimension provided, but no x2 dimensions")

        ndim = 1 + (1 if xbnd2 else 0) + (1 if xbnd3 else 0)
        self.ndim = ndim
        self.num_levels = num_levels
        self._refinement_fields: RefinementCollection | None = None

        # get the zone count at the lowest refinement that matches either
        # the provided block size or desired length scale resolution
        ratio_highest_lowest_refinement = 2 ** (num_levels - 1)
        Ns = [
            nblock * nxb if nblock else None
            for nblock, nxb in [(nblocks1, nxb1), (nblocks2, nxb2), (nblocks3, nxb3)]
        ]
        # move to scale at lowest refinement level
        dx1 = dx1 * ratio_highest_lowest_refinement if dx1 else None
        dx2 = dx2 * ratio_highest_lowest_refinement if dx2 else None
        dx3 = dx3 * ratio_highest_lowest_refinement if dx3 else None
        nx1 = _get_N(Ns[0], dx1, xbnd1[1] - xbnd1[0])
        nx2 = 1 if not xbnd2 else _get_N(Ns[1], dx2, xbnd2[1] - xbnd2[0])
        nx3 = 1 if not xbnd3 else _get_N(Ns[2], dx3, xbnd3[1] - xbnd3[0])
        nxb2 = nxb2 if nxb2 else 1
        nxb3 = nxb3 if nxb3 else 1

        super().__init__(
            strategy="adaptive",
            numlevel=self.num_levels,
            nx1=nx1,
            nx2=nx2,
            nx3=nx3,
            nxb1=nxb1,
            nxb2=nxb2,
            nxb3=nxb3,
            xbnd1=xbnd1,
            xbnd2=xbnd2,
            xbnd3=xbnd3,
        )
        self.refinement_fields = RefinementCollection()

    @property
    def refinement_fields(self) -> RefinementCollection:
        """Get the refinement_fields child."""
        if self._refinement_fields:
            return self._refinement_fields
        raise ValueError("refinement_fields object not set")

    @refinement_fields.setter
    def refinement_fields(self, value: RefinementCollection):
        """Assign new refinement_fields object."""
        self._refinement_fields = value
        self.add_child(value)

    def set_params(self, params) -> None:
        """Set input parameters for the grid and our refinement fields."""
        super().set_params(params)
