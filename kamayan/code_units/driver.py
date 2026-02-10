"""Module for driver parameters."""

from dataclasses import dataclass, asdict
import sys
from typing import Literal

from .nodes import Node
from .parameters import KamayanParams

_integrators = Literal["rk1", "rk2", "rk3"]
_BIG = 1.0e300


@dataclass
class Driver(Node):
    """Class to manage driver inputs."""

    integrator: _integrators = "rk2"
    nlim: int = -1
    tlim: float = 0.0
    ncycle_out_mesh: int = -10000
    dt_force: float = -_BIG
    dt_factor: float = 2.0
    dt_ceil: float = _BIG
    dt_min: float = 0.0
    dt_min_cycle_limit: int = 10
    dt_init: float = _BIG
    dt_init_force: bool = False

    def __post_init__(self):
        """Init the node."""
        super().__init__()

    def set_params(self, params: KamayanParams):
        """Set our inputs."""
        params["parthenon/time"] = asdict(self)
