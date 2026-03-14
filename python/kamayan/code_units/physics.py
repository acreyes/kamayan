"""Module to manage all the physics input parameters."""

from dataclasses import dataclass
from typing import Literal, Optional


from .parameters import KamayanParams
from .nodes import Node, auto_property_node
from .Hydro import Hydro
from .eos import KamayanEos

FLUID = Literal["1t", "3t"]
MHD = Literal["off", "ct"]


@dataclass
class KamayanPhysics(Node):
    """Class to manage all the physics modules in kamayan."""

    fluid: FLUID = "1t"
    mhd: MHD = "off"

    eos = auto_property_node(KamayanEos, "eos")
    hydro = auto_property_node(Hydro, "hydro")

    def set_params(self, params: KamayanParams):
        """Set physics inputs."""
        params["physics"] = {"fluid": self.fluid, "MHD": self.mhd}

    def __post_init__(self):
        """Init the Node."""
        super().__init__()
        self._hydro: Optional[Hydro] = None
        self._eos: Optional[KamayanEos] = None
