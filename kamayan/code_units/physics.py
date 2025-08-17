"""Module to manage all the physics input parameters."""

from dataclasses import dataclass
from typing import Literal, Optional


from .parameters import KamayanParams
from .nodes import Node
from .Hydro import Hydro
from .eos import KamayanEos

_fluid = Literal["1t", "3t"]
_mhd = Literal["off", "ct"]


@dataclass
class KamayanPhysics(Node):
    """Class to manage all the physics modules in kamayan."""

    fluid: _fluid = "1t"
    mhd: _mhd = "off"

    def set_params(self, params: KamayanParams):
        """Set physics inputs."""
        params["physics"] = {"fluid": self.fluid, "MHD": self.mhd}

    def __post_init__(self):
        """Init the Node."""
        super().__init__()
        self._hydro: Optional[Hydro] = None
        self._eos: Optional[KamayanEos] = None

    @property
    def eos(self) -> KamayanEos:
        """Get the eos child."""
        if self._eos:
            return self._eos
        raise ValueError("eos object not set")

    @eos.setter
    def eos(self, value: KamayanEos):
        """Assign new eos object."""
        self._eos = value
        self.add_child(value)

    @property
    def hydro(self) -> Hydro:
        """Get the hydro child."""
        if self._hydro:
            return self._hydro
        raise ValueError("hydro object not set")

    @hydro.setter
    def hydro(self, value: Hydro):
        """Assign new hydro object."""
        self._hydro = value
        self.add_child(value)
