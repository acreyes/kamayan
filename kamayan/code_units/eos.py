"""Module to manage eos."""

from dataclasses import dataclass
from typing import Literal


from .nodes import Node
from .parameters import KamayanParams

_model = Literal["single", "tabulated", "multitype"]
_eos_mode = Literal["dens_pres", "dens_ener", "dens_temp"]


@dataclass
class KamayanEos(Node):
    """Class to manage eos inputs."""

    model: _model = "single"
    mode_init: _eos_mode = "dens_pres"

    def __post_init__(self):
        """Init the node."""
        super().__init__()

    def set_params(self, params: KamayanParams):
        """Set Eos inputs."""
        params["eos"] = {"mode_init": self.mode_init, "model": self.model}


@dataclass
class GammaEos(KamayanEos):
    """Single species gamma-law gas."""

    gamma: float = 5.0 / 3.0
    Abar: float = 1.0
    Zbar: float = 1.0

    def __post_init__(self):
        """Init KamyanEos."""
        super().__post_init__()

    def set_params(self, params: KamayanParams):
        """Set gamma law input parameters."""
        params["eos/single"] = {"Abar": self.Abar, "Zbar": self.Zbar}
        params["eos/gamma"] = {"gamma": self.gamma}
