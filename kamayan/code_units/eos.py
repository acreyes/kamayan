"""Module to manage eos."""

from dataclasses import dataclass
from typing import Literal

from .nodes import Node
from .parameters import KamayanParam

_model = Literal["single", "tabulated", "multitype"]
_eos_mode = Literal["dens_pres", "dens_ener", "dens_temp"]


@dataclass
class KamayanEos(Node):
    """Class to manage eos inputs."""

    model = KamayanParam[str, _model]("Eos", "eos/model", "single")
    mode_init = KamayanParam[str, _eos_mode]("Eos", "eos/mode_init", "dens_pres")

    def __post_init__(self):
        """Init the node."""
        super().__init__()


@dataclass
class GammaEos(KamayanEos):
    """Single species gamma-law gas."""

    gamma = KamayanParam[float, float]("Eos", "eos/gamma/gamma", 5.0 / 3.0)
    Abar = KamayanParam[float, float]("Eos", "eos/single/Abar", 1.0)
    Zbar = KamayanParam[float, float]("Eos", "eos/single/Zbar", 1.0)

    def __post_init__(self):
        """Init KamyanEos."""
        super().__post_init__()
