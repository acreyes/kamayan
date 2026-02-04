"""Unit for managing hydro settings."""

from dataclasses import dataclass
from typing import Literal

from kamayan.code_units.parameters import KamayanParams
from kamayan.code_units.nodes import Node

_reconstruction = Literal["fog", "plm", "ppm", "wenoz"]
_slope_limiter = Literal["minmod", "van_leer", "mc"]
_recon_vars = Literal["primitive"]
_riemann = Literal["hll", "hllc"]
_emf_method = Literal["arithmetic"]
_nghost: dict[_reconstruction, int] = {"fog": 1, "plm": 2, "ppm": 3, "wenoz": 3}


@dataclass
class Hydro(Node):
    """Hydro input parameters and settings."""

    reconstruction: _reconstruction = "fog"
    recon_vars: _recon_vars = "primitive"
    slope_limiter: _slope_limiter = "minmod"
    riemann: _riemann = "hll"
    emf_method: _emf_method = "arithmetic"
    cfl: float = 0.8

    def __post_init__(self):
        """Register our node."""
        super().__init__()

    def set_params(self, params: KamayanParams):
        """Set hydro inputs."""
        pmesh = params.get_data("grid", "parthenon/mesh")
        nghost = max(pmesh.Get(int, "nghost"), _nghost[self.reconstruction])
        pmesh["nghost"] = nghost

        params["hydro"] = {
            "reconstruction": self.reconstruction,
            "slope_limiter": self.slope_limiter,
            "riemann": self.riemann,
            "ReconstructionVars": self.recon_vars,
            "EMF_averaging": self.emf_method,
            "cfl": self.cfl,
        }
