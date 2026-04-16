"""Module to manage parthenon outputs."""

from dataclasses import dataclass, asdict
from typing import Optional

from .parameters import KamayanParams
from .nodes import Node


@dataclass
class OutputType:
    """Class for describing a parthenon output."""

    name: str
    file_type: str
    dt: Optional[float] = None
    dn: Optional[int] = None
    variables: Optional[str] = None

    def set_params(self, n_out: int, params: KamayanParams):
        """Set our input parameters."""
        self.dt = self.dt if self.dt else -1.0
        self.dn = self.dn if self.dn else -1
        d = {k: v for k, v in asdict(self).items() if v is not None}
        params[f"parthenon/output{n_out}"] = d


class KamayanOutputs(Node):
    """Class for output input parameters."""

    def __init__(self):
        """Initialize empty list of outputs."""
        super().__init__()
        self._outputs: dict[str, OutputType] = {}

    def clear(self):
        """Clear all registerd outputs."""
        self._outputs.clear()

    def add(
        self,
        name,
        file_type: str,
        dt: Optional[float] = None,
        dn: Optional[int] = None,
        variables: Optional[str] = None,
    ):
        """Add a new output."""
        if dt and dn:
            raise ValueError("Can only provide one of dt or dn for a given output")

        self._outputs[name] = OutputType(
            name=name, file_type=file_type, dt=dt, dn=dn, variables=variables
        )

    def set_params(self, params: KamayanParams):
        """Set inputs for all outputs."""
        n = 0
        for op in self._outputs.values():
            op.set_params(n, params)
            n += 1
