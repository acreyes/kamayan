from pathlib import Path


class RuntimeParametersBlock:
    """Class to hold runtime parameters for a single block in an input file."""

    def __init__(
        self, block: str, parms: None | dict[str, int | float | str] = None
    ) -> None:
        """Initialize the Runtime Parameters for a given block."""
        self.block = block
        self.parms = {}
        if parms:
            self.parms = parms

    def __getitem__(self, key: str) -> str | int | float:
        """Accessor for the parms on this block."""
        return self.parms[key]

    def __setitem__(self, key: str, value: str | int | float) -> None:
        """Accessor for the parms on this block."""
        # TODO(acreyes): would be great to pass this throuh a RuntimeParameters
        # object to verify the value
        self.parms[key] = value

    def write_input(self, file_handle) -> None:
        """Write our block of parms to the file."""
        file_handle.write(f"<{self.block}>\n")
        parms = [f"{key} = {value}\n" for key, value in self.parms.items()]
        file_handle.write("".join(parms))


class InputParameters:
    """Class to hold all runtime parameter blocks."""

    def __init__(self, input_file: Path):
        """Initialize the input parameter file."""
        self.blocks: set[RuntimeParametersBlock] = set()
        self.input_file = input_file

    def add(self, block: RuntimeParametersBlock) -> None:
        """Add a new block to the set."""
        self.blocks.add(block)

    def write(self) -> None:
        """Write the input file."""
        with open(self.input_file, "w") as fid:
            for block in self.blocks:
                block.write_input(fid)
