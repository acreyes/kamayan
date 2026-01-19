"""Kamayan main CLI entry point.

Usage:
    kamayan <script_or_module> [--dry-run] [parthenon_args...]

Examples:
    kamayan ./my_simulation.py --dry-run
    kamayan ./my_simulation.py -r restart.dat
    kamayan my_package.my_simulation
    kamayan ./my_simulation.py parthenon/time/nlim=100
"""

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from kamayan.cli.utils import load_simulation

if TYPE_CHECKING:
    pass


app = typer.Typer(
    help="Kamayan simulation manager",
    rich_markup_mode="markdown",
    no_args_is_help=True,
)


@app.command("run")
def run_command(
    script: Path = typer.Argument(
        ...,
        help="Python script path (e.g., ./sim.py) or module path (e.g., mypackage.mysim)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Generate input file without executing"
    ),
    parthenon_args: list[str] = typer.Argument(
        None,
        help="Arguments forwarded to Parthenon (e.g., `parthenon/time/nlim=100`, `-r restart.rhdf`)",
    ),
):
    """Run a Kamayan simulation from a script or module.

    Additional arguments after the script name are forwarded directly to Parthenon.

    **Parthenon Arguments:**

    - `-r <file>` - Restart from checkpoint file
    - `-a <file>` - Analyze/postprocess file
    - `-d <directory>` - Set run directory
    - `-t hh:mm:ss` - Set wall time limit
    - `-n` - Parse input and quit (dry run)
    - `-m <nproc>` - Output mesh structure and quit
    - `-c` - Show configuration and quit
    - `-h` - Show Parthenon help
    - `block/param=value` - Override input parameters

    **Examples:**

    ```bash
    # Basic run
    kamayan ./my_sim.py

    # Override number of cycles
    kamayan ./my_sim.py parthenon/time/nlim=100

    # Restart from checkpoint
    kamayan ./my_sim.py -r output.00050.rhdf

    # Run with custom run directory
    kamayan ./my_sim.py -d /scratch/run01

    # Multiple parameter overrides
    kamayan ./my_sim.py parthenon/time/nlim=50 parthenon/time/tlim=0.5
    ```
    """
    try:
        sim_func = load_simulation(str(script))
        km = sim_func()
    except Exception as e:
        typer.echo(f"Error loading simulation: {e}", err=True)
        raise typer.Exit(1)

    if dry_run:
        km.write_input()
        typer.echo(f"Input file written to: {km.input_file}")
    else:
        km.execute(*(parthenon_args or []))


def main():
    """Entry point for the kamayan CLI."""
    app()


if __name__ == "__main__":
    app()
